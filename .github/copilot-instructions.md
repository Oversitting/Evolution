# Evolution Simulator - Copilot Instructions

## Project Overview

**Evolution Simulator** is a GPU-accelerated 2D evolution simulator where digital organisms with neural network brains evolve through natural selection. Built in Rust using wgpu (GPU API), winit (windowing), and egui (UI).

### Core Purpose
Simulate thousands of organisms in parallel, each with:
- A 20→16→6 neural network brain controlling behavior (sense, think, act pipeline)
- Energy management (drain, movement costs, food consumption)
- Reproduction with genetic inheritance and mutation
- Real-time visualization on GPU

---

## Architecture & Data Flows

### High-Level Game Loop (see [src/main.rs](../src/main.rs) and [src/app.rs](../src/app.rs))

```
Input (winit events)
    ↓
App state updates (pause, zoom, UI)
    ↓
Simulation Step:
    1. Readback GPU state (CPU gets energy/pos from PREV tick)
    2. Reproduction Logic (CPU deducts energy, spawns new)
    3. Sync to GPU (Upload changes)
    4. Dispatch Compute (GPU runs sense → think → act)
    ↓
Render & UI (render world + organisms + egui panels)
    ↓
Display frame
```

### Module Boundaries

| Module | Responsibility | Key Files |
|--------|---|---|
| **simulation** | CPU logic: organism pools, genome management, reproduction, world state | `simulation/organism.rs`, `genome.rs`, `world.rs` |
| **compute** | GPU parallel execution: buffers management, compute shaders dispatch | `compute/pipeline.rs`, `buffers.rs` |
| **render** | GPU rendering: organism triangles, food textures, UI layer | `render/mod.rs`, `camera.rs` |
| **ui** | egui overlay: HUD, stats graphs, inspector, founder browser/editor, controls, theme | `ui/mod.rs`, `ui/stats.rs`, `ui/inspector.rs`, `ui/theme.rs` |
| **config** | All tunable parameters (serializable to TOML/JSON) | `config.rs` |

### GPU Compute Pipeline (the "sense→think→act" loop)

Three compute shaders dispatch in sequence—see `src/compute/shaders/`:

1. **sense.wgsl**: Cast 8 vision rays per organism, detect food/organisms in range
2. **think.wgsl**: Feed sensory input (20 floats) through 2-layer neural network
3. **act.wgsl**: Output control signals (move, rotate, eat, reproduce) + update position/energy

Each shader reads organism + genome state, writes updated organism state. RNG is seeded from organism ID for reproducibility.

**Critical detail**: Genome weights stored separately in `GpuBuffers` ([compute/buffers.rs](../src/compute/buffers.rs)) as:
- `weights_1`, `biases_1`: input→hidden layer (320+16 floats per organism)
- `weights_2`, `biases_2`: hidden→output layer (96+6 floats per organism)

---

## Key Data Structures

### Organism (64+ bytes, CPU+GPU sync in [simulation/organism.rs](../src/simulation/organism.rs))
```rust
position: [f32; 2]       // World coords (wraps at edges)
velocity: [f32; 2]       // Current velocity
rotation: f32            // 0 = right, π/2 = up
energy: f32              // 0-200 (max configurable via max_energy); dies if ≤0
age: u32                 // Ticks since birth
flags: u32               // Bit 0: alive
genome_id: u32           // INVARIANT: genome_id == organism slot index
generation: u32          // Generation number
offspring_count: u32     // Number of children produced
parent_id: u32           // Parent organism ID
reproduce_signal: f32    // Neural network reproduction output
species_id: u32          // Species cluster ID
// Morphology traits (Phase 6)
morph_size: f32          // Size multiplier (affects rendering & movement cost)
morph_speed_mult: f32    // Speed multiplier
morph_vision_mult: f32   // Vision range multiplier
morph_metabolism: f32    // Metabolism efficiency (higher = less drain)
```

### Genome (Neural network weights + morphology)
- **Stored at organism's slot index**: `genome_id == organism slot index` (this invariant bounds genome IDs to max_organisms)
- Mutation happens at spawn: random weights drawn from normal distribution, scaled by `mutation.strength`
- **Morphology traits** stored in `MorphTraits` struct, inherited with mutation
- Stored in GPU buffers accessed by compute shaders
- **Must be synced to GPU immediately after creation** via `update_nn_weights_for_genome()`

### Simulation Config ([src/config.rs](../src/config.rs) + `config.toml`)
All parameters are grouped logically:
- `population`: max/initial organisms
- `energy`: drain rates, movement costs, max_age, age_drain_factor, crowding_factor
- `reproduction`: threshold, cooldown, cost, signal_min, **sexual_enabled, mate_range, crossover_ratio**
- `mutation`: rate (probability), strength (std dev)
- `vision`: ray count (8), FOV (degrees), range (pixels)
- `physics`: max_speed, rotation limits, organism_radius
- `food`: growth rate, spawn patches, patch_size, energy value, effectiveness
- `world`: width, height (wrapping)
- `morphology`: **size/speed/vision/metabolism ranges, mutation rate** (Phase 6)
- `biomes`: **enabled, biome_count, growth/speed/drain multipliers** (Phase 6)

**Key energy/population parameters:**
- `max_age`: Organisms die when age exceeds this value
- `age_drain_factor`: Quadratic energy drain increase with age (1.0 = 100% extra at max_age)
- `crowding_factor`: Extra drain when at max population (1.0 = 100% extra)
- `food.effectiveness`: Multiplier for food energy gain (1.0 = normal, lower = harder)
- `food.patch_size`: Size of initial food patches (40 = large sustainable patches)

**Ecosystem dynamics:**
- Organisms spawn distributed across food patches (not clustered)
- Food only exists in patches - no baseline food everywhere
- Logistic food regrowth: food grows where food already exists
- Rare spontaneous food spawning creates new patches to discover

**Runtime configuration**: Edit `config.toml` without recompiling. Use CLI args for testing:
```bash
cargo run --release -- --config custom.toml --auto-exit 30 --paused --speed 16
```

Changes to config update GPU uniforms via `update_config()` each frame ([compute/buffers.rs](../src/compute/buffers.rs)).

---

## Critical Workflows & Conventions

### 1. Adding a New Organism Behavior
Organisms are controlled solely by their neural network output (6 floats: movement forward/rotate, eat, reproduce, etc.). **To change behavior**:
1. Modify network output interpretation in `act.wgsl` (output mapping)
2. Adjust input/output dimensions in `config.rs` if changing network architecture
3. Regenerate genomes if INPUT_DIM or OUTPUT_DIM changes

**Example**: Adding "turn" behavior
- In `act.wgsl`, map output[3] to rotation delta
- Update VisionConfig or mutation parameters to balance the new behavior

### 2. GPU Buffer Synchronization Pattern
**GPU is the source of truth for organism state.** CPU only pushes changes for:
- New organisms spawned during reproduction
- Parent energy updates after reproduction
- New genome weights after mutation

```rust
// After reproduction - sync parent and child organisms
for (idx, org_gpu) in &result.organism_changes {
    self.compute.buffers.update_organism_at(&self.queue, *idx, org_gpu);
}
// Sync new genome weights
for genome_id in &result.new_genome_ids {
    if let Some(weights) = self.simulation.genomes.get_weights_flat(*genome_id) {
        self.compute.buffers.update_nn_weights_for_genome(&self.queue, *genome_id, &weights);
    }
}
```

**Key invariant**: Never call `update_organisms()` (full sync) every frame—this overwrites GPU state with stale CPU data.

### 3. Reproduction & Genetic Inheritance
**Flow is strictly sequential to prevent race conditions:**
1. **Readback**: `read_gpu_state()` pulls strictly previous frame data.
2. **Logic**: CPU checks `reproduce_signal` from that readback.
3. **Write**: Energy is deducted from parent on CPU.
4. **Sync**: Updated parent (low energy) and child are pushed to GPU.
5. **Dispatch**: Compute shader runs on the *new* state.

**System Config**:
- `system.readback_interval`: Must be **1** (every tick) to prevent "free energy" glitches where the GPU snapshot overwrites the CPU's energy deduction. The runtime now sanitizes invalid values back to `1`.
- Runtime config sanitization also protects zero-sized worlds, zero food capacity, impossible reproduction energy relationships, inverted morphology bounds, and zero seasonal period before those values reach world allocation or GPU shaders.

### 4. Rendering Organisms (CPU-driven)
- Organisms rendered as triangles pointing in direction of rotation
- Position+rotation+energy packed into instance data
- Energy controls color/brightness (shader side)
- No deep entity management—render in array order each frame

### 5. Build & Test Workflow
```bash
cargo check          # Fast lint check
cargo test           # Unit tests plus public-API workflow regressions
cargo check --examples  # Verify example-based regression programs compile
cargo build          # Debug build (unoptimized)
cargo build --release  # Optimized (LTO enabled) for real simulation
cargo run --release    # Launch simulator
```

**Debug vs Release**: Dev profile has `opt-level=1` for fast iteration; dependencies compiled at opt-level=3. Always use `--release` for meaningful simulation performance.

### 6. Configuration Changes
Most tuning doesn't require recompilation:
- Pause simulation (Space), adjust sliders in UI (if implemented)
- Or edit hardcoded `SimulationConfig::default()` in `config.rs`, recompile

Compute uniforms updated each frame—no shader recompilation needed for parameter tweaks.

---

## Common Pitfalls & Patterns

### ❌ Don't: Modify organism state on CPU after compute dispatch
Compute shaders read shared buffers—changes aren't visible until next dispatch. Always update via reproduction logic or between ticks.

### ❌ Don't: Forget bytemuck derive on GPU data structs
All `#[repr(C)]` structs sent to GPU need `#[derive(Pod, Zeroable)]` for safe casting.

### ❌ Don't: Change rays in config without updating neural network
The neural network INPUT_DIM is hardcoded to 20 = 8 rays × 2 + 4 internal. Changing `vision.rays` in config.toml without rebuilding genomes will break organism behavior. **Always keep rays=8.**

### ❌ Don't: Spawn organisms at random positions
Organisms need to spawn ON or near food to survive. Random spawning with sparse food leads to mass starvation. See `simulation/mod.rs` spawn logic.

### ✅ Do: Use Xoshiro256PlusPlus for deterministic seeded RNG
Used in [simulation/mod.rs](../src/simulation/mod.rs) and compute shaders. Seeding from organism ID ensures reproducible evolution per run. See [docs/DETERMINISM.md](../docs/DETERMINISM.md) for full reproducibility guide.

### ✅ Do: Batch GPU operations
Compute dispatch is cheap; minimize state-switching between sense→think→act (three dispatches per tick).

### ✅ Do: Use PCG hash for GPU randomness
The world shader uses PCG hash for food spawning. Never use sin() for pseudo-random—it creates visible patterns.

---

## File Navigation Quick Reference

- **Entry point**: [src/main.rs](../src/main.rs) – event loop setup, window creation
- **App state machine**: [src/app.rs](../src/app.rs) – tick logic, event handling, subsystem coordination
- **GPU compute orchestration**: [src/compute/pipeline.rs](../src/compute/pipeline.rs)
- **Organism data & pooling**: [src/simulation/organism.rs](../src/simulation/organism.rs)
- **Rendering setup**: [src/render/mod.rs](../src/render/mod.rs)
- **Compute shaders**: [src/compute/shaders/](../src/compute/shaders/) – sense.wgsl, think.wgsl, act.wgsl, world.wgsl
- **Technical spec**: [docs/DESIGN.md](../docs/DESIGN.md) – comprehensive architecture + equations
- **Determinism guide**: [docs/DETERMINISM.md](../docs/DETERMINISM.md) – reproducibility guarantees
- **Roadmap**: [docs/PLAN.md](../docs/PLAN.md) – next features and known issues
- **Build optimization**: [docs/BUILD_OPTIMIZATION.md](../docs/BUILD_OPTIMIZATION.md) – faster builds guide
- **Debugging guide**: [docs/DEBUGGING.md](../docs/DEBUGGING.md) – comprehensive debugging procedures
- **Audit prompt files**: [.github/prompts/](./prompts/) – reusable Copilot prompts for feature, config, and release audits
- **Audit agent**: [.github/agents/repo-auditor.agent.md](./agents/repo-auditor.agent.md) – repo-specific audit specialist

---

## Build & Debug Commands

```bash
# Fast iteration (default dev profile)
cargo build                      # Incremental, ~3s
cargo run                        # Run with dev optimizations

# Faster with more optimization
cargo run --profile dev-fast     # opt-level=2, no debug

# Release-like testing (thin LTO)
cargo run --profile release-fast

# Full release
cargo run --release

# Demo examples for isolated testing
cargo run --example food_test    # Food generation/rendering
cargo run --example nn_test      # Neural network validation

# Quick smoke test
cargo run -- --auto-exit 5       # Runs for 5 seconds and exits
```

---

## Debugging Tips

1. **"Organisms not moving"**: Check act.wgsl output mappings match expected behavior range
2. **"GPU errors on dispatch"**: Verify buffer sizes match organism count; check compute.rs bindings
3. **"Population crash"**: Inspect energy drain vs food spawn rate in config—starvation likely
4. **"All organisms die at max_age"**: Check age_drain_factor is > 0 for gradual aging, or increase max_age
5. **"Food not visible"**: Check food_max_per_cell is passed to renderer for proper normalization
6. **"Food everywhere"**: Check world.wgsl hash function; use PCG hash not sin()
7. **"Artifacts on screen"**: Camera zoom/pan math in [src/render/camera.rs](../src/render/camera.rs)—verify viewport transform
8. **Performance drops**: Profile GPU dispatch time vs rendering; may need to reduce initial_organisms or ray count

**For detailed debugging, see [docs/DEBUGGING.md](../docs/DEBUGGING.md)**

---

## When Making Changes

- **Changes to network architecture**: Update INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM in [src/simulation/genome.rs](../src/simulation/genome.rs); regenerate genomes
- **New shader inputs/outputs**: Ensure kernel thread dispatches match organism count; update sense/think/act signatures
- **UI additions**: Use egui context in [src/ui/mod.rs](../src/ui/mod.rs); stats tracking in [src/ui/stats.rs](../src/ui/stats.rs). Founder pool browsing/editing also lives in `ui/mod.rs`.
- **User interaction tools**: Add keybinds in `handle_keyboard()` and mouse handlers in `handle_mouse_button()` in [src/app.rs](../src/app.rs)
- **Performance optimization**: Profile with `cargo build --release` first; shader optimizations in wgsl files have highest impact

### Keybind Reference
| Key | Action |
|-----|--------|
| Space | Pause/Resume |
| . | Step one tick (when paused) |
| 1-7 | Speed (1x, 2x, 4x, 8x, 16x, 32x, 64x) |
| WASD/Arrows | Pan camera |
| Scroll | Zoom in/out |
| R | Reset camera |
| F | Follow selected organism |
| E | Feed selected (+20 energy) |
| K | Kill selected organism |
| F5 | Quick save |
| F9 | Quick load |
| I | Toggle inspector panel |
| O | Toggle founder pool browser |
| H | Toggle help overlay |
| Esc | Toggle settings |
| Click | Select organism |
| Right-click | Kill organism or spawn food |

---

## 🚨 Documentation Maintenance (CRITICAL)

**Always keep documentation aligned with code changes.** When making changes to the codebase, you MUST:

### Update These Files When Relevant:
1. **[docs/PLAN.md](../docs/PLAN.md)** – Mark tasks as complete (`[x]`), add new tasks, update current status
2. **[docs/DESIGN.md](../docs/DESIGN.md)** – Update if architecture, data structures, or algorithms change
3. **[.github/copilot-instructions.md](./copilot-instructions.md)** – Update module responsibilities, file paths, or workflows
4. **[README.md](../README.md)** – Update if build instructions, features, or usage changes

### Checklist After Code Changes:
- [ ] Did you add/remove/rename a file? → Update file references in all docs
- [ ] Did you change a struct/function signature? → Update DESIGN.md data structures
- [ ] Did you complete a PLAN.md task? → Mark it `[x]` and update status section
- [ ] Did you add a new feature? → Document it in README.md and DESIGN.md
- [ ] Did you change dependencies? → Verify versions in DESIGN.md match Cargo.toml
- [ ] Did you fix a significant bug? → Consider adding to debugging tips

### Version Alignment:
Keep these in sync:
- `Cargo.toml` dependency versions ↔ DESIGN.md technology stack table
- `src/simulation/genome.rs` constants ↔ DESIGN.md neural network section
- `src/config.rs` parameters ↔ DESIGN.md configuration section

**Outdated documentation leads to confusion and bugs. Treat docs as part of the code.**

