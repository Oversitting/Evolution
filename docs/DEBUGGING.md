# Debugging Guide

This document provides comprehensive debugging procedures for the Evolution Simulator. Use the demo examples and CLI tools to isolate and diagnose issues.

## Table of Contents

1. [Quick Diagnostics](#quick-diagnostics)
2. [Demo Examples](#demo-examples)
3. [Common Issues & Solutions](#common-issues--solutions)
4. [GPU Debugging](#gpu-debugging)
5. [Performance Profiling](#performance-profiling)
6. [Logging & Tracing](#logging--tracing)

---

## Quick Diagnostics

### Basic Health Check

Run the simulator for 5 seconds with verbose logging:

```bash
RUST_LOG=debug cargo run -- --auto-exit 5 2>&1 | tee test_log.txt
```

Check for:
- Population staying stable (not crashing to 0)
- Energy levels reasonable (not all 0 or all max)
- Generation counter incrementing (reproduction working)
- Config sanitation warnings such as forced `readback_interval=1`, clamped world size, or restored food capacity

### Key Metrics to Watch

| Metric | Healthy Range | Problem Indicator |
|--------|---------------|-------------------|
| Population | 100-1000 | Dropping to 0 = starvation; stuck at max = over-abundance |
| Avg Energy | 40-80 | Near 0 = food problem; at max = too easy |
| Max Generation | Increasing | Stuck at 0 = no reproduction |
| Repro Ready | 5-50% | 0% = energy or age issue; 100% = thresholds too low |

---

## Demo Examples

### Food Grid Test (`food_test`)

**Purpose**: Test food generation, growth, and rendering in isolation.

**Run**:
```bash
cargo run --example food_test
```

**Controls**:
- `Space`: Pause/resume
- `R`: Reset food to initial patches
- `G`: Toggle spontaneous generation on/off
- `1-4`: Speed multiplier
- `Mouse scroll`: Zoom
- `Middle mouse drag`: Pan
- `Escape`: Exit

**What to Check**:
1. Food patches should be visible as green areas on dark background
2. Patches should grow and spread slowly (logistic growth)
3. With `G` toggled on, occasional random food should appear (very rare)
4. With `G` toggled off, no new food should appear outside existing patches

**Diagnosing Problems**:
- **All green**: Spontaneous generation rate too high, or growth rate explosion
- **No food visible**: Initial patches not created, or food_max_per_cell wrong
- **Food not growing**: growth_rate too low, or threshold (0.1) too high
- **Patterns in random food**: Hash function not working correctly

### Neural Network Test (`nn_test`)

**Purpose**: Validate neural network forward pass computation on CPU.

**Run**:
```bash
cargo run --example nn_test
```

**What it Tests**:
1. Zero input → zero output (unbiased networks)
2. Identity-like weights → predictable propagation
3. Negative weights → sign changes
4. Large weights → saturation (tanh approaches ±1)
5. Random weights → outputs stay in valid range (-1, 1)

**Diagnosing Problems**:
- **NaN outputs**: Division by zero or overflow in computation
- **Outputs outside [-1, 1]**: Missing tanh activation
- **All outputs identical**: Weights not varied properly

### Performance Benchmark (`perf_test`)

**Purpose**: Identify shader performance bottlenecks.

**Run**:
```bash
cargo run --example perf_test --release
```

**What it Tests**:
- World shader timing (with and without PCG hash)
- Sense shader timing (vision raycasting)
- Think shader timing (neural network forward pass)
- Act shader timing (movement and eating)

**Sample Output**:
```
--- Summary (per tick avg) ---
World (with hash):     0.226 ms
World (simple):        0.195 ms
Sense (8 rays):        0.349 ms
Think (NN):            0.322 ms
Act:                   0.091 ms
```

**Performance Targets**:
- Total < 1 ms/tick for 60 FPS with headroom
- Sense shader should not dominate (was 2.2 ms with O(n²) organism detection)

### Food System Test (`food_system_test`)

**Purpose**: Validate food growth, distribution, and spontaneous generation.

**Run**:
```bash
cargo run --example food_system_test --release
```

**Tests**:
1. Food growth dynamics (logistic growth to capacity)
2. Food patch distribution (coverage and quadrant balance)
3. Spontaneous generation rate (should be rare but not zero)

### Reproduction Test (`repro_test`)

**Purpose**: Validate reproduction conditions, cooldowns, and energy transfer.

**Run**:
```bash
cargo run --example repro_test --release
```

**Tests**:
1. Reproduction conditions (energy, signal, age, cooldown)
2. Cooldown mechanics (decrement and saturation)
3. Energy transfer (parent to child)
4. Generation tracking

---

## Common Issues & Solutions

### Issue: Population Crashes to Zero

**Symptoms**: All organisms die within the first few hundred ticks.

**Diagnostic Steps**:
1. Check food exists: `cargo run --example food_test` - verify green patches visible
2. Check energy drain: Look at `passive_drain` in config.toml
3. Check food energy value: `food.energy_value` might be too low
4. Check startup logs for config sanitation warnings; a broken config may have been auto-clamped into a survivable but unintended state

**Solutions**:
```toml
# In config.toml, try:
[energy]
passive_drain = 0.1  # Lower if organisms starving

[food]
energy_value = 5.0   # Higher if not enough energy from eating
max_per_cell = 10.0  # More food available
growth_rate = 0.05   # Faster regrowth
```

### Issue: Food Everywhere (Uniform Green)

**Symptoms**: Entire world is green instead of distinct patches.

**Diagnostic Steps**:
1. Check world.wgsl shader's spontaneous generation rate
2. Verify PCG hash function working (should be sparse random)
3. Check if food_max_per_cell passed correctly to render shader

**Solutions** (already fixed in this codebase):
- Use PCG hash instead of sin() for randomness
- Rate should be < 0.00001 (about 5 cells per tick in 1024x512 world)
- Dispatch workgroups using correct X and Y dimensions

### Issue: Organisms Not Moving

**Symptoms**: Organisms stay in place, not exploring.

**Diagnostic Steps**:
1. Check act.wgsl output mappings
2. Verify neural network weights are initialized
3. Check max_speed in config

**Solutions**:
```toml
[physics]
max_speed = 2.0       # Increase if too slow
max_rotation = 0.3    # Increase for faster turning
```

### Issue: No Reproduction

**Symptoms**: Generation counter stays at 0.

**Diagnostic Steps**:
1. Check energy levels - need to reach `reproduction.threshold`
2. Check ages - need to reach `reproduction.min_age`
3. Check reproduce signal from neural network

**Look at log output**:
```
repro_ready=0   # No organisms meeting reproduction criteria
```

**Solutions**:
```toml
[reproduction]
threshold = 80.0    # Lower if organisms can't reach 100
min_age = 100       # Lower if organisms die before reproducing
signal_min = 0.3    # Lower threshold for neural network signal
```

### Issue: GPU Errors

**Symptoms**: Crashes with wgpu errors, validation layer messages.

**Diagnostic Steps**:
1. Enable validation layer: Install Vulkan SDK
2. Check buffer sizes match organism count
3. Verify struct padding matches between Rust and WGSL

**Common Fixes**:
- Ensure `#[repr(C)]` on all GPU structs
- Check `bytemuck::Pod, Zeroable` derives
- Match padding in WGSL structs exactly

---

## GPU Debugging

### Enable WGPU Validation

Set environment variable before running:

```bash
# PowerShell
$env:WGPU_BACKEND = "vulkan"

# Or for more debug output
$env:RUST_LOG = "wgpu=debug"
```

### Check Shader Compilation

Shader errors will appear at runtime since we use `include_str!`. Look for:
```
panicked at 'failed to compile shader: ...
```

### Validate Buffer Sizes

Add logging to check buffer sizes:
```rust
log::debug!("Organism buffer size: {} bytes", organisms.len() * size_of::<OrganismGpu>());
```

---

## Performance Profiling

### Frame Time Breakdown

The UI shows timing information:
- **Readback ms**: GPU→CPU organism data transfer
- **Upload ms**: CPU→GPU buffer updates  
- **Submit ms**: GPU command queue submission
- **Compute ms**: Total compute time

### Known Performance Issues (Fixed)

#### O(n²) Organism Detection in Sense Shader

**Symptom**: Very slow simulation (2+ ms per tick), especially with 1000+ organisms.

**Root Cause**: The sense shader checked every organism against every other organism for each ray step:
```wgsl
// BAD - O(n²) complexity
for (var other = 0u; other < config.num_organisms; other++) {
    // distance check
}
```

**Fix**: Removed organism-to-organism detection (not critical for food-seeking evolution):
```wgsl
// GOOD - O(n) with food grid lookup only
if food[grid_idx] > 0.5 {
    hit_type = 0.5;  // Food
    break;
}
```

**Performance Improvement**: 6.3x faster sense shader (2.2 ms → 0.35 ms per tick).

**Future Improvement**: Implement spatial hash for efficient organism proximity queries if organism detection needed.

### High Readback Time

Readback happens every tick by default (configurable via `system.readback_interval` in config).

If readback is slow:
1. Reduce readback frequency (increase `READBACK_INTERVAL`)
2. Reduce organism count
3. Consider async readback patterns

### High Compute Time

If compute shaders are slow:
1. Reduce `vision.rays` (fewer raycasts per organism)
2. Reduce `vision.range` (shorter rays)
3. Use `--release` profile

---

## Logging & Tracing

### Log Levels

```bash
# Only errors
RUST_LOG=error cargo run

# Info level (default)
RUST_LOG=info cargo run

# Debug (verbose)
RUST_LOG=debug cargo run

# Trace (very verbose)
RUST_LOG=trace cargo run

# Module-specific
RUST_LOG=evolution_sim::compute=debug cargo run
```

### Key Log Messages

Look for these periodic log messages:
```
Tick 0: pop=300, avg_energy=80.0, max_gen=0, repro_ready=0, ages=[0, 0, 0, 0, 0], repro_signals=[...]
```

This tells you:
- `pop`: Current alive organisms
- `avg_energy`: Mean energy level
- `max_gen`: Highest generation reached
- `repro_ready`: Organisms meeting reproduction criteria
- `ages`: Sample of organism ages
- `repro_signals`: Sample of neural network reproduction outputs

### Writing Custom Diagnostics

Add to `simulation/mod.rs` or `compute/pipeline.rs`:

```rust
log::debug!("Custom diagnostic: {}", value);
```

---

## CLI Testing Commands

### Quick Smoke Test
```bash
cargo run -- --auto-exit 5
```

### Extended Stability Test
```bash
cargo run -- --auto-exit 60 --speed 4
```

### Start Paused for Inspection
```bash
cargo run -- --paused
```

### Use Custom Config
```bash
cargo run -- --config test_config.toml
```

### Capture Output
```bash
cargo run -- --auto-exit 30 2>&1 | tee test_output.txt
```

---

## Adding New Tests

### Adding a Demo Example

1. Create `examples/your_demo.rs`
2. Add any required shaders in `examples/`
3. Document in this file
4. Test with `cargo run --example your_demo`

### Adding a Unit Test

1. Add `#[cfg(test)]` module in the relevant source file
2. Or create `tests/your_test.rs` for integration tests
3. Run with `cargo test`

### Example Test Structure

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_organism_spawning() {
        let config = SimulationConfig::default();
        let mut sim = Simulation::new(&config);
        
        let initial_count = sim.organism_count();
        // ... test logic ...
        assert!(sim.organism_count() >= initial_count);
    }
}
```

---

## Summary Checklist

When debugging, work through this checklist:

- [ ] Check logs for errors/warnings
- [ ] Verify config.toml values are reasonable
- [ ] Run food_test to verify food system
- [ ] Run nn_test to verify neural network
- [ ] Check GPU is being used (look for adapter info in logs)
- [ ] Try --release build if performance issues
- [ ] Compare against known-good config values
- [ ] Check recent code changes for regressions
