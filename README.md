# Evolution Simulator

A GPU-accelerated 2D evolution simulator where digital organisms with neural network brains live, compete, reproduce, and evolve.

## Features

- **GPU-Accelerated Simulation**: Thousands of organisms simulated in parallel using compute shaders
- **Neural Network Brains**: Each organism has a small neural network (20→16→6) controlling its behavior
- **Natural Selection**: Organisms that find food survive to reproduce; evolution emerges naturally
- **Real-time Visualization**: Watch evolution happen in real-time with visible food layer
- **Runtime Configuration**: Edit `config.toml` without recompiling
- **Age-Based Mortality**: Organisms experience increasing energy drain as they age
- **Population Dynamics**: Crowding factor adds pressure when population is at capacity
- **Difficulty Control**: Adjustable food effectiveness for environment challenge
- **Species Detection**: Organisms grouped by genetic similarity with distinct colors
- **Predation System**: Optional attack behavior with energy transfer (disabled by default)
- **Dynamic Environments**: Seasonal food cycles and moving resource hotspots
- **Polished UI**: Theme system, toolbar, help overlay, improved graphs
- **Morphology**: Evolvable traits (size, speed, vision, metabolism) that affect physics and rendering
- **Sexual Reproduction**: Optional crossover-based reproduction with mate finding
- **Biomes**: Regional environments (Fertile, Barren, Swamp, Harsh) with different effects
- **Founder Pool Workflow**: Reuse curated founders across runs with readable JSON storage and in-app editing
- **Config Safety Guards**: Runtime sanitization clamps values that would break allocation, wrapping, or shader math

## Requirements

- **OS**: Windows 10+, Linux
- **GPU**: Any GPU/driver stack supported by `wgpu` (DX12/Vulkan tested path on Windows)
- **VRAM**: 2 GB minimum
- **Rust**: 1.75 or later

## Building

```bash
# Clone the repository
git clone https://github.com/yourusername/evolution-sim.git
cd evolution-sim

# Build in release mode
cargo build --release

# Run
cargo run --release
```

## Command Line Options

```bash
# Run with default settings
cargo run --release

# Use a custom configuration file
cargo run --release -- --config my_config.toml

# Auto-exit after 30 seconds (useful for testing/benchmarking)
cargo run --release -- --auto-exit 30

# Start paused with 2x speed
cargo run --release -- --paused --speed 2

# Show all options
cargo run --release -- --help
```

## Controls

| Key | Action |
|-----|--------|
| WASD / Arrows | Pan camera |
| Scroll | Zoom in/out |
| Space | Pause/Resume |
| 1-7 | Speed (1x, 2x, 4x, 8x, 16x, 32x, 64x) |
| . (period) | Single-step when paused |
| R | Reset camera |
| Click | Select organism |
| F | Follow selected |
| I | Toggle inspector |
| H | Toggle help overlay |
| O | Toggle founder pool browser |
| E | Feed selected (+20 energy) |
| K | Kill selected organism |
| Esc | Menu |
| F5 | Quick save |
| F6 | Export founder store |
| F9 | Quick load |

Stats are toggled from the toolbar button in the top-right corner.

## How It Works

1. **Sense**: Each organism casts vision rays to detect food and other organisms
2. **Think**: Sensory data is processed through the organism's neural network
3. **Act**: Network outputs control movement, eating, and reproduction
4. **Evolve**: Successful organisms reproduce, passing mutated copies of their neural networks to offspring

Over time, organisms evolve increasingly sophisticated foraging behaviors.

To reduce the number of completely naive starts, the simulator can export a founder store and reuse those genomes as future founders. Founders keep their evolved neural weights and morphology, but start in new positions with fresh energy so each run still has environmental variation. The readable `founder_pool.json` store can be curated either from the CLI tooling or directly in the in-app founder browser.

## Project Structure

```
evolution-sim/
├── config.toml           # Runtime configuration (edit without recompiling)
├── src/
│   ├── main.rs           # Entry point & CLI
│   ├── app.rs            # Application state
│   ├── config.rs         # Simulation parameters
│   ├── simulation/       # Simulation logic (organism, genome, world)
│   ├── compute/          # GPU compute shaders (sense, think, act)
│   ├── render/           # Rendering (organisms, world, camera)
│   └── ui/               # User interface (egui HUD, inspector, theme)
├── docs/
│   ├── DESIGN.md         # Technical specification
│   └── PLAN.md           # Project roadmap
└── .github/
    ├── agents/                  # Custom Copilot agents for repo workflows
    ├── prompts/                 # Reusable Copilot prompts for audits/checks
    └── copilot-instructions.md  # AI assistant context
```

## Configuration

Edit `config.toml` to adjust simulation parameters without recompiling. The file can be partial: omitted values fall back to built-in defaults, and invariant-controlled values like `vision.rays` and organism `readback_interval` are enforced by the runtime. The runtime also clamps obviously unsafe values such as zero-sized worlds, zero food capacity, impossible reproduction energy relationships, inverted morphology bounds, and zero seasonal period before they reach allocation or shader code.

```toml
# Optional fixed seed for reproducibility
# seed = 42

[population]
max_organisms = 4000
initial_organisms = 600

[energy]
starting = 70.0         # Starting energy for new organisms
maximum = 200.0         # Max energy capacity
passive_drain = 0.14    # Energy lost per tick
max_age = 2000          # Ticks before death from old age
age_drain_factor = 1.0  # Extra drain at max_age (quadratic scaling)
crowding_factor = 1.0   # Extra drain when population at capacity

[reproduction]
threshold = 70.0        # Minimum energy to reproduce
cost = 50.0             # Energy cost (also child's starting energy)
min_age = 150           # Minimum ticks before reproduction
signal_min = 0.3        # Minimum neural network output to trigger reproduction

[mutation]
rate = 0.05             # Probability of mutating each weight
strength = 0.2          # Standard deviation of mutation noise

[food]
energy_value = 4.0      # Energy gained from eating
growth_rate = 0.003     # Food regeneration rate
max_per_cell = 10.0     # Maximum food per cell
effectiveness = 1.0     # Multiplier for food energy (lower = harder)
initial_patches = 200   # Number of food patches at start
patch_size = 10         # Radius of food patches

[bootstrap]
enabled = true          # Persist curated founders between runs
path = "founder_pool.json"
founder_count = 16      # Typical curated founder count for stable mixed starts
survivor_count = 256    # Max living organisms kept when exporting the pool
load_on_start = true    # Reuse founders on future starts
save_on_exit = true     # Update the pool automatically on shutdown

[system]
food_readback_interval = 60
diagnostic_interval = 60
```

The startup population is mixed: stored founders are used first, and any remaining slots are still filled with fresh random organisms. The bootstrap path now supports two formats:

- `founder_pool.json`: readable founder pool with labels, scores, and evaluation metadata
- `survivor_bank.bin`: legacy binary bank format, still supported for compatibility

Press `F6` at any time to export the current living survivors immediately.

Press `O` or click the library button in the top bar to open the founder browser. It supports filtering, enabled-only views, sorting, and editing founder labels, tags, notes, and enabled state.

For offline training, run:

```bash
cargo run --example train_survivor_bank -- --epochs 4 --train-ticks 1000 --eval-ticks 1200 --seed 42
```

That example repeatedly trains a survivor bank and validates it against leaner starter presets. It is still useful for experimentation, but the more transparent founder-pool workflow is now the primary path for curation and inspection.

Runtime founder exports are protected from regression as well: when exporting to the legacy binary path, the app only overwrites `survivor_bank.bin` when the newly exported bank scores stronger than the existing one.

For direct food-navigation founder search, run:

```bash
cargo run --release --example foraging_founder_search -- --output founder_pool.json --batch-size 10000 --iterations 8 --target-founders 128
```

That search places each random organism near food but never directly pointed at it, then keeps the founders that actually turn toward and reach the food.

To inspect or convert founder stores:

```bash
cargo run --example founder_pool_tool -- summary --path founder_pool.json
cargo run --example founder_pool_tool -- list --path founder_pool.json --limit 20
cargo run --example founder_pool_tool -- convert-bank --input survivor_bank.bin --output founder_pool.json
```

## Testing

```bash
# Run unit tests for config/genome/world/species invariants
cargo test

# Verify all example-based checks and demos still compile
cargo check --examples

# Run broad verification examples
cargo run --example feature_test
cargo run --example integration_test
cargo run --example simulation_smoke_test
cargo run --example simulation_feature_probe
cargo run --example train_survivor_bank

# Long-run metrics CSV for tuning
cargo run --example metrics_logger -- --ticks 2000 --interval 20 --output metrics.csv

# Long-run metrics using the real app config and founder pool
cargo run --example metrics_logger -- --config config.toml --ticks 2400 --interval 200 --output metrics-realapp.csv

# Longer-run validation with a scarcer-food preset
cargo run --release --example metrics_logger -- --config config.longrun.toml --ticks 25000 --interval 2500 --output metrics-25k.csv

# Run neural network validation
cargo run --example nn_test

# Measure startup genome / sensory / action diversity
cargo run --example nn_diversity_probe -- --config config.toml

# Same probe, but with survivor-bank founders disabled
cargo run --example nn_diversity_probe -- --config config.toml --disable-bootstrap

# Direct founder search for nearby off-angle food navigation
cargo run --release --example foraging_founder_search -- --output founder_pool.json --batch-size 10000 --iterations 8 --target-founders 128

# Inspect or convert readable founder pools
cargo run --example founder_pool_tool -- summary --path founder_pool.json

# Run species and reproduction verification
cargo run --example species_test
cargo run --example repro_test
```

`cargo test` now also covers public-API workflow regressions for file-based config sanitization and the primary `founder_pool.json` bootstrap path.

## Copilot Workflows

Workspace-scoped Copilot customizations live under `.github/agents/` and `.github/prompts/`.

- `Repo Auditor` agent: deep repo audits for code/docs/tests/config drift
- `Feature Audit` prompt: verify a feature against implementation and tests
- `Config Workflow Check` prompt: validate config invariants, founder-pool flows, and diagnostics
- `Release Readiness` prompt: build/test/docs alignment pass before release

## Demo Examples

```bash
# Performance benchmark
cargo run --example perf_test

# Long-run metrics CSV for tuning
cargo run --example metrics_logger -- --ticks 2000 --interval 20 --output metrics.csv

# Reproduction system test
cargo run --example repro_test
```

## License

None