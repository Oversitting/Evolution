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

## Requirements

- **OS**: Windows 10+, Linux
- **GPU**: Vulkan 1.2
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
| E | Feed selected (+20 energy) |
| K | Kill selected organism |
| Esc | Menu |
| F5 | Quick save |
| F9 | Quick load |

## How It Works

1. **Sense**: Each organism casts vision rays to detect food and other organisms
2. **Think**: Sensory data is processed through the organism's neural network
3. **Act**: Network outputs control movement, eating, and reproduction
4. **Evolve**: Successful organisms reproduce, passing mutated copies of their neural networks to offspring

Over time, organisms evolve increasingly sophisticated foraging behaviors.

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
    └── copilot-instructions.md  # AI assistant context
```

## Configuration

Edit `config.toml` to adjust simulation parameters without recompiling:

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
growth_rate = 0.05      # Food regeneration rate
max_per_cell = 10.0     # Maximum food per cell
effectiveness = 1.0     # Multiplier for food energy (lower = harder)
initial_patches = 200   # Number of food patches at start
patch_size = 10         # Radius of food patches
baseline_food = 0.0     # Minimum food everywhere (0.0 = patches only)
spawn_chance = 0.000001 # Chance to seed new food patches
spawn_amount = 2.0      # Food placed when a new patch appears
```

## Testing

```bash
# Run unit tests (17 tests for Phase 1/2 features)
cargo run --example feature_test

# Run integration tests (7 tests for runtime behavior)
cargo run --example integration_test

# Run neural network validation
cargo run --example nn_test

# Run food system demo
cargo run --example food_test
```

## Demo Examples

```bash
# Performance benchmark
cargo run --example perf_test

# Reproduction system test
cargo run --example repro_test
```

## License

None