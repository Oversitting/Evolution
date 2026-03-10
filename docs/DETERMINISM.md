# Determinism & Reproducibility Guide

**Version**: 1.0  
**Created**: January 29, 2026  
**Status**: Active

This document describes the determinism guarantees of the Evolution Simulator and how to ensure reproducible simulation runs.

---

## Overview

The Evolution Simulator is designed to be **fully deterministic** when given the same configuration and seed. This means:

1. **Same seed = Same results**: Two simulations with identical seeds will produce identical organism populations, positions, behaviors, and evolution outcomes.

2. **Speed-independent**: Running at 1x, 16x, or 64x speed produces identical results (only affects ticks processed per frame).

3. **Readback-independent**: The `readback_interval`, `food_readback_interval`, and `diagnostic_interval` settings don't affect simulation logic.

4. **Frame-rate independent**: FPS fluctuations don't affect the simulation (each tick is discrete and self-contained).

---

## How to Enable Determinism

### 1. Set a Fixed Seed

In `config.toml`:
```toml
seed = 12345  # Any u64 value
```

Or programmatically:
```rust
let mut config = SimulationConfig::default();
config.seed = Some(12345);
```

### 2. Keep Configuration Identical

All configuration parameters must be identical for reproducible results:
- Population settings
- Energy parameters  
- Food settings
- Mutation rates
- World dimensions
- Physics constants

---

## Technical Implementation

### Random Number Generation

| Component | RNG Type | Seed Source |
|-----------|----------|-------------|
| CPU Simulation | Xoshiro256PlusPlus | `config.seed` |
| World Generation | Uses CPU RNG | Inherited from simulation |
| Genome Mutation | Uses CPU RNG | Inherited from simulation |
| GPU Food Spawning | PCG Hash | `tick + position` |

### CPU-Side Determinism

The simulation uses `Xoshiro256PlusPlus` from the `rand_xoshiro` crate, which is:
- Fast
- High-quality statistical properties
- Reproducible across platforms

All CPU random operations flow through the simulation's single RNG instance:
```rust
pub struct Simulation {
    pub rng: Xoshiro256PlusPlus,
    // ...
}
```

### GPU-Side Determinism

GPU shaders use a PCG-based hash function for random values:
```wgsl
fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
```

The seed combines `tick`, position, and organism ID to ensure:
- Deterministic output for the same inputs
- Different values for different organisms/positions
- No visible patterns (unlike sin-based hashes)

---

## Speed Multiplier Independence

The speed multiplier only affects how many simulation ticks are processed per rendered frame:

```rust
// In app.rs update()
if !self.paused {
    for _ in 0..self.speed_multiplier {
        self.simulation_step();  // Each step is identical
    }
}
```

Available speeds:
| Key | Speed | Ticks per Frame |
|-----|-------|-----------------|
| 1 | 1x | 1 |
| 2 | 2x | 2 |
| 3 | 4x | 4 |
| 4 | 8x | 8 |
| 5 | 16x | 16 |
| 6 | 32x | 32 |
| 7 | 64x | 64 |

Since each tick is a discrete, deterministic operation, batching multiple ticks produces identical results.

---

## Readback Interval Independence

The system settings control CPU↔GPU synchronization frequency:

```toml
[system]
readback_interval = 1        # How often CPU reads organism state
food_readback_interval = 60  # How often CPU reads food grid
diagnostic_interval = 60     # How often to log diagnostics
```

These settings affect **when** the CPU observes GPU state, not **what** that state is:

- `readback_interval = 1`: CPU sees organism state every tick (required for accurate reproduction)
- `readback_interval > 1`: CPU may check reproduction less frequently, but organisms still exist in correct states

**Important**: For accurate reproduction, `readback_interval` should remain at `1`. Higher values may cause delayed reproduction checks but won't change final simulation state over many ticks.

---

## Known Limitations

### 1. GPU Thread Race Conditions (Contained)

Multiple organisms on the same food cell may have a read-write race when eating:
```wgsl
let food_available = food[food_idx];  // Multiple threads read same value
// ...
food[food_idx] = food_available - food_eaten;  // Race condition
```

This is **deterministic within a single GPU** because:
- Thread execution order is consistent on the same hardware
- Same positions → same execution order → same results

**Cross-GPU Note**: Results may differ across different GPU models/drivers due to thread scheduling differences. For exact reproducibility across machines, use the same GPU model.

### 2. Floating Point Precision

IEEE 754 floating-point operations are deterministic, but:
- Different compiler optimizations may reorder operations
- GPU shader compilers may optimize differently across drivers

For maximum reproducibility, use release builds with consistent optimization settings.

### 3. Save/Load Continuity

When loading a saved simulation:
- RNG state is re-seeded as `seed.wrapping_add(tick)`
- This ensures continued determinism but means a loaded simulation diverges from an uninterrupted run at the same tick

---

## Testing Determinism

Run the determinism test suite:
```bash
cargo run --example determinism_test --release
```

This validates:
1. **Seed determinism**: Same seed → identical results
2. **Readback independence**: Different intervals → same final state
3. **Initial state determinism**: Same seed → identical starting conditions
4. **Speed independence**: Different speed multipliers → same final state

---

## Debugging Non-Determinism

If you observe non-deterministic behavior:

1. **Check seed is set**: Ensure `config.seed = Some(value)`

2. **Check for thread_rng()**: Search codebase for `thread_rng` or `from_entropy`:
   ```bash
   grep -r "thread_rng\|from_entropy" src/
   ```

3. **Check HashMap usage**: Standard `HashMap` has non-deterministic iteration. Use `BTreeMap` for ordered iteration.

4. **Check GPU shader RNG**: Ensure all random values derive from `tick`, position, or organism ID.

5. **Compare snapshots**: Use the determinism test framework to capture and compare simulation states.

---

## Future Improvements

- [ ] Save/restore full RNG state for perfect continuity
- [ ] Add cross-platform determinism tests
- [ ] Implement deterministic GPU eating (atomic operations)
- [ ] Add replay/verification mode
