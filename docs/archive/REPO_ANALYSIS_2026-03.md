# Repository Analysis - March 2026

This document records a code-and-documentation audit of the Evolution Simulator repository as of March 10, 2026.

## Scope

- Compared top-level docs against current Rust/WGSL implementation.
- Verified the main crate build, unit tests, and example target compilation.
- Reviewed core simulation, UI, configuration, and advanced Phase 6 features.

## Verified Functional Features

- GPU sense -> think -> act -> world pipeline is implemented and wired through the compute pipeline.
- Neural-network genome layout is 20 -> 16 -> 6 and is synchronized to GPU buffers.
- CPU-side reproduction with GPU state readback is implemented.
- Predation is implemented in `act.wgsl` behind config gating.
- Species clustering, morphology traits, sexual reproduction, and biome generation are present in code.
- Save/load persists organisms, genomes, food, config, and current tick.
- UI includes HUD, toolbar, inspector, stats panel, help overlay, and settings panel.

## Verified Documentation Drift Before Fixes

- The determinism guide incorrectly claimed organism readback settings were logic-independent.
- The design spec described morphology crossover as averaging, but the implementation uses uniform per-trait selection.
- The design spec still documented the organism GPU struct as 56 bytes; the actual struct is 72 bytes.
- Phase 6 biome default multipliers in docs did not match the code.
- Multiple example verification programs had drifted behind current genome/morphology APIs and no longer compiled.
- The stats toolbar tooltip advertised an `S` shortcut that does not exist because `S` is already used for camera movement.

## Fixes Applied During This Audit

- Added `SimulationConfig::sanitize()` and enforced it when loading configs and creating the app.
- Enforced the critical runtime invariants:
  - `vision.rays` is forced to `8`.
  - `system.readback_interval` is forced to `1`.
  - `food_readback_interval` and `diagnostic_interval` are clamped away from `0`.
- Added unit tests for:
  - config sanitation
  - partial config parsing
  - crossover endpoint behavior
  - morphology mutation bounds
  - biome generation behavior
- Repaired stale example verification programs to compile against current APIs.
- Added serde defaults so `config.toml` can omit unchanged sections and invariant-controlled settings.
- Added a headless GPU-backed smoke test example that checks runtime invariants across the full simulation loop.
- Fixed CPU-side initial spawn and offspring spawn positioning so organisms are wrapped into world bounds before the first GPU tick.
- Corrected the stats toolbar tooltip.
- Updated README, DESIGN, PLAN, determinism guidance, config comments, and Copilot instructions to match verified behavior.

## Residual Risks

- Obstacles are still not a finished feature. Storage and collision hooks exist, but there is no workflow that populates obstacle cells during normal use.
- Predation still performs an O(n) nearest-target scan in the compute shader.
- Sexual reproduction remains asymmetric in energy/cooldown cost: the initiating parent pays the full reproduction cost and cooldown, while the mate only needs to satisfy mate-selection criteria.
- Save/load still does not preserve the exact RNG state, so a resumed run diverges from an uninterrupted run.

## Recommended Ongoing Checks

- Run `cargo test` for unit-level invariants.
- Run `cargo check --examples` in CI to prevent further example drift.
- Keep README and design docs tied to shipped defaults, especially for Phase 6 parameters and runtime invariants.