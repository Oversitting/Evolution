# Repository Audit - March 2026

This document records a repo-wide audit of Evolution Simulator covering implementation, tests, configuration safety, workflow tooling, and documentation alignment.

## Scope

- Compared README, DESIGN, PLAN, DEBUGGING, determinism guidance, and Copilot instructions against the current Rust and WGSL implementation.
- Reviewed unit tests, example-based checks, and the main public workflow surfaces.
- Verified config handling and founder-pool bootstrap behavior directly in code before accepting audit claims.

## Verified Behavior

- Core GPU pipeline is wired as `sense -> think -> act -> world`.
- Phase 6 core features are implemented: morphology, sexual reproduction, biome generation, and founder-pool bootstrap workflow.
- Save/load and founder-store version checks are present in the save/load layer.
- `diagnostic_interval` and `food_readback_interval` are wired into the compute pipeline.
- Founder-pool JSON is the primary curated bootstrap path, with filtering and score ordering applied on load.

## Findings

### High

- Direct automated coverage is still weakest around GPU compute correctness, render correctness, and the app frame/state machine. The repository has strong CPU-side checks and many example binaries, but those paths still rely heavily on manual or example-driven verification.

### Medium

- Config sanitization was previously too narrow. The runtime already enforced `vision.rays=8` and `readback_interval=1`, but it did not guard other values that could break world allocation or shader math, such as zero world dimensions, zero food capacity, zero seasonal period, impossible reproduction energy relationships, or inverted morphology bounds.
- The main JSON founder-pool workflow was documented and implemented, but it lacked a dedicated public-API regression test outside the save/load module.
- Repo-scoped Copilot workflow files were missing. There was strong baseline project context in `.github/copilot-instructions.md`, but no reusable prompts or custom audit agent for repeated repo audits and release checks.

### Low

- Phase 6 status wording was still ambiguous in the roadmap even though the core feature set is complete and only follow-on items remain.
- One earlier audit pass overclaimed a few gaps. Direct verification showed that save/load version checks and diagnostic interval wiring were already implemented.

## Changes Applied During This Audit

- Hardened `SimulationConfig::sanitize()` with runtime-safe guards for:
  - zero `population.max_organisms`
  - zero world dimensions
  - invalid energy maximum/start relationships
  - invalid reproduction threshold/cost/min-age relationships
  - zero `food.max_per_cell`
  - zero `food.patch_size`
  - invalid baseline/spawn food values
  - zero `food.seasonal_period`
  - invalid hotspot count when hotspots are enabled
  - inverted morphology min/max bounds
  - invalid unit-interval fields such as mutation rate and energy transfer
  - empty bootstrap path
- Added unit coverage for the expanded sanitization behavior and a direct `SimUniform::from_config()` mapping check.
- Added public integration-style workflow regressions for:
  - file-based config sanitization
  - `founder_pool.json` bootstrap filtering, ordering, and quality-score handling
- Added workspace Copilot workflow files:
  - `.github/agents/repo-auditor.agent.md`
  - `.github/prompts/feature-audit.prompt.md`
  - `.github/prompts/config-workflow-check.prompt.md`
  - `.github/prompts/release-readiness.prompt.md`
- Updated README, PLAN, DEBUGGING, and Copilot instructions to reflect the current behavior and workflow surface.

## Residual Risks

- GPU shader correctness is still not validated with targeted automated assertions for sensing, action math, or render output.
- App-level input, pause/resume sequencing, follow-camera behavior, and UI event interactions still have little to no direct automated coverage.
- Example binaries remain important verification artifacts; `cargo check --examples` should continue to be treated as a required regression gate.

## Recommended Next Steps

1. Add headless GPU-backed correctness tests for selected sense/act/world shader invariants.
2. Add app-state regression tests for pause, step, follow, and selection persistence.
3. Add a small config permutation matrix around reproduction, food scarcity, and founder-pool bootstrap startup mixes.