---
description: "Validate config invariants, founder-pool workflows, logging, and runtime safety for Evolution Simulator."
name: "Config Workflow Check"
argument-hint: "Config area or workflow to validate"
agent: "Repo Auditor"
---
Review the requested configuration or workflow path in Evolution Simulator.

Focus on:
- Runtime invariants that can break allocation, wrapping, readback cadence, or shader math
- Founder-pool and bootstrap behavior
- Logging and diagnostics quality
- Missing file-based or public-API regression tests

Return:
- Validated invariants
- Unsafe or weakly validated inputs
- Missing tests
- Recommended fixes and docs updates