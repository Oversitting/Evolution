---
description: "Deep feature audit for Evolution Simulator with docs-vs-code-vs-tests verification."
name: "Feature Audit"
argument-hint: "Feature, subsystem, or milestone to audit"
agent: "Repo Auditor"
---
Audit the requested Evolution Simulator feature or subsystem.

Requirements:
- Do not assume the documentation is accurate.
- Verify implementation in Rust and WGSL.
- Check whether unit tests, integration tests, or example programs actually validate the feature.
- Identify missing edge cases and test permutations.
- Call out docs that should be updated.

Return:
- Verified behavior
- Mismatches or stale docs
- Coverage gaps
- Concrete next changes