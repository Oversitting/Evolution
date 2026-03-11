---
description: "Run a release-readiness pass for Evolution Simulator covering build, tests, docs alignment, and notable residual risks."
name: "Release Readiness"
argument-hint: "Release scope or branch goal"
agent: "Repo Auditor"
---
Perform a release-readiness pass for Evolution Simulator.

Check:
- Build status
- Test status and meaningful coverage gaps
- Example compilation status
- Documentation alignment with shipped behavior
- Config and workflow regressions that could surprise users

Return:
- Blocking issues
- Non-blocking risks
- Docs to update
- Final release recommendation