---
name: production-stopped-protect
description: Yu-Gi-Oh AI resident training is ordered stopped; never touch production STOP/state/adopted model; experiments only in new isolated roots
metadata:
  node_type: memory
  type: project
  originSessionId: 5b47d53c-2abc-4064-ace9-c3ea6debae79
  modified: 2026-11-09T03:17:56.679Z
---

The user ordered the resident (continuous) learning for the Elfnotes+Kewl Tune AI stopped (still in force as of 2026-11-09). Production root: experiments/continuous-elfnotes-openings-20261109 (STOP content `now`, state `stopped`).

**Why:** user instruction; finite tests or code changes requested later do NOT count as permission to resume.
**How to apply:** never delete production STOP, never run `continuous_learning.py run/resume`, never touch the Windows controller/Startup. Run experiments in a new experiment root with a freshly generated snapshot; verify the production fingerprint before/after (see scripts/opening_rehearsal_ab.py `production_fingerprint`). No LLM/API calls inside training loops. Related: [[opening-rehearsal-stage-a-result]].
