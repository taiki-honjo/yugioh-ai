---
name: opening-rehearsal-stage-a-result
description: 2026-11-09 result of PPO + train-split opening rehearsal A/B; retention not achieved; PPO direction causes the loss; placement adapter bug found
metadata:
  node_type: memory
  type: project
  originSessionId: 5b47d53c-2abc-4064-ace9-c3ea6debae79
  modified: 2026-11-09T03:18:02.648Z
---

Stage A (handoff change A) was run on 2026-11-09 in experiments/opening-rehearsal-ab-20261109 (report: docs/opening-rehearsal-ab-2026-11-09.md). With 256 decisions × 3 seeds, validation openings (baseline 65/66) were: A (plain PPO) 57/45/60; rehearsal λ=1 57/56/64; λ=5 60/62/63. The 9 candidates were all rejected by case_id retention, and none was adopted.

Diagnostic: a random-sign perturbation with the same |Δ| lost 0 cases, while half the PPO step lost 1–2. So the regression comes from the PPO gradient direction, not from fragility or a train/inference path mismatch (parity is covered by tests). Teacher-forced CE barely moves even when free-run openings break, so CE alone is not a retention metric.

Found a pre-existing bug, not fixed: center_placement.py PlacementMemory.observe does `remaining.remove(code)` and raises ValueError after Remix when the chosen card was not recorded. A trained candidate triggered it during a battle evaluation.

**Why:** the next step (larger λ, lower LR/fewer updates, KL, freezing) needs the user's decision, and stage B (multi-landing evaluation) is design-only: docs/interruption-landing-evaluation-design-2026-11-09.md.
**How to apply:** don't proceed to 4096-step or resident runs until retention passes. Compare one change at a time in new roots. Related: [[production-stopped-protect]].
