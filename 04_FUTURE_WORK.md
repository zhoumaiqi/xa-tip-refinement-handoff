# Future work

## Priority 1 — Failure phenotype review

Review persistent wrong-track and hard-rescue failures on Train/Val, separating wrong linear structure, diffuse/multi-peak heatmaps, border-like failures, and cases where raw was already correct. This is the most direct way to decide whether future supervision or architecture changes target the actual error.

## Priority 2 — Independent-case evidence

Continue acquiring and auditing independent cases. Phase 9B shows that four additional training cases improved safety but were not sufficient for a stable primary mean improvement. Preserve truly untouched cases rather than consuming every new case for development.

## Priority 3 — Temporal research only after evidence supports it

Temporal information showed a real OOF signal, but simple channel stacking, late fusion, and explicit difference inputs did not generalize consistently. If resumed, investigate alignment-aware or motion-aware feature fusion with more independent cases. Do not begin by simply increasing the channel stack from 3 to 5 or 7 frames.

## Before one-shot Final Test

Freeze in writing: training cases, preprocessing, model architecture, exact checkpoint/ensemble policy, seed policy, epochs, fixed alpha, handling of Series first frames, and all reported metrics. Then run exactly one prespecified evaluation on cases 908 and 771. Do not select a checkpoint or alpha from Test outcomes.
