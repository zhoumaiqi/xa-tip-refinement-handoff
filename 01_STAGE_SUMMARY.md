# Stage summary

## Executive conclusion

The strongest supported engineering baseline is Phase 6G single-frame local refinement with fixed alpha 0.75. Later work identified temporal signal and improved safety, but neither temporal representations nor expanded training cases produced a stable primary cross-case mean-error improvement. Extended-12 with alpha 0.50 is a conservative candidate for future locked comparison, not a promoted baseline.

## Evidence chain

1. **Full-frame baseline:** fitted Train but failed held-out cases severely. Diagnostics implicated position shortcuts, visual/case domain shift, sparse-heatmap MSE weakness, and limited independent-case diversity.
2. **Local refinement:** project semantics established `raw_*` as deployable coarse localization. Raw-centered 256 crops had sufficient observed coverage and enabled a useful local HRNet.
3. **Phase 6G:** fixed alpha 0.75 gave the best reliable safety/rescue compromise available at that stage. It remains the baseline checkpoint in this handoff.
4. **Adaptive gates and residual models:** observable single-frame features did not select correction strength robustly under strict case nesting.
5. **Temporal work:** simple causal stacks repeatedly helped OOF but reversed on Val. Feature fusion failed Stage 1; explicit differences also failed independent Val stability. Temporal work remains research-only.
6. **Data expansion:** Phase 9A/9A.5 established six new technically and visually acceptable cases. Phase 9B used four for training and sealed two as Test. More cases reduced severe/persistent harm but did not reliably improve matched OOF mean.
7. **Alpha recalibration:** alpha 0.50 was preferred for Extended-12 OOF and improved tail safety, but frozen-Val mean was marginally worse; alpha is a trade-off knob, not a root-cause fix.

## Current measured anchors

- Phase 6G frozen Val: raw Mean 22.299 px; full correction about 20.008 px; fixed alpha .75 about 20.003 px, Median 13.753 px, P95 48.976 px.
- Phase 9B matched old-case OOF case-macro Mean: Baseline-8 17.044 ± 0.411 px; Extended-12 17.409 ± 0.613 px.
- Phase 9B frozen matched Val Mean: Baseline-8 21.456 ± 1.935 px; Extended-12 21.257 ± 0.590 px; descriptive only, and the OOF-C conclusion remains controlling.
- Phase 9C Extended-12: alpha .75 OOF macro 17.409 px; alpha .50 17.072 px. Frozen-Val Mean 21.257 px at .75 vs 21.315 px at .50, while P95 and severe-harm metrics improved at .50.

## What is production-relevant versus research-only

- **Engineering anchor:** Phase 6G configuration/checkpoint and raw-centered single-frame inference.
- **Traceable candidates:** five Phase 9B full-train checkpoints. They were not selected using Final Test.
- **Research-only:** temporal stacks/fusion/differences, adaptive gates, and Extended alpha .50 until an independently frozen evaluation justifies promotion.
