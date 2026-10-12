# Phase 8B — Explicit Temporal-Difference Input

## Plain-language result

Giving the model signed change maps helped in some Train-case OOF runs, but it was not reliable: G won **3/5** OOF seeds and was on average **+0.235 px** relative to F (positive means worse). On the independent matched Val cases, G won **0/5** seeds and was on average **+2.584 px worse**. Therefore the frozen conclusion is **2 — explicit differences have partial benefit but remain cross-case unstable**.

## Frozen design and audit

- Train-only `DIFF_SCALE` (pooled |D1|/|D2| P99.5): **0.54450839**.
- F: `[current, 0.5, 0.5]`; G: `[current, signed D1, signed D2]` encoded around 0.5.
- Shared current `raw_t` crop, 256×256; weighted heatmap MSE; 3 epochs; fixed alpha=.75.
- Stage 0 clipping stayed below the prespecified warning threshold.
- Paired initialization hashes matched for every F/G run. No NaN, collapse, or coordinate bug was observed.
- OOF was frozen before Val. **Test was not used.**

## Stage 1

- PASS: F/G case-macro Mean **17.192/17.076 px**; G wins/ties/losses **5/0/3**.
- Harmful >10 decreased from **9.13%** to **4.49%**.

## Five-seed OOF (mean ± SD)

- Case-macro Mean: F **17.536 ± 0.288**, G **17.771 ± 0.766 px**.
- Frame Mean: F **16.256**, G **16.070 px**; P95: F **51.639**, G **50.254 px**.
- Harmful >10: F **8.06%**, G **6.85%**.
- OOF wins were 3/5, but seed 20260922 had a large G regression; the case-macro average therefore did not improve.

## Matched Val, 920 frames (mean ± SD over 5 seeds)

- Mean: F **21.447 ± 2.897**, G **24.031 ± 2.910 px**.
- Median: F **14.168**, G **19.538 px**.
- P95: F **49.723**, G **51.788 px**.
- Harmful >10: F **17.50%**, G **19.22%**.
- G won 0/5 seeds; OOF-win/Val-loss reversal occurred in **3** seeds.

## Interpretation

OOF easy-case safety improved and OOF P95 decreased, but hard-case rescue was mixed and independent-Val easy safety, hard rescue, and persistent failures worsened. A frame difference is not catheter motion: it can contain patient/C-arm/background/contrast/noise changes. The evidence supports only that this representation sometimes helps; it does not support adopting it as the temporal baseline.

No Phase 8C, longer window, gate, alpha tuning, new normalization, augmentation, or Test was run.
