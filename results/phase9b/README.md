# Phase 9B — Extended-Train Single-Frame Baseline

## Plain-language result

- Deterministic hash assigned **4274, 14-cbct, 2768, 658** to Extended Train and permanently reserved **908, 771** as Final Test. No case characteristic or result influenced this assignment.
- Train expanded from 8 cases / 12 Series / 2784 frames to **12 cases / 17 Series / 4113 frames**.
- On matched old-case OOF, Extended won **1/5 seeds**. Mean case-macro change was **+0.365 px** (negative is better). Baseline seed SD was 0.411px; Extended seed SD was 0.613px. Frozen OOF decision: **OOF-C**.
- On the frozen matched 920-frame Val, Extended won **2/5 seeds**; mean change was -0.199px.
- Final conclusion: **3**. This is not a Final-Test claim; cases 908 and 771 remain unopened for training/inference/metrics.

## Primary matched results (alpha = 0.75)

All values below are the mean across the five prespecified seeds; `±` is the sample SD across seeds.

| Evaluation | Method | Mean error | Median error | P95 | harmful >10 | Case-macro Mean |
|---|---|---:|---:|---:|---:|---:|
| Old-case OOF (2760 matched frames) | BASELINE-8 | 15.578 ± 0.515 | 9.796 ± 0.356 | 50.671 ± 2.388 | 6.40% ± 1.54% | **17.044 ± 0.411** |
| Old-case OOF (2760 matched frames) | EXTENDED-12 | 16.165 ± 0.380 | 10.261 ± 0.491 | 49.731 ± 2.241 | 4.11% ± 0.70% | **17.409 ± 0.613** |
| Frozen Val (920 matched frames) | BASELINE-8 | 21.456 ± 1.935 | 14.323 ± 1.213 | 49.548 ± 2.139 | 11.46% ± 3.22% | 18.288 ± 1.427 |
| Frozen Val (920 matched frames) | EXTENDED-12 | 21.257 ± 0.590 | 14.480 ± 0.682 | 47.853 ± 2.534 | 7.04% ± 5.21% | 17.968 ± 0.111 |

The primary OOF endpoint worsened by **0.365 px**, with Extended winning only **1/5 seeds**. Val Mean improved descriptively by **0.199 px** and Extended won **2/5 seeds**, but this does not override the pre-Val OOF-C decision. Extended showed lower Val across-seed variability and consistently better tail/safety summaries, which is useful secondary evidence but not enough to call the data expansion a reliable mean-error improvement.

## Case consistency

Extended OOF wins across five seeds:

case_id
20260706004     5
131             4
15716           3
4904            3
361             1
6702            1
9950            1
202607130347    0

Val five-seed mean error by case:

method   BASELINE_8  EXTENDED_12
case_id                         
13424     30.961979    31.126104
2148      13.521252    14.259711
6700      10.379504     8.516788

The most repeatable OOF improvements were `20260706004` (5/5 seeds) and `131` (4/5). Case `202607130347` worsened in all 5 seeds and remained the clearest difficult case for this intervention. On Val, `6700` improved by 1.863 px on average, while `13424` was essentially unchanged (+0.164 px) and `2148` worsened by 0.738 px.

## Safety, rescue, and persistent errors

- **OOF raw <5 px:** mean error improved 6.303 → 5.752 px and harmful >10 fell 12.50% → 7.47%; however harmful >5 increased 25.44% → 29.70%.
- **Val raw <5 px:** mean error improved 9.846 → 8.204 px and harmful >10 fell 26.20% → 11.74%; harmful >5 was similar/slightly worse, 49.11% → 51.55%, with substantial seed variability.
- **OOF hard frames:** mean gain decreased for raw 20–40 px (2.663 → 1.514 px) and raw 40–60 px (3.389 → 1.419 px). Thus hard-case rescue did **not** improve in matched OOF.
- **Val hard frames:** raw 20–40 px mean gain decreased 10.692 → 9.229 px, while raw 40–60 px mean gain improved 0.569 → 1.457 px. This is mixed descriptive evidence.
- **Persistent harmful behavior:** OOF harmful frames averaged 179.2 → 113.4 and persistent frames in segments ≥3 averaged 93.8 → 68.6. On Val these changed 119.0 → 64.8 and 71.0 → 35.0, respectively. The maximum OOF harmful segment did not improve (14.2 → 14.8 frames), so persistent failure was reduced in frequency but not eliminated.

## Interpretation

The prespecified final choice is **3: increasing these cases did not produce a reliable improvement in the primary cross-case mean-error endpoint; the main bottleneck cannot be attributed only to case count.** There is nevertheless a reproducible secondary safety signal: fewer severe corrections, lower P95, and fewer persistent harmful frames. A future decision about freezing an Extended model must not be based on this Val alone and requires a separately locked one-shot evaluation on the still-reserved Final Test.

The Final Test contains cases `908` and `771` (5 Series, 1668 frames). Only identity metadata was used for split/leakage checks. Training, pixel, GT, inference, and metric read counts are all zero.

## Frozen method

Single current XA frame copied to three channels; raw-centered 256×256 crop; full-frame per-frame min-max; ImageNet-pretrained `hrnet_w18_small_v2.gluon_in1k`; 64×64 heatmap; sigma 2; weighted MSE `mean((1+9*GT)*(Pred-GT)^2)`; Adam 0.001, weight decay 0; batch 16; three fixed epochs; integer argmax; alpha 0.75. No augmentation, temporal input, gate, Test access, or parameter tuning. Phase7D ARM-A baseline predictions/checkpoints were reused without seed selection.

Extended full Train has 258 optimizer steps per epoch versus 174 for the historical 8-case Train. Therefore any improvement supports the complete training-data expansion intervention (more independent cases plus more samples/updates), not a claim that case diversity alone is proven causal.

Rare visually obscured frames retain minor label uncertainty, but human review found no systematic annotation problem. No frame was deleted, relabeled, or down-weighted.
