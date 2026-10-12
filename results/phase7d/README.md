# Phase 7D — Repeat-Seed Training Stability Audit

Training completed for all five prespecified seeds. No Test data, seed selection, ensembling, or tuning was used.

## Plain-language conclusions

- Single-frame CONTROL OOF case-macro error across seeds: mean 18.067 px, SD 0.843 px, range 1.920 px.
- ARM A vs B mean `B-A` delta: 1.023 px. Independent A/B conclusion: A.
- TEMPORAL beat matched CONTROL in 5/5 OOF repeats. Mean temporal improvement was 2.034 px.
- TEMPORAL reduced persistent harmful frames in 3/5 seeds.
- On matched Val, TEMPORAL won 1/5 repeats; OOF-win/Val-loss occurred 4 times.
- Frozen primary decision: **Decision 2 — Temporal signal exists but training variance is comparable**.

## Case consistency

Temporal OOF wins by case across five seeds:

case_id
361             5
6702            5
9950            5
202607130347    5
15716           4
20260706004     4
131             1
4904            0

## Val across-seed alpha=.75 mean error

          mean       std        min        max
arm                                           
A    21.456179  1.934787  19.226141  24.081026
B    22.519643  0.672272  21.993233  23.673389
C    23.539047  1.662087  20.947102  25.019321

## Design

Five fixed base seeds were run. Each OOF fold used `effective_seed = base_seed + one_based_fold_index`, matching the historical Phase 7B convention. ARM A trained on all 2,784 frames; ARM B and C trained on the same 2,760 legal targets. All OOF and Val evaluation used the same legal target frames. Each model used ImageNet-pretrained HRNet-W18-small-v2, three fixed epochs, weighted MSE, integer argmax, and frozen alpha=.75.

Historical reuse was allowed only after config and epoch checks; details are in `reused_model_audit.csv`.
