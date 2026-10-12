# Phase 8A — Shared-Backbone 3-Frame Feature Fusion

Three causal frames are processed by one shared HRNet backbone. Spatial branch-0 features `[B,18,64,64]` are concatenated and fused with `Conv1x1(54→18) + ReLU` before the original heatmap head.

## Fixed design

- Crop: current `raw_t` centered 256×256 for all frames.
- Target: `true_tip_t` only.
- Loss: weighted MSE.
- Epochs: 3 fixed.
- Shrinkage: alpha=0.75 fixed.
- No future frames, new labels, augmentation, tuning, or Test.

Physical batch: 16; gradient accumulation: 1; effective batch: 16; AMP: false.
Parameters: backbone 3,860,720, fusion 990, head 19, total 3,861,729. Shared weights verified.

## Stage 1

{
  "stage1_pass": false,
  "D_case_macro_mean": 17.626943756479147,
  "E_case_macro_mean": 17.698544740234635,
  "delta_E_minus_D": 0.07160098375548785,
  "wins": 3,
  "ties": 0,
  "losses": 5,
  "noninferior_cases": 3,
  "D_harmful_gt10": 0.11920289855072463,
  "E_harmful_gt10": 0.10615942028985507,
  "harmful_gate_limit_abs_increase": 0.01,
  "harmful_gate_pass": true,
  "val_read": false,
  "test_used": false
}

Stage 1 failed; Stage 2 and Val were not run. No later phase was started.

## Plain-language result

The three frames were successfully processed separately by one shared HRNet and fused at the spatial-feature level. However, on the prespecified eight-case OOF gate, temporal ARM E did **not** beat structurally matched ARM D on the primary case-macro mean: `17.699 px` versus `17.627 px` (E was `0.072 px` worse). E won 3 cases and lost 5, so it did not meet the required 5/8 non-inferior-case condition.

Frame-weighted mean improved from `17.101 px` to `16.038 px`, and harmful >10 px fell from `11.920%` to `10.616%`, but P95 worsened from `51.862 px` to `54.100 px`. Those secondary improvements do not override the frozen primary gate.

Conclusion: this first feature-fusion temporal design did not show sufficient OOF value to justify the five-seed experiment. Stage 2 and Val were deliberately not run.
