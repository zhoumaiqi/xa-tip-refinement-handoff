# Recommended Research Roadmap

## Purpose

This document turns the current Phase 6G / Phase 9 findings into a practical
next-step research plan. It is intended for work using Train and frozen Val
only. Reserved Final Test cases `908` and `771` remain sealed until one final
policy is frozen in writing.

## Starting point

The current engineering anchor is the Phase 6G-style single-frame local
refiner:

```text
Current XA frame
  -> raw-centered 256 x 256 crop
  -> ImageNet-pretrained HRNet-W18-small-v2
  -> 64 x 64 one-channel heatmap
  -> integer argmax candidate
  -> final = raw + 0.75 x (candidate - raw)
```

It improves average performance in some cases, but key unresolved failure
modes are cross-case variation, wrong-track / wrong-structure predictions,
over-correction of already-good raw points, and inconsistent hard-case rescue.

Do not assume a larger network alone will solve these issues.

## Guiding principles

1. Diagnose a failure before selecting an architecture intended to solve it.
2. Treat independent cases, not individual frames, as the main unit of
   generalization evidence.
3. Freeze each experimental comparison before reading frozen Val.
4. Track mean error and safety together: median, P95/P99, easy-case harm,
   hard-case gain, and persistent harmful segments.
5. Do not access Final Test while selecting models, thresholds, alpha,
   augmentation, seeds, checkpoints, or stopping rules.
6. Change one primary scientific variable per experiment whenever feasible.

## Phase A: Build a failure-phenotype review set

### Goal

Turn aggregate metrics into an understanding of why predictions fail.

### Work

Using only permitted Train/Val frames, create per-frame diagnostic panels with:

```text
Full XA frame: raw, candidate, final, true tip
Local 256 x 256 crop: raw, candidate, final, true tip
Predicted heatmap: top peak, independent second peak, peak margin, entropy
Metadata: case, series, frame, raw error, final error, gain/loss
```

Review and label a representative set of:

- successful hard-case rescues;
- harmful corrections where raw was already good;
- persistent harmful segments;
- multi-peak heatmaps;
- wrong-track / wrong-structure predictions;
- border or padding-adjacent failures;
- low-contrast, occluded, or ambiguous frames.

### Deliverable

A failure table with a stable label taxonomy and example images for each class.

### Decision use

- Many `raw-already-good` errors: prioritize rejection / conservative routing.
- Many `multi-peak` or `wrong-track` errors: prioritize structural supervision
  or candidate verification.
- Many low-contrast/occlusion errors: consider image-quality signals, data
  acquisition diversity, and uncertainty-aware review.
- Many border failures: revisit crop geometry or padding behavior.

## Phase B: Low-risk localization improvements

### Goal

Reduce small localization error without changing the clinical problem or
claiming to solve wrong-track failures.

### Candidate experiments

1. **Heatmap plus local offset head**

   Predict:

   ```text
   64 x 64 heatmap
   + local dx, dy correction near the selected peak
   ```

   The heatmap chooses the approximate 4 x 4 input-pixel cell. The offset head
   predicts the continuous within-cell adjustment.

2. **Higher-resolution heatmap**

   Compare the current `64 x 64` output with `128 x 128`, keeping the
   raw-centered ROI and all data splits unchanged.

3. **Continuous coordinate decoding**

   Compare integer argmax with a carefully validated soft-argmax or local
   peak-fitting decoder. Maintain explicit safeguards against diffuse or
   multi-peak maps.

### Evaluation

Use the same case-level OOF design and frozen Val protocol as the baseline.
Report whether any gain is concentrated in low-error cases, hard cases, or
only frame-weighted averages.

### Expected limitation

These methods can reduce heatmap quantization error. They cannot, by
themselves, prevent a confident prediction on the wrong catheter-like
structure.

## Phase C: Confidence, uncertainty, and safe non-correction

### Goal

Let the system identify frames where it should not strongly move the raw
coordinate.

### Inputs to study

- heatmap peak value;
- independent first-versus-second peak margin;
- heatmap entropy;
- distance from raw to candidate;
- candidate proximity to ROI border;
- disagreement among independently trained model seeds;
- image quality or visibility features.

### Recommended formulation

Do not begin with a per-case adaptive alpha rule. Instead evaluate a
predefined two-action policy:

```text
Trusted prediction: apply the fixed correction policy.
Untrusted prediction: retain raw or route to human review.
```

Train and calibrate the trust decision only with nested case-level OOF
predictions. Frozen Val is confirmation only.

### Success criteria

The policy should lower severe harm and persistent harmful segments without
unacceptably removing hard-case rescue. Report coverage: what fraction of
frames receive correction, retain raw, or request review.

## Phase D: Address wrong-track failures with structural evidence

### Goal

Move beyond "find a point that looks like a tip" toward "find the tip of the
correct instrument trajectory."

### Option 1: Multi-task model

Use a shared encoder with separate heads for:

```text
1. Tip heatmap
2. Short catheter / guidewire centerline probability map
3. Local direction vector or orientation map
4. Optional visibility / quality score
```

The point, line, and direction predictions should be geometrically
consistent. This gives the model more evidence to distinguish the target
instrument from nearby similar structures.

### Option 2: Top-K candidate verifier

```text
Heatmap proposes several separated peaks
  -> a lightweight verifier scores each candidate's local line continuity,
     orientation, and relation to raw
  -> select the best verified candidate or retain raw
```

This is especially relevant if Phase A finds many multi-peak heatmaps.

### Data strategy

Begin with a focused annotation set: label a short target centerline segment
and/or local direction for 100-300 representative difficult frames. Do not
start by requiring complete segmentation masks for every frame.

## Phase E: Data and augmentation for cross-case generalization

### Goal

Teach the model to remain reliable across acquisition conditions and
independent cases.

### Data priority

Acquire and audit more independent cases. Preserve a genuinely untouched set;
do not consume every new case as development data.

### Domain-specific augmentation candidates

- intensity and contrast changes;
- gamma variation;
- XA-like noise and modest blur;
- local occlusion or artifact simulation;
- small geometric transformations applied consistently to image, raw, and
  label;
- sampling strategies that expose difficult phenotypes more often.

Each augmentation family should be introduced in a single-variable experiment.
Avoid generic augmentation combinations that make causality impossible to
interpret.

### Self-supervised pretraining

If additional non-Test, unlabelled XA frames are available, pretrain the
encoder on XA before supervised tip fine-tuning. A masked-image reconstruction
or contrastive objective can teach the backbone XA-specific texture, noise,
and line-structure statistics that ImageNet photographs do not contain.

Do not train a large vision transformer from scratch on the current labelled
data volume alone.

## Phase F: Resume temporal modeling only with alignment

### Starting evidence

Simple three-frame stacking, late fusion, and explicit frame differences
showed OOF signal but did not generalize reliably to frozen Val.

### Recommended direction

Do not add five or seven raw frames as channels. If temporal research resumes,
use causal alignment-aware fusion:

```text
Current frame is the target.
Past-frame features are aligned to the current frame.
Only aligned evidence is fused into the current-frame prediction.
```

Candidate methods include local correlation, deformable convolution or
attention, learned optical-flow-style feature warping, and sequence models
that expose temporal confidence rather than forcing every frame to use history.

### Entry condition

Start this phase only after Phase A identifies a meaningful fraction of
failures that could plausibly be resolved by motion continuity, and after
additional independent cases are available.

## Recommended execution order

1. Phase A: diagnostic panels and failure taxonomy.
2. Phase C: uncertainty / reject-or-correct policy using existing model seeds.
3. Phase B: heatmap plus offset head or higher-resolution output.
4. Phase D: structural supervision or Top-K candidate verification.
5. Phase E: new independent cases, XA-specific augmentation, and
   self-supervised XA pretraining.
6. Phase F: alignment-aware temporal research.

## Experiments to avoid for now

- Selecting models, alpha, thresholds, checkpoints, or seeds using Final Test.
- Blindly replacing HRNet with a much larger Transformer.
- Adding more raw temporal channels without spatial alignment.
- Reporting frame-level gains while ignoring case-level failure.
- Tuning a separate alpha for each validation case.
- Reusing the same frozen Val repeatedly without a prespecified protocol.

## Before one-shot Final Test

Freeze the complete policy in writing:

```text
Training cases and labels
Preprocessing and crop geometry
Model architecture and initialization
Training duration and checkpoint rule
Seed or ensemble rule
Heatmap decoder / offset logic
Confidence or review policy
Fixed alpha or correction policy
Series-first-frame handling
Metrics and subgroup reports
```

Then evaluate once on cases `908` and `771`. A disappointing result must not
trigger Test-guided model selection.
