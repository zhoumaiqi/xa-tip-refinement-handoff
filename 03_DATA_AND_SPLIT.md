# Data and split

## Coordinate and label authority

- `x = col`, `y = row`, origin at top-left, in original DICOM pixel coordinates.
- `raw_*`: product automatic coarse localization; available at deployment and permitted as crop center/input.
- `true_tip_*`: manually corrected final tip; supervision/evaluation only.
- CSV is the tip-label authority. JSON was auxiliary metadata and a JSON mismatch alone was not an exclusion reason.

## Audited inventory

The historical raw inventory contained 19 top-level cases, 57 physical XA Series, and 12,752 frames. Thirty-six Series had primary CSV labels. Raw images are external and are not duplicated here. Phase 9A inventory/mapping/eligibility tables are under `results/phase9a/`.

## Frozen partitions

- Original Train: 8 cases, 12 Series, 2,784 frames.
- Extended new Train: 4274, 14-cbct, 2768, 658; 5 Series, 1,329 frames.
- Extended total: 12 cases, 17 Series, 4,113 frames.
- Frozen Val: 13424, 2148, 6700; 4 Series, 928 frames.
- Reserved Final Test: 908, 771; 5 Series, 1,668 frames.

The six reviewed candidates were split before Phase 9B using SHA256 of `phase9b|20260924|<exact_case_id>`: first two hashes to Test and remaining four to Extended Train. The frozen split files and leakage audit are copied under `results/phase9b/` and `configs/`.

## Human review and conflict policy

All ten candidate Series across six new cases passed the user's visual review with no systematic frame-label shift, row/col swap, sustained drift, or wrong-structure labeling. Rare shadow/occlusion ambiguity remained within expected manual-label uncertainty; frames were neither deleted nor relabeled. `6700/XA-03` had conflicting annotations and remained excluded; 6700 stayed frozen Val.

## Portability

The package does not contain raw DICOM or the full 6.3 MB per-frame pixel-readability table. It includes compact inventory, mapping, eligibility, split, leakage, and candidate-manifest records. Reproduction on another machine requires reconnecting the audited external data tree and checking hashes/identities before use.
