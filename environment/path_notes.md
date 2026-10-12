# Path notes

Historical configs and CSV manifests contain absolute Windows paths to the original workspace and `C:\Users\Walter\Desktop\TIP`. The package preserves them for provenance rather than silently rewriting experimental records.

Before running elsewhere:

1. Place or mount the audited raw data separately.
2. Update copied config fields such as `manifest`, `output_dir`, and prior-phase paths.
3. Ensure the expected `scripts/`, `configs/`, and `outputs/` layout or adjust launcher paths.
4. Set `PYTHONPATH` to include the project scripts and installed packages; do not depend on the old `work/phase*_pydeps` directories.
5. Re-run split/leakage audits before training. Do not connect reserved cases 908/771 to any development inference path.

Known absolute prefixes found in the historical project include the original Codex workspace and `C:\Users\Walter\Desktop\TIP`. This handoff does not include those raw files.
