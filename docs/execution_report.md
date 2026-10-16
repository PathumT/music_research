# Execution report — 2026-10-16

The supplied workspace `F:\Music Research` was empty. There was no existing repository code, frontend, raw dataset directory or trained checkpoint to preserve. The pasted master prompt was read before implementation. Work here focuses on the requested dataset/model workflow and a serving API foundation.

Verified environment: Windows, Python 3.14.7, PyTorch 2.14.1 CPU execution. Dependencies were installed in `.venv`; exact installed distributions are captured in `requirements-lock.txt`. That lock was tested here, not on Linux or other Python versions. No CUDA training was executed.

Executed successfully:

- `python scripts/smoke_test.py`: 30 synthetic segments, 10 per split. CNN, attention and confidence-guided expert fusion each trained for 2 epochs, 2 batches per epoch (4 optimizer updates). Verified loss gradients, learned parameter changes, saved/reloaded checkpoints, finite inference, held-out fixture evaluation, split-leakage rejection and synthetic-checkpoint exclusion from the registry. Also checked the API model list and unavailable-model response.
- `python -m pytest -q`: **12 passed**. Covers all five official-format adapter contracts, IRMAS original-recording map requirement, conservative NSynth mapping, Slakh mixture/stem MIDI alignment, MAESTRO compositions, MedleyDB stems, reproducible manifests, full fixture loader traversal, corrupt/missing audio, traversal rejection, masked BCE, actual optimizer parameter changes, metric masks/confusion counts, feature edge cases and frozen fusion experts.
- All six required scripts' `--help` calls completed successfully. Training/evaluation implementation is exercised by smoke through the same functions invoked by their CLI entry points.
- Python source compilation passed. The smoke test and fixture tests initially exposed duplicate-waveform split leakage in the smoke generator and NumPy MIDI integer JSON serialization; both were corrected before successful verification.

Actual smoke artifacts:

- `data/smoke/results/{baseline,attention,fusion}/best.pt`
- Each model directory includes `history.json`, `loss.png`, `metadata.json`, `evaluation.json` and `per_instrument.csv`.
- `data/smoke/results/comparison.json` and `data/smoke/verification.json`
- Portable summary: `research/results/smoke_verification.json`

These are synthetic plumbing results, not research accuracy. The fixture test F1 values are intentionally not presented as instrument-recognition performance. All smoke checkpoints are marked synthetic and cannot enter the production registry or analyze songs through the research serving function.

Executed with the requested default dataset config:

```text
python scripts/prepare_dataset.py --config configs/dataset_sources.yaml
```

The command wrote `data/manifests/report.json`, `segments.json`, split JSON and split CSV files, then exited nonzero with `No research audio prepared`. All five configured local source roots were unavailable. Actual prepared research counts: **0 segments; train 0; validation 0; test 0; target shortfall 10,000**. Empty manifests are diagnostic outputs, not a usable dataset.

Not executed or not completed:

- Downloading full multi-GB datasets. MedleyDB full audio requires provider access; IRMAS also needs a verified recording-group map and care with its restrictive provider terms. Official public NSynth/MAESTRO download helpers are implemented, but large archive transfers and extraction were not exercised.
- Preparing approximately 10,000 real labeled segments, verifying source-format adapters against complete downloaded releases, or checking real-corpus composition overlap. Adapter verification used generated format-contract fixtures, not downloaded dataset examples.
- Real-data baseline, attention or fusion training, full-dataset evaluation, any claim of model accuracy or improvement, or production checkpoint creation. Default training requires all three nonempty prepared partitions and refuses the current empty corpus.
- Separated-source fusion, MedleyDB frame activity import, transcription training/evaluation, pedal-aware acoustic-tail labels, advanced augmentations and cross-corpus composition matching. Original aligned MIDI paths and clipped notes are preserved where present.
- Frontend integration against an existing application: none was present. FastAPI endpoints and the checkpoint registry contract are implemented; the registry remains empty until actual research training succeeds. A frontend, persistent SQL import database, authenticated job administration and background queue are not built here.

Next reproducible step: follow `docs/dataset_setup.md`, supply downloaded audio/metadata and verified group identities, configure source roots and accepted licenses, then rerun preparation. Inspect the report, class distribution and split audit before using the documented training commands with `max_batches: null` and `allow_synthetic: false`.
