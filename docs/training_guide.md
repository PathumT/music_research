# Training and evaluation

First run `python scripts/smoke_test.py`. It generates 30 explicitly synthetic tones, ten in each partition, and trains each architecture for two epochs / two batches per epoch on CPU. These targets exercise code; tones are not evidence of instrument recognition. Fixture checkpoints cannot enter the research registry. Smoke loss curves, metadata, metrics and checkpoints are under `data/smoke/results`; a portable verification summary is under `research/results/smoke_verification.json`.

Training YAML fields:

| Field | Meaning |
|---|---|
| `manifest` | Prepared `segments.json`; requires nonempty train/validation/test partitions |
| `output` / `registry` | Run artifacts directory / FastAPI checkpoint registry JSON |
| `seed` / `device` | Deterministic seed / `cpu` or `cuda` |
| `epochs` / `batch_size` | Maximum epochs / segment batch size |
| `learning_rate` / `patience` | AdamW rate / early-stop epochs without validation improvement |
| `max_batches` | Optional batch cap for train/validation during quick checks; null for full training |
| `allow_synthetic` | False for research; true only for fixtures |
| `preprocessing` | Sample rate, input duration, FFT size, hop length and Mel bands |

The loader decodes only requested offsets, averages channels, resamples, pads and computes log Mel power with per-window standardization. The same feature code serves uploads. Gain/noise augmentation applies only to training and is disabled for IRMAS; time stretching, pitch shifting, reverb and random cross-song remix augmentation are not implemented. Original Slakh mixtures supply actual polyphonic examples.

CNN uses convolution, batch normalization, pooling and global average pooling. Attention adds bidirectional GRU and learned temporal pooling. Models produce ten independent logits. Unknown targets use an explicit loss mask. AdamW, gradient clipping, validation loss scheduling, early stopping and best-checkpoint saving run on actual batches. Fusion freezes the trained CNN/attention experts and learns per-class confidence-weighted convex logit combination and bias on training data. Experts must match the manifest and preprocessing. It does not implement separated-audio fusion; do not report that research experiment as completed.

Thresholds are selected per instrument on validation data only and stored in `best.pt`. Held-out test is used only by the evaluation command. Checkpoints use `weights_only=True` loading and include taxonomy, preprocessing, data SHA256, config, parameter count, software versions, seed, epochs, optimizer update count and training time. Only load checkpoints produced by your trusted local workflow. Registry hashes detect subsequent changes.

Outputs per model: `best.pt`, `metadata.json`, `history.json`, `loss.png`, `evaluation.json`, `per_instrument.csv`. Evaluation records actual masked macro/micro F1, per-instrument precision/recall/F1/AP, TP/FP/FN/TN and segment-level mistakes. Unknown classes are excluded from scoring. Positive-only sources cannot measure false positives for unknown categories; evaluate on fully annotated sources for meaningful comparative recognition precision. Classes with no known examples have null metrics. `comparison.json` reports missing checkpoints explicitly. No transcription accuracy or separated-source results are claimed.

For a 4 GB GTX 1650, begin at batch size 4 or 8 and inspect memory usage; CPU is the default. Full 10k experiments were not run without source audio. Architecture scale and classification quality need real held-out evaluation. The serving API chunks songs but currently performs inference synchronously in a FastAPI threadpool. A production job queue, cancellation, authentication, storage database and frontend remain outside this dataset/training implementation. Do not expose this local research server directly to the Internet.
