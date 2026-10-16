# MusicAI Research Lab — dataset and training workflow

This workspace was empty. This implementation supplies local dataset import, grouped manifests, CPU/CUDA training, evaluation and a FastAPI serving foundation. No existing frontend was present to integrate. No real datasets or production checkpoints are bundled. Synthetic smoke outputs are explicitly excluded from the serving registry.

From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
python scripts/smoke_test.py
python -m pytest -q
```

See [dataset setup](docs/dataset_setup.md) before enabling sources. Review each license, download/extract audio and metadata, set the local roots and `license_accepted: true` in `configs/dataset_sources.yaml`. Then:

```powershell
python scripts/prepare_dataset.py --config configs/dataset_sources.yaml
python scripts/train_baseline.py --config configs/training.yaml
python scripts/train_attention.py --config configs/training.yaml
python scripts/train_fusion.py --config configs/training.yaml
python scripts/evaluate_models.py --config configs/training.yaml
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` to select a registered model and upload audio. `GET /api/models` lists verified research checkpoints; `POST /api/analysis?model=baseline` accepts a multipart `audio` file. Attention and fusion use the same API. Registry entries are created automatically only after real-data training and checkpoint inference verification. No model is ready before that succeeds.

All six requested commands implement `--help`. Five accept a required `--config` YAML path. Smoke accepts optional `--output` (default `data/smoke`). Run commands from the repository root; YAML paths resolve from that directory. See [training guide](docs/training_guide.md) and [execution report](docs/execution_report.md) for verified scope and limitations.
