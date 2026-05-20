# Mentor Review Guide

This repository contains FALCON, a federated learning workflow for PFAS groundwater modeling in California. The main implementation is in `falcon/`.

## Recommended Setup

Use a local Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Entry Points

Run commands from the repository root.

```bash
python falcon/run.py
python falcon/xgb.py
```

`falcon/run.py` launches the FALCON federated training workflow. `falcon/xgb.py` runs the centralized XGBoost baseline used for comparison.

## Files Worth Reviewing First

- `falcon/config.py`: feature lists, PFAS classes, molecular descriptors, paths, and training config.
- `falcon/run.py`: end-to-end FALCON experiment flow.
- `falcon/pipeline/preprocess.py`: data cleaning, feature engineering, and labeling.
- `falcon/pipeline/bayesModel.py`: model architecture and local/global training helpers.
- `falcon/federated/`: agent creation, topology, gossip, and federated training.
- `falcon/xgb.py`: baseline model.

## Data and Generated Outputs

The project expects input data under `data/`. Training can produce figures, run directories, logs, and pickle caches. Those outputs are intentionally ignored in `.gitignore` so the GitHub repo stays reviewable.

