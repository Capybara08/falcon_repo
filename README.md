# FALCON: Federated Assessment and Learning of Contaminants, Occurrences, and Networks of Per- and Polyfluoroalkyl Substances

FALCON is a decentralized modeling framework for predicting the fate and transport of per- and polyfluoroalkyl substances (PFAS) across geographically distributed groundwater systems.
It addresses sparse monitoring in rural or under-sampled regions by leveraging federated learning. Instead of pooling all data centrally, FALCON enables regional clusters of groundwater wells to collaboratively train models while keeping data local. This allows knowledge learned in data-rich regions to improve predictions in data-scarce areas.

## Key Features
- Decentralized PFAS fate and transport modeling
- Clustered federated learning across groundwater well networks
- Knowledge transfer between geographically distinct regions
- Designed for sparse and uneven environmental datasets
- Uses real-world California PFAS dataset (Dong et al.)
- Bayesian ML architecture
- Classifier chains for multiclass binary label PFAS prediction
- Model inputs are environmental variables; environmental-variable weights are modulated by a chemically informed hypernetwork
- Hypernetwork uses static molecular descriptors, with future work planned for richer molecular descriptors (e.g., using QM, etc)

## Project Structure
- `data/`: input datasets and generated preprocessing artifacts
- `falcon/pipeline/`: preprocessing, feature engineering, clustering, plotting, and model code
- `falcon/federated/`: federated learning protocol implementation
- `falcon/utils/`: utility functions
- `figs/`: generated figures
- `falcon/config.py`: experiment configuration
- `falcon/run.py`: launches FALCON training and evaluation
- `falcon/xgb.py`: centralized XGBoost baseline

## Installation

Set up a local virtual environment with pip:

```bash
git clone https://github.com/your-username/falcon-pfas.git
cd falcon-pfas
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Running

Run commands from the repository root.

```bash
python falcon/run.py
```

To run the centralized XGBoost baseline:

```bash
python falcon/xgb.py
```

## Reviewer Notes

Start with `REVIEW_GUIDE.md` for the recommended review path, setup options, and main files to inspect.
