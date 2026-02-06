# Latent Space Representation of Electricity Market Curves

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This repository contains the code for the paper **"Latent Space Representation of Electricity Market Curves: Maintaining Structural Integrity"** by Martin Výboh, Zuzana Chladná, Gabriela Grmanová, and Mária Lucká.

📄 **Paper**: [arXiv preprint](https://arxiv.org/abs/2503.11294v2)

## Getting Started

### Installation

```bash
conda env create -f environment.yml

conda activate curves_env
```

### Running the Pipeline

The dimensionality reduction pipeline consists of four main steps:

1. **Train dimensionality reduction models**
```bash
python -m curves.dim-reduct.train
```
Trains PCA, kPCA, UMAP, or Autoencoder models on supply and demand curves. Model selection and hyperparameters are to be configured in `config.yml`.

2. **Generate reconstructions with moving window retraining**
```bash
python -m curves.dim-reduct.predict
```
Applies trained models to test data with periodic retraining to account for potential temporal context drifts.

3. **Apply isotonic transformation (optional)**
```bash
python -m curves.dim-reduct.isotonic_transform
```
Enforces monotonicity constraints on reconstructed curves using isotonic regression.

4. **Evaluate reconstruction quality**
```bash
python -m curves.dim-reduct.evaluate
```
Calculates RMSE, MAE, Bias, and WAPE metrics overall and by time periods (hourly, weekday).

### Configuration

Edit `config.yml` to specify:
- Dataset paths and date ranges
- Dimensionality reduction method (pca/kpca/umap/autoencoder)
- Number of components for supply and demand
- Evaluation settings (retrain interval, monotonic evaluation)

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         curves and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── environment.yml   <- The requirements file for reproducing the environment
│
├── setup.cfg          <- Configuration file for flake8
│
└── curves   <- Source code for use in this project.
    │
    ├── __init__.py
    │
    ├── autoencoder.py          <- Code with AutoEncoder model class definitions.
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── dim-reduct              <- Dimensionality reduction pipeline
    │   ├── __init__.py
    │   ├── train.py            <- Train dimensionality reduction models (PCA, kPCA, UMAP, Autoencoder)
    │   ├── predict.py          <- Generate reconstructions using trained models with moving window retraining
    │   ├── evaluate.py         <- Calculate reconstruction metrics (RMSE, MAE, Bias, WAPE) overall and by time periods
    │   └── isotonic_transform.py <- Apply isotonic regression to enforce monotonicity on reconstructed curves
    │
    └── plots.py                <- Code to create visualizations
```

--------

