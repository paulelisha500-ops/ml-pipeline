# ml-pipeline

Titanic survival classifier with a fully automated build. Every push to
`main` runs the whole chain:

```
Kaggle (data) -> train -> Hugging Face (model) -> Docker Hub (image)
```

GitHub Actions orchestrates it; nothing is run by hand.

## Results

| Metric | Value |
|---|---|
| CV accuracy (5-fold) | 0.827 +/- 0.040 |
| Holdout accuracy | 0.805 |
| Holdout ROC-AUC | 0.848 |
| Majority-class baseline | 0.616 |

Published to [huggingface.co/Elisha622/ml-pipeline](https://huggingface.co/Elisha622/ml-pipeline).

## Layout

| Path | Purpose |
|---|---|
| `src/features.py` | Feature engineering, stateless so it cannot leak |
| `src/train.py` | Trains, evaluates, writes `models/` |
| `src/predict.py` | Scores a CSV with the saved model |
| `Dockerfile` | Packages code + trained model |
| `.github/workflows/ml-pipeline.yml` | The pipeline |

## The model

A random forest over passenger class, sex, age, family counts, fare,
port of embarkation, title parsed from the name, family size, an
is-alone flag, and the cabin deck letter.

Preprocessing and the classifier are one scikit-learn `Pipeline`, so
imputation is fitted on training folds only — the holdout score is not
inflated by leakage, and the saved artifact cannot drift out of sync
with the preprocessing it was trained under.

## Using the image

Print the model's metrics:

```bash
docker run elisha622/ml-pipeline:latest
```

Score your own CSV:

```bash
docker run -v "$PWD":/work elisha622/ml-pipeline:latest \
  python src/predict.py /work/input.csv /work/output.csv
```

## Running locally

```bash
pip install -r requirements.txt
kaggle datasets download -d yasserh/titanic-dataset -p data/ --unzip
python src/train.py
```

## Required repository secrets

| Secret | Purpose |
|---|---|
| `KAGGLE_USERNAME` / `KAGGLE_KEY` | Dataset download |
| `HF_TOKEN` | Hugging Face write token |
| `DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN` | Image push (needs Read & Write) |

## Changing the dataset

Edit the slug in the workflow. `train.py` finds whichever CSV arrives in
`data/`, but the feature engineering is Titanic-specific — a different
dataset needs `src/features.py` rewritten to match its columns.

## Notes

The runner is pinned to `ubuntu-24.04` rather than `ubuntu-latest`, so
the Ubuntu 26 migration on 2026-10-19 cannot change the build
environment without an explicit commit.
