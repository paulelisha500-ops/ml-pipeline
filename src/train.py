"""Train a Titanic survival classifier and write the model artifact.

Run by GitHub Actions after the Kaggle dataset is downloaded into data/.
Everything it produces lands in models/, which is then published to
Hugging Face by push_hf.py.
"""

import glob
import json
import os
import shutil
import sys

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, engineer_features

DATA_DIR = "data"
MODEL_DIR = "models"
TARGET = "Survived"
RANDOM_STATE = 42


def find_csv(data_dir):
    matches = sorted(glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True))
    if not matches:
        raise SystemExit(
            f"No CSV found under {data_dir}/. "
            "Check the dataset slug in .github/workflows/ml-pipeline.yml"
        )
    return matches[0]


def build_pipeline():
    """Preprocessing and model as one object, so nothing can drift apart.

    Imputation is inside the pipeline on purpose: fitted on the training
    fold only, so the test split never informs the medians.
    """
    numeric = Pipeline([("impute", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
        ]
    )
    pre = ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("pre", pre), ("model", model)])


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    csv_path = find_csv(DATA_DIR)
    print(f"Dataset: {csv_path}")
    raw = pd.read_csv(csv_path)
    print(f"Rows: {len(raw)}   Columns: {len(raw.columns)}")

    if TARGET not in raw.columns:
        raise SystemExit(
            f"Column '{TARGET}' not found. Columns present: {list(raw.columns)}. "
            "This trainer expects the Titanic training set, which includes labels."
        )

    X = engineer_features(raw)
    y = raw[TARGET].astype(int)
    print(f"Features: {X.shape[1]}   Positive rate: {y.mean():.4f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    pipe = build_pipeline()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
    print(f"CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
        "cv_accuracy_std": round(float(cv_scores.std()), 4),
        "holdout_accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "holdout_precision": round(float(precision_score(y_test, pred)), 4),
        "holdout_recall": round(float(recall_score(y_test, pred)), 4),
        "holdout_f1": round(float(f1_score(y_test, pred)), 4),
        "holdout_roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "majority_class_baseline": round(float(max(y.mean(), 1 - y.mean())), 4),
    }

    print()
    print(classification_report(y_test, pred, target_names=["died", "survived"]))
    print(f"Holdout accuracy: {metrics['holdout_accuracy']:.4f}  "
          f"(baseline {metrics['majority_class_baseline']:.4f})")

    model_path = os.path.join(MODEL_DIR, "model.joblib")
    joblib.dump(pipe, model_path)
    print(f"Wrote {model_path}")

    with open(os.path.join(MODEL_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    raw.describe(include="all").to_csv(os.path.join(MODEL_DIR, "summary.csv"))

    # Ship the feature code with the model: joblib stores a reference to
    # engineer_features, so the artifact is unusable without it.
    shutil.copy(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "features.py"),
        os.path.join(MODEL_DIR, "features.py"),
    )
    write_model_card(csv_path, raw, metrics)

    print("Training complete.")


def write_model_card(csv_path, raw, metrics):
    card = f"""---
license: mit
tags:
  - tabular-classification
  - titanic
  - scikit-learn
---

# Titanic survival classifier

Random forest predicting passenger survival. Trained automatically by
GitHub Actions; see the pipeline at
https://github.com/paulelisha500-ops/ml-pipeline

## Results

| Metric | Value |
|---|---|
| CV accuracy (5-fold) | {metrics['cv_accuracy_mean']:.4f} +/- {metrics['cv_accuracy_std']:.4f} |
| Holdout accuracy | {metrics['holdout_accuracy']:.4f} |
| Holdout precision | {metrics['holdout_precision']:.4f} |
| Holdout recall | {metrics['holdout_recall']:.4f} |
| Holdout F1 | {metrics['holdout_f1']:.4f} |
| Holdout ROC-AUC | {metrics['holdout_roc_auc']:.4f} |
| Majority-class baseline | {metrics['majority_class_baseline']:.4f} |

Trained on {metrics['n_train']} rows, evaluated on {metrics['n_test']} held out
with a stratified split.

## Features

Derived from the raw columns: passenger class, sex, age, siblings/spouses,
parents/children, fare, port of embarkation, title parsed from the name,
family size, an is-alone flag, and the deck letter from the cabin.

Imputation sits inside the model pipeline, so it is fitted on training
folds only and the evaluation is not optimistic.

## Usage

```python
import joblib, pandas as pd
from features import engineer_features

model = joblib.load("model.joblib")
df = pd.read_csv("{os.path.basename(csv_path)}")
predictions = model.predict(engineer_features(df))
```

`features.py` must be importable — it ships in the repository alongside
the training code.

## Source data

`{os.path.basename(csv_path)}` — {len(raw)} rows, {len(raw.columns)} columns.
"""
    with open(os.path.join(MODEL_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(card)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
