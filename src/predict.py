"""Score passengers with the trained survival model.

    python src/predict.py <input.csv> [output.csv]

Run with no arguments to print the model's metrics.
"""

import json
import os
import sys

import joblib
import pandas as pd

from features import engineer_features

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")


def show_info():
    print(__doc__.strip())
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, encoding="utf-8") as f:
            metrics = json.load(f)
        print("\nModel metrics:")
        for key, value in metrics.items():
            print(f"  {key:28} {value}")
    else:
        print(f"\nNo metrics at {METRICS_PATH}.")


def main(argv):
    if len(argv) < 2:
        show_info()
        return 0

    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"Model not found at {MODEL_PATH}. Run train.py first.")

    in_path = argv[1]
    out_path = argv[2] if len(argv) > 2 else "predictions.csv"

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(in_path)

    out = pd.DataFrame(index=df.index)
    if "PassengerId" in df:
        out["PassengerId"] = df["PassengerId"]
    out["Survived"] = model.predict(engineer_features(df))
    out["SurvivalProbability"] = model.predict_proba(engineer_features(df))[:, 1].round(4)

    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} predictions to {out_path}")
    print(out.head())
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.exit(main(sys.argv))
