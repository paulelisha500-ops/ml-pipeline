import glob
import os

import pandas as pd

DATA_DIR = "data"
MODEL_DIR = "models"


def find_csv(data_dir):
    """Return the first CSV in the data dir, whatever Kaggle named it."""
    matches = sorted(glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True))
    if not matches:
        raise SystemExit(
            f"No CSV found under {data_dir}/. "
            "Check the dataset slug in .github/workflows/ml-pipeline.yml"
        )
    return matches[0]


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    csv_path = find_csv(DATA_DIR)
    print(f"Using dataset: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Rows: {len(df)}   Columns: {len(df.columns)}")
    print(df.head())

    summary_path = os.path.join(MODEL_DIR, "summary.csv")
    df.describe(include="all").to_csv(summary_path)
    print(f"Wrote {summary_path}")

    card_path = os.path.join(MODEL_DIR, "README.md")
    with open(card_path, "w", encoding="utf-8") as f:
        f.write("# ml-pipeline\n\n")
        f.write("Generated automatically by GitHub Actions.\n\n")
        f.write(f"- Source file: `{os.path.basename(csv_path)}`\n")
        f.write(f"- Rows: {len(df)}\n")
        f.write(f"- Columns: {', '.join(str(c) for c in df.columns)}\n")
    print(f"Wrote {card_path}")

    print("Training step complete.")


if __name__ == "__main__":
    main()
