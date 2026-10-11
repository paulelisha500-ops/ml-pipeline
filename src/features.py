"""Feature engineering for the Titanic survival dataset.

Kept in its own module so the saved model artifact can import it at
load time (joblib stores a reference to the function, not its code).
"""

import re

import numpy as np
import pandas as pd

TITLE_RE = re.compile(r",\s*([^\.]+)\.")

COMMON_TITLES = {"Mr", "Mrs", "Miss", "Master"}

NUMERIC_FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone"]
CATEGORICAL_FEATURES = ["Sex", "Embarked", "Title", "Deck"]


def _extract_title(name):
    if not isinstance(name, str):
        return "Unknown"
    match = TITLE_RE.search(name)
    if not match:
        return "Unknown"
    title = match.group(1).strip()
    # Normalise the handful of equivalent spellings in the dataset.
    title = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}.get(title, title)
    return title if title in COMMON_TITLES else "Rare"


def engineer_features(df):
    """Derive model features from the raw Titanic columns.

    Pure and stateless: it must not learn anything from the data, or
    fitting it on the full frame would leak the test split.
    """
    out = pd.DataFrame(index=df.index)

    for col in ["Pclass", "Age", "SibSp", "Parch", "Fare"]:
        out[col] = pd.to_numeric(df[col], errors="coerce") if col in df else np.nan

    out["FamilySize"] = out["SibSp"] + out["Parch"] + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)

    out["Sex"] = df["Sex"] if "Sex" in df else "Unknown"
    out["Embarked"] = df["Embarked"] if "Embarked" in df else "Unknown"
    out["Title"] = df["Name"].map(_extract_title) if "Name" in df else "Unknown"
    # Cabin is ~77% missing; the deck letter still carries signal.
    out["Deck"] = df["Cabin"].astype(str).str[0] if "Cabin" in df else "Unknown"
    out.loc[out["Deck"].isin(["n", "N"]), "Deck"] = "Unknown"

    for col in CATEGORICAL_FEATURES:
        out[col] = out[col].fillna("Unknown").astype(str)

    return out[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
