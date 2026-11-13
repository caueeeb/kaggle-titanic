"""Experimento: atributos adicionais e outros modelos, com validação repetida.

Compara a linha de base do projeto (11 atributos + Regressão Logística) com
variações, usando validação cruzada estratificada repetida (5 x 5) sobre os 891
registros rotulados. Resultados são estimativas LOCAIS de validação cruzada;
não são pontuação do leaderboard do Kaggle.

Uso::

    python -m src.experiments
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    create_features,
    split_features_target,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]


def add_extra_features(df: pd.DataFrame, ticket_counts: pd.Series) -> pd.DataFrame:
    """Atributos extras. ticket_counts usa os tickets de treino + teste (sem rótulos)."""

    df = df.copy()

    df["TicketGroupSize"] = df["Ticket"].map(ticket_counts).fillna(1)
    df["FarePerPerson"] = df["Fare"] / df["TicketGroupSize"]
    df["AgeMissing"] = df["Age"].isna().astype(int)
    df["Deck"] = df["Cabin"].str[0].fillna("U")

    return df


EXTRA_NUMERIC = ["TicketGroupSize", "FarePerPerson", "AgeMissing"]
EXTRA_CATEGORICAL = ["Deck"]


def make_pipeline(classifier, numeric, categorical):
    preprocessor = ColumnTransformer(transformers=[
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])

    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def main() -> None:
    train = pd.read_csv(PROJECT_DIR / "data" / "raw" / "train.csv")
    test = pd.read_csv(PROJECT_DIR / "data" / "raw" / "test.csv")

    ticket_counts = pd.concat([train["Ticket"], test["Ticket"]]).value_counts()

    base_X, y = split_features_target(train)
    full = add_extra_features(create_features(train), ticket_counts)
    extra_X = full.drop(columns=["Survived", "PassengerId", "Name", "Ticket", "Cabin"])

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

    experiments = [
        ("Baseline: 11 features + Logistic Regression", LogisticRegression(max_iter=1000),
         base_X, NUMERIC_FEATURES, CATEGORICAL_FEATURES),
        ("+ TicketGroupSize, FarePerPerson, AgeMissing, Deck + Logistic Regression",
         LogisticRegression(max_iter=1000), extra_X,
         NUMERIC_FEATURES + EXTRA_NUMERIC, CATEGORICAL_FEATURES + EXTRA_CATEGORICAL),
        ("11 features + Random Forest", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, random_state=42),
         base_X, NUMERIC_FEATURES, CATEGORICAL_FEATURES),
        ("11 features + HistGradientBoosting", HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, random_state=42),
         base_X, NUMERIC_FEATURES, CATEGORICAL_FEATURES),
        ("+ extra features + HistGradientBoosting", HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, random_state=42),
         extra_X, NUMERIC_FEATURES + EXTRA_NUMERIC, CATEGORICAL_FEATURES + EXTRA_CATEGORICAL),
    ]

    rows = []

    for name, classifier, X, numeric, categorical in experiments:
        pipeline = make_pipeline(classifier, numeric, categorical)
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=1)

        rows.append({
            "Experiment": name,
            "Mean accuracy": scores.mean(),
            "Std (25 folds)": scores.std(),
            "Std error": scores.std() / np.sqrt(len(scores)),
        })

    table = pd.DataFrame(rows)

    out = PROJECT_DIR / "reports"
    out.mkdir(exist_ok=True)
    table.to_csv(out / "feature_experiments.csv", index=False)

    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
