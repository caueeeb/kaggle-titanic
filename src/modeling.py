"""Pipeline, validação e comparação de modelos do Titanic."""

from __future__ import annotations

import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

RANDOM_STATE = 42


def build_preprocessor() -> ColumnTransformer:
    """Imputação + padronização (numéricas) e imputação + one-hot (categóricas)."""

    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(transformers=[
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])


def build_pipeline(classifier=None) -> Pipeline:
    """Pipeline completo (padrão: Regressão Logística, como no notebook)."""

    if classifier is None:
        classifier = LogisticRegression(max_iter=1000)

    return Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("classifier", classifier),
    ])


def candidate_models() -> dict:
    """Classificadores comparados na validação cruzada."""

    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, random_state=RANDOM_STATE
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=2, random_state=RANDOM_STATE
        ),
        "SVM": SVC(C=1.0, kernel="rbf"),
    }


def holdout_evaluation(X, y, test_size: float = 0.20):
    """Treina na parte de treino e avalia no holdout estratificado 80/20."""

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    model = build_pipeline().fit(X_train, y_train)
    predictions = model.predict(X_val)

    metrics = {
        "accuracy": accuracy_score(y_val, predictions),
        "precision": precision_score(y_val, predictions),
        "recall": recall_score(y_val, predictions),
        "f1": f1_score(y_val, predictions),
        "confusion_matrix": confusion_matrix(y_val, predictions).tolist(),
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
    }

    return model, metrics, (X_train, X_val, y_train, y_val)


def compare_models(X_train, y_train, n_splits: int = 5) -> pd.DataFrame:
    """Validação cruzada estratificada de todos os candidatos (somente no treino)."""

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"accuracy": "accuracy", "f1": "f1", "roc_auc": "roc_auc"}

    rows = []

    for name, classifier in candidate_models().items():
        pipeline = Pipeline(steps=[
            ("preprocessor", clone(build_preprocessor())),
            ("classifier", classifier),
        ])

        scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)

        rows.append({
            "Model": name,
            "Mean accuracy": scores["test_accuracy"].mean(),
            "Accuracy std": scores["test_accuracy"].std(),
            "Mean F1": scores["test_f1"].mean(),
            "Mean ROC-AUC": scores["test_roc_auc"].mean(),
        })

    return pd.DataFrame(rows).sort_values("Mean accuracy", ascending=False).reset_index(drop=True)
