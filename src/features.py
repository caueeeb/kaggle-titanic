"""Engenharia de atributos do Titanic (mesma lógica do notebook)."""

from __future__ import annotations

import pandas as pd

# Colunas removidas antes do modelo (identificadores e texto bruto)
DROP_COLUMNS = ["Survived", "PassengerId", "Name", "Ticket", "Cabin"]

NUMERIC_FEATURES = [
    "Pclass", "Age", "SibSp", "Parch", "Fare",
    "FamilySize", "IsAlone", "HasCabin",
]

CATEGORICAL_FEATURES = ["Sex", "Embarked", "Title"]

TITLE_ALIASES = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}

# Nota: o nome do passageiro 760 gera o título "the Countess", que não coincide com
# "Countess" abaixo e portanto não é agrupado como "Rare". A regra original do notebook
# foi mantida de propósito para que o modelo e a submissão sigam reproduzíveis.
RARE_TITLES = [
    "Don", "Rev", "Dr", "Major", "Lady", "Sir", "Col",
    "Capt", "Countess", "Jonkheer", "Dona",
]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria FamilySize, IsAlone, HasCabin e Title a partir das colunas originais."""

    df = df.copy()

    # Tamanho da família (o próprio passageiro + parentes a bordo)
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    # Passageiro viajando sozinho
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Existência de cabine registrada
    df["HasCabin"] = df["Cabin"].notna().astype(int)

    # Título extraído do nome ("Braund, Mr. Owen Harris" -> "Mr")
    df["Title"] = df["Name"].str.extract(r",\s*([^.]*)\.", expand=False)
    df["Title"] = df["Title"].replace(TITLE_ALIASES)
    df["Title"] = df["Title"].replace(RARE_TITLES, "Rare")

    return df


def split_features_target(df: pd.DataFrame):
    """Separa X (11 colunas de entrada) e y a partir de um DataFrame de treino."""

    prepared = create_features(df)

    X = prepared.drop(columns=DROP_COLUMNS)
    y = prepared["Survived"]

    return X, y


def prepare_test_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara o conjunto de teste da competição (sem a coluna Survived)."""

    prepared = create_features(df)

    return prepared.drop(columns=[c for c in DROP_COLUMNS if c in prepared])
