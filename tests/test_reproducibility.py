"""Verifica os números do README com o dado real (ignorado se data/raw/ não existir)."""

from pathlib import Path

import pandas as pd
import pytest

from src.features import split_features_target
from src.modeling import holdout_evaluation

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

pytestmark = pytest.mark.skipif(
    not (DATA_DIR / "train.csv").is_file(),
    reason="Kaggle Titanic data not available in data/raw/",
)


def test_holdout_metrics_match_readme():
    train = pd.read_csv(DATA_DIR / "train.csv")
    X, y = split_features_target(train)

    _, metrics, _ = holdout_evaluation(X, y)

    assert metrics["train_rows"] == 712
    assert metrics["validation_rows"] == 179
    assert round(metrics["accuracy"], 4) == 0.8324
    assert metrics["confusion_matrix"] == [[97, 13], [17, 52]]


def test_committed_submission_has_expected_shape():
    submission = pd.read_csv(Path(__file__).resolve().parents[1] / "submissions" / "submission.csv")

    assert submission.shape == (418, 2)
    assert (submission["Survived"] == 0).sum() == 252
    assert (submission["Survived"] == 1).sum() == 166
