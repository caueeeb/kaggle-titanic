"""Fluxo completo do Titanic: validação, seleção, modelo final e submissão.

Uso (a partir da raiz do repositório)::

    python -m src.train                       # lê data/raw/, grava models/, reports/, submissions/
    python -m src.train --output-dir saida    # grava em outra pasta
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from src.features import prepare_test_features, split_features_target
from src.modeling import build_pipeline, compare_models, holdout_evaluation

PROJECT_DIR = Path(__file__).resolve().parents[1]


def run(data_dir: Path, output_dir: Path) -> dict:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")

    X, y = split_features_target(train)

    # 1) Holdout estratificado 80/20 e comparação por validação cruzada (só no treino)
    _, holdout, (X_train, _, y_train, _) = holdout_evaluation(X, y)
    comparison = compare_models(X_train, y_train)

    # 2) Modelo final: Regressão Logística com todos os 891 registros rotulados
    final_model = build_pipeline().fit(X, y)

    X_test = prepare_test_features(test)
    predictions = final_model.predict(X_test).astype(int)

    submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": predictions})

    assert len(submission) == len(test)
    assert list(submission.columns) == ["PassengerId", "Survived"]

    # 3) Artefatos
    (output_dir / "models").mkdir(parents=True, exist_ok=True)
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)
    (output_dir / "submissions").mkdir(parents=True, exist_ok=True)

    joblib.dump(final_model, output_dir / "models" / "logistic_regression_titanic.joblib")
    submission.to_csv(output_dir / "submissions" / "submission.csv", index=False)
    comparison.to_csv(output_dir / "reports" / "cv_model_comparison.csv", index=False)

    summary = {
        "holdout_validation": holdout,
        "submission_rows": int(len(submission)),
        "predicted_survivors": int(predictions.sum()),
        "predicted_non_survivors": int((predictions == 0).sum()),
        "final_training_rows": int(len(X)),
        "feature_count": int(X.shape[1]),
        "sklearn_version": sklearn.__version__,
    }

    with (output_dir / "reports" / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")

    print(comparison.round(4).to_string(index=False))
    print(
        f"\nHoldout: accuracy={holdout['accuracy']:.4f} precision={holdout['precision']:.4f} "
        f"recall={holdout['recall']:.4f} f1={holdout['f1']:.4f}"
    )
    print(
        f"Submission: {summary['submission_rows']} rows, "
        f"{summary['predicted_non_survivors']} non-survivors, "
        f"{summary['predicted_survivors']} survivors"
    )

    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Treina o modelo do Titanic e gera a submissão.")
    parser.add_argument("--data-dir", default=str(PROJECT_DIR / "data" / "raw"))
    parser.add_argument("--output-dir", default=str(PROJECT_DIR))
    args = parser.parse_args(argv)

    run(Path(args.data_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
