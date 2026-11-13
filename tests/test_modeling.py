import pandas as pd

from src.features import prepare_test_features, split_features_target
from src.modeling import build_pipeline, candidate_models


def test_pipeline_fits_and_predicts_with_missing_values(passengers):
    X, y = split_features_target(passengers)

    model = build_pipeline().fit(X, y)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    assert len(predictions) == len(X)
    assert set(predictions) <= {0, 1}
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_pipeline_handles_unseen_category(passengers):
    X, y = split_features_target(passengers)
    model = build_pipeline().fit(X, y)

    new_passenger = passengers.drop(columns=["Survived"]).head(1).copy()
    new_passenger["Embarked"] = "Z"                       # porto nunca visto
    new_passenger["Name"] = "Doe, Dr. Jane"               # título nunca visto no treino

    prediction = model.predict(prepare_test_features(new_passenger))

    assert prediction[0] in (0, 1)


def test_all_candidates_are_defined():
    assert set(candidate_models()) == {
        "Logistic Regression", "Random Forest", "Extra Trees", "SVM",
    }


def test_submission_format(passengers):
    X, y = split_features_target(passengers)
    model = build_pipeline().fit(X, y)

    test_like = passengers.drop(columns=["Survived"])
    predictions = model.predict(prepare_test_features(test_like)).astype(int)

    submission = pd.DataFrame({
        "PassengerId": test_like["PassengerId"],
        "Survived": predictions,
    })

    assert list(submission.columns) == ["PassengerId", "Survived"]
    assert submission["Survived"].isin([0, 1]).all()
