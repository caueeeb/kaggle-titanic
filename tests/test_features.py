from src.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    create_features,
    prepare_test_features,
    split_features_target,
)


def test_family_size_and_is_alone(passengers):
    result = create_features(passengers).set_index("PassengerId")

    assert result.loc[1, "FamilySize"] == 2      # SibSp 1 + Parch 0 + 1
    assert result.loc[8, "FamilySize"] == 5      # SibSp 3 + Parch 1 + 1
    assert result.loc[5, "IsAlone"] == 1
    assert result.loc[1, "IsAlone"] == 0


def test_has_cabin(passengers):
    result = create_features(passengers).set_index("PassengerId")

    assert result.loc[2, "HasCabin"] == 1
    assert result.loc[1, "HasCabin"] == 0


def test_title_extraction_and_grouping(passengers):
    result = create_features(passengers).set_index("PassengerId")

    assert result.loc[1, "Title"] == "Mr"
    assert result.loc[2, "Title"] == "Mrs"
    assert result.loc[3, "Title"] == "Miss"
    assert result.loc[8, "Title"] == "Master"
    # Títulos raros são agrupados
    assert result.loc[11, "Title"] == "Rare"   # Rev
    assert result.loc[12, "Title"] == "Rare"   # Dr


def test_create_features_does_not_modify_input(passengers):
    before = passengers.copy()

    create_features(passengers)

    assert passengers.equals(before)


def test_model_input_has_expected_columns(passengers):
    X, y = split_features_target(passengers)

    assert sorted(X.columns) == sorted(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    assert X.shape[1] == 11
    assert len(y) == len(passengers)


def test_prepare_test_features_works_without_target(passengers):
    test_like = passengers.drop(columns=["Survived"])

    X = prepare_test_features(test_like)

    assert X.shape[1] == 11
    assert "Survived" not in X.columns
