# Titanic — Machine Learning from Disaster

[![CI](https://github.com/caueeeb/kaggle-titanic/actions/workflows/ci.yml/badge.svg)](https://github.com/caueeeb/kaggle-titanic/actions/workflows/ci.yml)

Binary classification for the Kaggle [Titanic](https://www.kaggle.com/competitions/titanic) competition: exploratory analysis, feature engineering, a scikit-learn pipeline, model comparison with stratified cross-validation, and a competition submission. The trained pipeline is published on Hugging Face: [caueeb/titanic-survival-logistic-regression](https://huggingface.co/caueeb/titanic-survival-logistic-regression).

> **Resumo (PT-BR):** projeto de classificação do Titanic com pipeline scikit-learn (imputação, padronização, one-hot), comparação de quatro modelos por validação cruzada e submissão ao Kaggle (pontuação pública **0,77033**). O código foi organizado em módulos testados, e `python -m src.train` reproduz a submissão e os números abaixo. É um projeto de aprendizado: um baseline bem validado, não uma solução otimizada para o ranking.

## Results

| Evaluation | Result |
|---|---:|
| Kaggle public leaderboard score | **0.77033** |
| Stratified 80/20 holdout accuracy (179 passengers) | 83.24% (precision 80.0%, recall 75.4%, F1 77.6%) |
| 5-fold CV accuracy on the 712 training passengers | 82.59% ± 2.98 |

The Kaggle score and the local validation numbers are measured on different data (the hidden test labels versus a holdout of the training file), so they are different estimates, not a contradiction. With only 891 labeled passengers, differences of about one percentage point between models are within noise (see [Further experiments](#further-experiments)).

### Model comparison

Stratified 5-fold CV on the training portion of the holdout split:

| Model | Mean accuracy | Accuracy std. dev. | Mean F1 | Mean ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | **0.8259** | 0.0298 | **0.7728** | 0.8716 |
| Extra Trees | **0.8259** | 0.0238 | 0.7566 | 0.8804 |
| SVM | **0.8259** | **0.0172** | 0.7625 | 0.8642 |
| Random Forest | 0.8231 | 0.0224 | 0.7583 | **0.8869** |

Logistic Regression was selected because it tied for the highest accuracy and had the highest F1-score; it is also the simplest and easiest to explain. The final pipeline was refit on all 891 labeled passengers to predict the 418 test passengers (252 predicted non-survivors, 166 predicted survivors).

## Data and features

The competition data is not committed. Download `train.csv`, `test.csv` and `gender_submission.csv` from Kaggle (after accepting the rules) into `data/raw/`.

Training target: `Survived` (38.4% survived). Missing values: `Age` (177), `Cabin` (687), `Embarked` (2). Descriptive survival rates in the training data: female 74.2%, male 18.9%; first class 63.0%, second 47.3%, third 24.2% (associations, not causes).

Engineered features (`src/features.py`): `FamilySize` (`SibSp + Parch + 1`), `IsAlone`, `HasCabin`, and `Title` extracted from the name (rare titles grouped). The model receives 11 inputs: `Pclass`, `Age`, `SibSp`, `Parch`, `Fare`, `FamilySize`, `IsAlone`, `HasCabin` (numeric) and `Sex`, `Embarked`, `Title` (categorical).

Preprocessing lives inside the pipeline (median imputation and scaling for numeric columns; most-frequent imputation and one-hot encoding for categorical ones), so it is fitted only on training folds.

## Further experiments

`python -m src.experiments` runs repeated stratified 5×5 cross-validation on all 891 labeled rows and compares extra features (`TicketGroupSize`, `FarePerPerson`, `AgeMissing`, `Deck`) and other models. These are local CV estimates, not leaderboard scores.

| Experiment | Mean accuracy | Std (25 folds) |
|---|---:|---:|
| Baseline: 11 features + Logistic Regression | 0.8236 | 0.0222 |
| + extra features + Logistic Regression | 0.8227 | 0.0260 |
| 11 features + Random Forest | 0.8281 | 0.0249 |
| 11 features + HistGradientBoosting | 0.8269 | 0.0200 |
| + extra features + HistGradientBoosting | 0.8334 | 0.0174 |

The best variant is about one point above the baseline, which is smaller than the fold-to-fold spread, so the project keeps the simpler baseline. These results were not submitted to Kaggle.

## Reproduce

Requires Python 3.11+ (developed on 3.13).

```bash
git clone https://github.com/caueeeb/kaggle-titanic.git
cd kaggle-titanic
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt

pytest                             # unit tests (synthetic data); data-dependent tests skip if data/raw/ is empty
python -m src.train                # validation, comparison, final model and submissions/submission.csv
python -m src.experiments          # optional extra experiments
```

With the original Kaggle files, `python -m src.train` regenerates the committed `submissions/submission.csv` byte for byte and the metrics above. The exploratory analysis is in `notebooks/01_titanic_analysis.ipynb`.

## Project structure

```text
kaggle-titanic/
├── .github/workflows/ci.yml
├── data/raw/                      # Kaggle CSVs (not versioned)
├── models/logistic_regression_titanic.joblib
├── notebooks/01_titanic_analysis.ipynb
├── reports/                       # CV comparison, holdout metrics, extra experiments
├── src/
│   ├── features.py                # feature engineering
│   ├── modeling.py                # pipeline, holdout, model comparison
│   ├── train.py                   # python -m src.train
│   └── experiments.py             # python -m src.experiments
├── submissions/submission.csv
├── tests/
├── pyproject.toml
├── requirements.txt               # pinned runtime dependencies
└── requirements-dev.txt
```

## Limitations

- The dataset is small (891 labeled rows), so estimates are noisy.
- The model is a baseline, not an optimized competition entry; no hyperparameter search was run on the final model.
- A passenger whose title is "the Countess" is not grouped into the rare-title category by the original rule; the rule is kept to preserve reproducibility.
- Patterns learned here reflect the specific historical event and should not be used for decisions about people.

## Technologies

Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Jupyter, pytest, Ruff and GitHub Actions.

## Author

**Cauê Buzolin** — Analysis and Systems Development student at Fatec Taquaritinga.
[GitHub](https://github.com/caueeeb) · [LinkedIn](https://www.linkedin.com/in/cau%C3%AA-buzolin-76a4283b3/) · [Hugging Face](https://huggingface.co/caueeb)

Released under the [MIT License](LICENSE). Kaggle competition data is subject to the competition rules.
