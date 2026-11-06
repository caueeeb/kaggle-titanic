# Titanic - Machine Learning from Disaster

A binary classification project for the Kaggle Titanic competition. The goal is to predict whether each passenger in the competition test set survived.

## Project overview

This project covers exploratory data analysis (EDA), feature engineering, preprocessing, model comparison, validation, and generation of a Kaggle submission file.

- **Competition:** [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)
- **Problem type:** Supervised binary classification
- **Target:** `Survived` (`0` = did not survive, `1` = survived)
- **Selected model:** Logistic Regression
- **Kaggle submission score:** **0.77033** (public leaderboard)

The public leaderboard score is reported separately from the local validation metrics because they are calculated on different data.

## Dataset

The competition provides:

- `train.csv`: 891 labeled passengers, including the `Survived` target.
- `test.csv`: 418 passengers without labels; these are the records submitted to Kaggle.
- `gender_submission.csv`: a sample submission illustrating the required output format.

The competition data is not committed to this repository. Download it from the [official Kaggle competition page](https://www.kaggle.com/competitions/titanic) after accepting the competition rules, then place the CSV files in `data/raw/`.

Expected input files:

```text
data/raw/
├── gender_submission.csv
├── test.csv
└── train.csv
```

## Exploratory data analysis

The training dataset contains 891 passengers and 12 columns. The target distribution is:

| Outcome | Passengers | Share |
|---|---:|---:|
| Did not survive (`0`) | 549 | 61.62% |
| Survived (`1`) | 342 | 38.38% |

Initial analysis found missing values in `Age` (177 rows), `Cabin` (687 rows), and `Embarked` (2 rows).

Some observed survival rates:

| Group | Survival rate |
|---|---:|
| Female passengers | 74.2% |
| Male passengers | 18.9% |
| First class | 63.0% |
| Second class | 47.3% |
| Third class | 24.2% |

These are descriptive associations in the training data, not causal conclusions.

## Feature engineering

The notebook creates the following additional features:

- `FamilySize`: `SibSp + Parch + 1`.
- `IsAlone`: whether the passenger traveled alone.
- `HasCabin`: whether a cabin value was recorded.
- `Title`: title extracted from the passenger name, with selected titles consolidated (for example, `Mlle` into `Miss`, and rare titles into `Rare`).

The model uses 11 input columns after excluding identifiers and raw text fields that are not passed directly to the estimator:

**Numerical:** `Pclass`, `Age`, `SibSp`, `Parch`, `Fare`, `FamilySize`, `IsAlone`, `HasCabin`.

**Categorical:** `Sex`, `Embarked`, `Title`.

## Preprocessing and model selection

A scikit-learn `Pipeline` and `ColumnTransformer` keep preprocessing and model fitting together:

- Numerical missing values are imputed with the median.
- Numerical features are standardized with `StandardScaler`.
- Categorical missing values are imputed with the most frequent value.
- Categorical features are encoded with `OneHotEncoder(handle_unknown="ignore")`.

Four classifiers were compared using stratified 5-fold cross-validation on the training portion of the local split:

| Model | Mean accuracy | Accuracy std. dev. | Mean F1 | Mean ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | **0.8259** | 0.0298 | **0.7728** | 0.8716 |
| Extra Trees | **0.8259** | 0.0238 | 0.7566 | 0.8804 |
| SVM | **0.8259** | **0.0172** | 0.7625 | 0.8642 |
| Random Forest | 0.8231 | 0.0224 | 0.7583 | **0.8869** |

Logistic Regression was selected because it tied for the highest mean accuracy and achieved the highest mean F1-score. Random Forest had the highest ROC-AUC, while SVM had the lowest accuracy standard deviation. The choice therefore considered multiple metrics rather than a single score.

## Local validation

A stratified 80/20 train-validation split (random state 42) was used for an initial local evaluation:

| Metric | Validation result |
|---|---:|
| Accuracy | **83.24%** |
| Precision | 80.00% |
| Recall | 75.36% |
| F1-score | 77.61% |

Confusion matrix (rows = actual class, columns = predicted class):

```text
[[97, 13],
 [17, 52]]
```

After model selection, the final Logistic Regression pipeline was refitted using all 891 labeled training records and used to predict the 418 records in `test.csv`.

## Kaggle submission

- **Public leaderboard score:** **0.77033**
- **Submission rows:** 418
- **Predicted non-survivors (`0`):** 252
- **Predicted survivors (`1`):** 166

The generated file is `submissions/submission.csv`. It contains exactly two columns, in the required order: `PassengerId` and `Survived`.

The trained pipeline is saved as `models/logistic_regression_titanic.joblib`.

## Project structure

```text
kaggle-titanic/
├── data/
│   ├── raw/                       # Kaggle CSV files; not tracked by Git
│   └── processed/                 # Optional processed data
├── models/
│   └── logistic_regression_titanic.joblib
├── notebooks/
│   └── 01_titanic_analysis.ipynb
├── reports/                       # Optional charts and evaluation outputs
├── src/
│   └── __init__.py
├── submissions/
│   └── submission.csv
├── tests/                          # Project tests, if present
├── .gitignore
├── README.md
└── requirements.txt
```

## Reproducing the project

The commands below use Windows PowerShell and Python 3.13.

```powershell
# Clone the repository
git clone https://github.com/caueeeb/kaggle-titanic.git
cd kaggle-titanic

# Create and activate a virtual environment
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Download the competition files from Kaggle and place them in `data/raw/`. Then open `notebooks/01_titanic_analysis.ipynb` in VS Code or Jupyter and select the project's `.venv` kernel.

## Limitations and next steps

- The model is a first end-to-end baseline, not an optimized competition solution.
- The local validation score and public leaderboard score are estimates on different datasets and should not be directly interpreted as the same measurement.
- Further work could explore grouped family/ticket features, feature selection, hyperparameter tuning, calibration, and additional classifiers while preserving leakage-safe validation.

## Technologies

Python, Pandas, NumPy, Matplotlib, Seaborn, scikit-learn, Jupyter Notebook, Joblib, Git, and GitHub.

## Author

**Cauê Buzolin**  
Analysis and Systems Development student at Fatec Taquaritinga.  
GitHub: [@caueeeb](https://github.com/caueeeb)

---

This project was developed as part of my learning journey in Machine Learning and Artificial Intelligence.
