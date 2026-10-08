# HeartGuard AI - Heart Disease Prediction

An end-to-end machine-learning project: a leakage-free scikit-learn workflow (EDA, 5 tuned models, cross-validation, multi-criteria model selection) packaged as a **Flask web app** with a responsive dashboard-style UI, a JSON API, automated tests and one-click **Render** deployment config.

> **Disclaimer.** This project is for educational and demonstration purposes only. It does not provide medical diagnosis or replace advice from a qualified healthcare professional.

## Results at a glance

Selected model: **Logistic Regression** (`C=0.316`, `l1_ratio=0`). All numbers below come from `Heart_Disease_Classification_Project.ipynb` (also stored in `model/model_metadata.json`). Positive class = heart disease.

**Repeated 5-fold CV on the training set (15 folds per model) - used to choose the model**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| KNN | 0.841 | 0.884 | 0.754 | 0.810 | 0.912 |
| **Logistic Regression** | 0.862 | 0.880 | 0.812 | 0.841 | 0.918 |
| Decision Tree | 0.829 | 0.860 | 0.749 | 0.798 | 0.857 |
| Random Forest | 0.848 | 0.864 | 0.797 | 0.824 | 0.920 |
| SVM | 0.848 | 0.899 | 0.754 | 0.817 | 0.916 |

**Hold-out test set (61 patients) - independent confirmation**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| KNN | 0.820 | 0.815 | 0.786 | 0.800 | 0.897 |
| **Logistic Regression** | 0.836 | 0.875 | 0.750 | 0.808 | 0.906 |
| Decision Tree | 0.770 | 0.750 | 0.750 | 0.750 | 0.851 |
| Random Forest | 0.787 | 0.800 | 0.714 | 0.755 | 0.905 |
| SVM | 0.836 | 0.875 | 0.750 | 0.808 | 0.906 |

*(Tables are rounded from `model_metadata.json`; the notebook's own display may differ by 0.001 on a few cells.)*

The model was chosen by the lowest average rank over six criteria (ROC-AUC, F1, recall, balanced accuracy, fold-to-fold stability, train-vs-CV overfitting gap), not by accuracy alone. With only 302 unique patients the test set is small (one patient = 1.6 percentage points), so differences of a point or two between models are within noise.

## Data findings worth knowing

| Finding | What was done |
|---|---|
| **723 of 1,025 rows are exact duplicates** (only 302 unique patients) | Removed before splitting; otherwise copies land in both train and test and inflate every metric. |
| **`target = 1` means *no* disease in this file** (opposite of the original brief). Verified: exercise angina, ST depression and blocked vessels are all *higher* when `target = 0`, and the 164-row `target = 1` class matches the 164 healthy patients of the UCI Cleveland data. | Derived `HeartDisease = 1 - target` so recall/precision describe *detecting disease*. The app labels follow this. |
| Category codes (`cp`, `restecg`, `slope`, `thal`) do not match the commonly circulated descriptions | Decoded by matching category frequencies to Cleveland and checking clinical direction of effect (see notebook, Task 1). |
| `ca = 4` (4 rows) and `thal = 0` (2 rows) are placeholder codes for missing values | Converted to `NaN`, imputed inside the pipeline. |
| The brief described a different 918-row file (`ChestPainType`, `ST_Slope`...) | Notebook description rewritten from the actual data. |

## Project structure

```text
heart-disease-project/
├── Heart_Disease_Classification_Project.ipynb   # Tasks 1-7, executed, with outputs
├── heart.csv                                    # dataset
├── model/
│   ├── heart_disease_model.pkl                  # full pipeline: impute + scale + one-hot + classifier
│   └── model_metadata.json                      # model name, features, target meaning, metrics, versions
├── app.py                                       # Flask app (validation, prediction, JSON API, error handling)
├── templates/                                   # base, index, result, error (Jinja2)
├── static/css/style.css  static/js/script.js    # UI (no external CDN/fonts)
├── tests/test_app.py                            # 46 automated tests
├── reports/figures/                             # plots exported by the notebook
├── requirements.txt  requirements-dev.txt
├── Procfile  render.yaml  .python-version       # deployment configuration
└── README.md
```

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py                                          # http://localhost:5000
# production-style:
gunicorn app:app --bind 0.0.0.0:5000 --workers 1 --threads 2
```

Re-run the notebook (regenerates `model/` and `reports/figures/`) and the tests:

```bash
pip install -r requirements-dev.txt
jupyter nbconvert --to notebook --execute --inplace Heart_Disease_Classification_Project.ipynb
python -m pytest -q
```

> The pickle must be loaded with the same scikit-learn version that saved it (pinned to `1.9.1`). If you change versions, re-run the notebook to regenerate the model.

## How prediction works

1. The browser form (human-readable choices, client-side validation) posts 13 values.
2. `app.py` validates every field again on the server (required, numeric, whole number where needed, range, allowed category codes) and shows friendly errors.
3. Values are arranged in the training column order and passed to the **saved pipeline**; Flask never re-implements preprocessing.
4. The page shows the class label and the estimated probability. Stack traces are never shown to users.

### JSON API

```bash
curl -X POST http://localhost:5000/api/predict -H "Content-Type: application/json" -d '{
  "age": 62, "sex": 1, "cp": 0, "trestbps": 140, "chol": 268, "fbs": 0, "restecg": 0,
  "thalach": 120, "exang": 1, "oldpeak": 3.0, "slope": 1, "ca": 2, "thal": 3 }'
```

Category codes: `sex` 1=male/0=female; `cp` 0=asymptomatic, 1=atypical angina, 2=non-anginal, 3=typical angina; `fbs`/`exang` 1=yes; `restecg` 0=LV hypertrophy, 1=normal, 2=ST-T abnormality; `slope` 0=down, 1=flat, 2=up; `ca` 0-3; `thal` 1=fixed defect, 2=normal, 3=reversible defect. `GET /health` returns service/model status.

## Deploy to Render

**Option A - Blueprint (fastest).** Push this folder to a GitHub repo, then in Render: *New > Blueprint* > select the repo. `render.yaml` supplies everything.

**Option B - manual Web Service.**

| Setting | Value |
|---|---|
| Environment / Language | Python 3 |
| Build Command | `pip install --upgrade pip && pip install -r requirements.txt` |
| Start Command | `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 60` |
| Python version | from `.python-version` (`3.13`); or set env var `PYTHON_VERSION` to a full version such as `3.13.5` |
| Health Check Path | `/health` |
| Instance type | Free |

Render supplies `$PORT`; the app binds to `0.0.0.0`. No other environment variables are required. On the free tier the service sleeps after inactivity, so the first request can take about a minute. The `Procfile` holds the same command for Heroku-style platforms.

## Limitations

- Small dataset (302 unique patients, one hospital population): estimates are noisy and may not generalise.
- Hyper-parameters were tuned on the same training data later used for model comparison (same procedure for every model, so the ranking is fair, but absolute CV scores are slightly optimistic; the hold-out set is the cleaner estimate).
- The deployed model is refit on all 302 unique rows after evaluation; reported metrics are from the train-only evaluation of the identical configuration.
- Not a medical device.
