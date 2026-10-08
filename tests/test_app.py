"""Automated tests for the HeartGuard AI Flask app and the saved model pipeline.

Run from the project root:  python -m pytest -q
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app as app_module  # noqa: E402

VALID = {"age": "54", "sex": "1", "cp": "0", "trestbps": "130", "chol": "246", "fbs": "0", "restecg": "1",
         "thalach": "150", "exang": "1", "oldpeak": "1.5", "slope": "1", "ca": "1", "thal": "3"}


@pytest.fixture()
def client():
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def test_model_files_exist_and_loaded():
    assert (ROOT / "model" / "heart_disease_model.pkl").exists()
    assert (ROOT / "model" / "model_metadata.json").exists()
    assert app_module.MODEL is not None and app_module.MODEL_ERROR is None


def test_home_page(client):
    r = client.get("/")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert "HeartGuard" in html and "educational and demonstration purposes only" in html
    for name in app_module.FIELDS:
        assert f'name="{name}"' in html


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.get_json()["model_loaded"] is True


def test_valid_prediction_page(client):
    r = client.post("/predict", data=VALID)
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert "model probability" in html and "not a diagnosis" in html.lower()
    assert ("Higher likelihood" in html) or ("Lower likelihood" in html)


def test_example_inputs_are_valid_and_sensible(client):
    low = client.post("/api/predict", json=app_module.EXAMPLES["example_a"]).get_json()
    high = client.post("/api/predict", json=app_module.EXAMPLES["example_b"]).get_json()
    assert low["prediction"] == 0 and high["prediction"] == 1
    assert low["probability_heart_disease"] < 0.5 < high["probability_heart_disease"]


def test_api_valid(client):
    r = client.post("/api/predict", json={k: float(v) for k, v in VALID.items()})
    body = r.get_json()
    assert r.status_code == 200
    assert body["prediction"] in (0, 1) and 0 <= body["probability_heart_disease"] <= 1
    assert "disclaimer" in body


def test_page_and_api_agree(client):
    api = client.post("/api/predict", json=VALID).get_json()
    html = client.post("/predict", data=VALID).get_data(as_text=True)
    assert f"{round(api['probability_heart_disease'] * 100, 1)}%" in html


@pytest.mark.parametrize("field", list(VALID))
def test_missing_field_rejected(client, field):
    data = {k: v for k, v in VALID.items() if k != field}
    r = client.post("/predict", data=data)
    assert r.status_code == 400
    assert "required" in r.get_data(as_text=True)
    api = client.post("/api/predict", json=data)
    assert api.status_code == 400 and field in api.get_json()["fields"]


@pytest.mark.parametrize("field,bad", [
    ("age", "abc"), ("age", "5"), ("age", "250"), ("age", "54.5"), ("age", "nan"), ("age", "inf"),
    ("trestbps", "-10"), ("chol", "5000"), ("thalach", "0"), ("oldpeak", "-1"), ("oldpeak", "99"),
    ("sex", "2"), ("cp", "9"), ("cp", "1.5"), ("thal", "0"), ("ca", "4"), ("slope", "-1"), ("restecg", "x"),
])
def test_invalid_values_rejected(client, field, bad):
    data = dict(VALID, **{field: bad})
    r = client.post("/predict", data=data)
    assert r.status_code == 400
    html = r.get_data(as_text=True)
    assert "Traceback" not in html and "correct the highlighted fields" in html
    assert client.post("/api/predict", json=data).status_code == 400


def test_empty_post(client):
    assert client.post("/predict", data={}).status_code == 400


def test_api_bad_body(client):
    assert client.post("/api/predict", data="not json", content_type="text/plain").status_code == 400
    assert client.post("/api/predict", json=[1, 2, 3]).status_code == 400


def test_404_and_405_are_friendly(client):
    r = client.get("/nope")
    assert r.status_code == 404 and "Traceback" not in r.get_data(as_text=True)
    assert client.get("/predict").status_code == 405
    assert client.get("/api/predict").status_code == 405


def test_oversized_request(client):
    assert client.post("/predict", data={"age": "x" * 50000}).status_code == 413


def test_prediction_error_is_hidden(client, monkeypatch):
    def boom(_):
        raise RuntimeError("secret internal detail")
    monkeypatch.setattr(app_module, "predict", boom)
    r = client.post("/predict", data=VALID)
    assert r.status_code == 500
    html = r.get_data(as_text=True)
    assert "secret internal detail" not in html and "Something went wrong" in html
    api = client.post("/api/predict", json=VALID)
    assert api.status_code == 500 and "secret" not in api.get_data(as_text=True)


def test_model_unavailable_is_handled(client, monkeypatch):
    monkeypatch.setattr(app_module, "MODEL", None)
    monkeypatch.setattr(app_module, "MODEL_ERROR", "The prediction model could not be loaded.")
    assert client.get("/").status_code == 200
    assert client.post("/predict", data=VALID).status_code == 503
    assert client.post("/api/predict", json=VALID).status_code == 503
    assert client.get("/health").status_code == 503


def test_metadata_consistent():
    meta = json.loads((ROOT / "model" / "model_metadata.json").read_text())
    assert meta["feature_names"] == list(app_module.FEATURES)
    assert set(meta["target_labels"]) == {"0", "1"}
    assert meta["model_name"] in [row["Model"] for row in meta["comparison_all_models_cv"]]
    assert 0.5 < meta["metrics"]["holdout_test_set"]["ROC-AUC"] <= 1


def test_pickle_reloads_in_fresh_process():
    """The saved pipeline must work in a brand-new interpreter, from raw inputs only."""
    code = (
        "import joblib, pandas as pd;"
        "m = joblib.load('model/heart_disease_model.pkl');"
        "cols = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal'];"
        "row = [54,1,0,130,246,0,1,150,1,1.5,1,1,3];"
        "X = pd.DataFrame([row], columns=cols).astype(float);"
        "print(m.predict(X)[0], round(float(m.predict_proba(X)[0,1]),4))"
    )
    out = subprocess.run([sys.executable, "-I", "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    pred, proba = out.stdout.split()
    assert pred in {"0", "1"} and 0 <= float(proba) <= 1
