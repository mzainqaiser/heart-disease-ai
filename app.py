"""HeartGuard AI - Flask application serving the trained heart-disease pipeline.

The saved pipeline (model/heart_disease_model.pkl) contains imputation, scaling,
one-hot encoding AND the classifier, so this module never re-implements any
preprocessing: it only validates user input, arranges it in the training column
order, and calls ``pipeline.predict`` / ``pipeline.predict_proba``.

Educational demonstration only - not a medical device.
"""
from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "heart_disease_model.pkl"
METADATA_PATH = BASE_DIR / "model" / "model_metadata.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("heartguard")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024  # a prediction form is tiny; reject anything larger


# --------------------------------------------------------------------------- #
# Input schema: single source of truth for the form, validation and display.   #
# Category codes are the exact numeric codes used in heart.csv / training.      #
# --------------------------------------------------------------------------- #
def _opts(*pairs):
    return [{"value": v, "label": l} for v, l in pairs]


FIELD_GROUPS = [
    {
        "id": "patient",
        "title": "Patient Information",
        "subtitle": "Basic demographic details",
        "icon": "user",
        "fields": [
            {"name": "age", "label": "Age", "kind": "number", "unit": "years", "min": 18, "max": 100,
             "step": 1, "integer": True, "placeholder": "e.g. 54", "help": "Age in whole years (18-100)."},
            {"name": "sex", "label": "Sex", "kind": "radio",
             "options": _opts((1, "Male"), (0, "Female")), "help": "Sex as recorded in the dataset."},
        ],
    },
    {
        "id": "measurements",
        "title": "Clinical Measurements",
        "subtitle": "Values from examination and exercise testing",
        "icon": "pulse",
        "fields": [
            {"name": "trestbps", "label": "Resting Blood Pressure", "kind": "number", "unit": "mm Hg",
             "min": 80, "max": 250, "step": 1, "integer": True, "placeholder": "e.g. 130",
             "help": "Resting systolic blood pressure (80-250)."},
            {"name": "chol", "label": "Serum Cholesterol", "kind": "number", "unit": "mg/dl",
             "min": 100, "max": 600, "step": 1, "integer": True, "placeholder": "e.g. 240",
             "help": "Total serum cholesterol (100-600)."},
            {"name": "thalach", "label": "Maximum Heart Rate", "kind": "number", "unit": "bpm",
             "min": 60, "max": 220, "step": 1, "integer": True, "placeholder": "e.g. 150",
             "help": "Highest heart rate reached during the exercise test (60-220)."},
            {"name": "oldpeak", "label": "ST Depression (Oldpeak)", "kind": "number", "unit": "mm",
             "min": 0, "max": 10, "step": 0.1, "integer": False, "placeholder": "e.g. 1.2",
             "help": "ST depression induced by exercise relative to rest (0-10)."},
        ],
    },
    {
        "id": "indicators",
        "title": "Clinical Indicators",
        "subtitle": "Symptoms, ECG and imaging findings",
        "icon": "stethoscope",
        "fields": [
            {"name": "cp", "label": "Chest Pain Type", "kind": "select",
             "options": _opts((0, "Asymptomatic"), (1, "Atypical angina"), (2, "Non-anginal pain"),
                              (3, "Typical angina")),
             "help": "Type of chest pain experienced."},
            {"name": "fbs", "label": "Fasting Blood Sugar > 120 mg/dl", "kind": "radio",
             "options": _opts((0, "No"), (1, "Yes")), "help": "Whether fasting blood sugar exceeds 120 mg/dl."},
            {"name": "restecg", "label": "Resting ECG", "kind": "select",
             "options": _opts((1, "Normal"), (0, "LV hypertrophy"), (2, "ST-T wave abnormality")),
             "help": "Resting electrocardiogram result (LV = left ventricular)."},
            {"name": "exang", "label": "Exercise-Induced Angina", "kind": "radio",
             "options": _opts((0, "No"), (1, "Yes")), "help": "Chest pain triggered by exercise."},
            {"name": "slope", "label": "ST Slope", "kind": "select",
             "options": _opts((2, "Upsloping"), (1, "Flat"), (0, "Downsloping")),
             "help": "Slope of the peak-exercise ST segment."},
            {"name": "ca", "label": "Major Vessels (CA)", "kind": "select",
             "options": _opts((0, "0 vessels"), (1, "1 vessel"), (2, "2 vessels"), (3, "3 vessels")),
             "help": "Number of major vessels coloured by fluoroscopy."},
            {"name": "thal", "label": "Thalassemia (Thal)", "kind": "select",
             "options": _opts((2, "Normal"), (1, "Fixed defect"), (3, "Reversible defect")),
             "help": "Thallium stress-test result."},
        ],
    },
]
FIELDS = {f["name"]: f for g in FIELD_GROUPS for f in g["fields"]}

# Display-only example inputs wired to the "Try an example" buttons in the UI.
EXAMPLES = {
    "example_a": {"age": 45, "sex": 0, "cp": 2, "trestbps": 120, "chol": 200, "fbs": 0, "restecg": 1,
                  "thalach": 172, "exang": 0, "oldpeak": 0.0, "slope": 2, "ca": 0, "thal": 2},
    "example_b": {"age": 62, "sex": 1, "cp": 0, "trestbps": 140, "chol": 268, "fbs": 0, "restecg": 0,
                  "thalach": 120, "exang": 1, "oldpeak": 3.0, "slope": 1, "ca": 2, "thal": 3},
}


# --------------------------------------------------------------------------- #
# Model loading (once, at start-up)                                            #
# --------------------------------------------------------------------------- #
MODEL = None
METADATA: dict = {}
MODEL_ERROR: str | None = None
FEATURES: list[str] = list(FIELDS)


def load_model() -> None:
    """Load pipeline + metadata. Failures are recorded, never raised, so the site can show a friendly message."""
    global MODEL, METADATA, MODEL_ERROR, FEATURES
    try:
        METADATA = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        FEATURES = list(METADATA["feature_names"])
        if set(FEATURES) != set(FIELDS):
            raise ValueError("Form schema and model feature list do not match.")
        MODEL = joblib.load(MODEL_PATH)
        if list(MODEL.classes_) != [0, 1]:
            raise ValueError(f"Unexpected model classes: {list(MODEL.classes_)}")
        MODEL_ERROR = None
        logger.info("Loaded model '%s' (v%s)", METADATA.get("model_name"), METADATA.get("model_version"))
    except Exception:  # noqa: BLE001 - deliberately broad: never crash the web process on start-up
        MODEL, MODEL_ERROR = None, "The prediction model could not be loaded."
        logger.exception("Model loading failed")


load_model()


def model_info() -> dict:
    """Small, template-friendly summary of the deployed model (all values come from model_metadata.json)."""
    if not METADATA:
        return {}
    cv = METADATA.get("metrics", {}).get("cross_validation_repeated_5fold_x3_train_set", {})
    ho = METADATA.get("metrics", {}).get("holdout_test_set", {})
    data = METADATA.get("data", {})
    return {
        "name": METADATA.get("model_name"),
        "version": METADATA.get("model_version"),
        "trained": (METADATA.get("training_date") or "")[:10],
        "n_features": len(METADATA.get("feature_names", [])),
        "rows": data.get("rows_used"),
        "cv_auc": cv.get("ROC-AUC"), "cv_recall": cv.get("Recall"), "cv_f1": cv.get("F1"),
        "cv_accuracy": cv.get("Accuracy"),
        "test_accuracy": ho.get("Accuracy"), "test_recall": ho.get("Recall"),
        "test_precision": ho.get("Precision"), "test_f1": ho.get("F1"), "test_auc": ho.get("ROC-AUC"),
    }


# --------------------------------------------------------------------------- #
# Validation + prediction                                                      #
# --------------------------------------------------------------------------- #
def validate(raw) -> tuple[dict, dict]:
    """Validate raw form/JSON input. Returns (clean_values, errors) - errors maps field -> message."""
    clean, errors = {}, {}
    for name, spec in FIELDS.items():
        value = raw.get(name)
        text = "" if value is None else str(value).strip()
        if text == "":
            errors[name] = "This field is required."
            continue
        try:
            number = float(text)
        except ValueError:
            errors[name] = "Please enter a valid number."
            continue
        if not math.isfinite(number):
            errors[name] = "Please enter a valid number."
            continue

        if spec["kind"] == "number":
            if spec["integer"] and not number.is_integer():
                errors[name] = "Please enter a whole number."
            elif not spec["min"] <= number <= spec["max"]:
                errors[name] = f"Value must be between {spec['min']} and {spec['max']} {spec['unit']}."
            else:
                clean[name] = number
        else:  # radio / select: must be one of the allowed category codes
            allowed = {opt["value"] for opt in spec["options"]}
            if not number.is_integer() or int(number) not in allowed:
                errors[name] = "Please choose one of the listed options."
            else:
                clean[name] = float(int(number))
    return clean, errors


def predict(clean: dict) -> dict:
    """Run the saved pipeline on one validated record."""
    frame = pd.DataFrame([clean])[FEATURES].astype(float)   # training column order, training dtype
    probability = float(MODEL.predict_proba(frame)[0, 1])
    label_id = int(MODEL.predict(frame)[0])
    labels = METADATA.get("target_labels", {})
    return {
        "prediction": label_id,
        "label": labels.get(str(label_id), "Higher likelihood" if label_id else "Lower likelihood"),
        "probability": probability,
        "percent": round(probability * 100, 1),
    }


def describe_inputs(clean: dict) -> list[dict]:
    """Human-readable echo of the submitted values for the result page."""
    rows = []
    for group in FIELD_GROUPS:
        items = []
        for spec in group["fields"]:
            value = clean[spec["name"]]
            if spec["kind"] == "number":
                shown = f"{value:g}" + (f" {spec['unit']}" if spec.get("unit") else "")
            else:
                shown = next(o["label"] for o in spec["options"] if o["value"] == int(value))
            items.append({"label": spec["label"], "value": shown})
        rows.append({"title": group["title"], "items": items})
    return rows


def _render_form(values=None, errors=None, banner=None, status=200):
    return render_template(
        "index.html", groups=FIELD_GROUPS, values=values or {}, errors=errors or {}, banner=banner,
        info=model_info(), model_ready=MODEL is not None, examples=EXAMPLES,
    ), status


# --------------------------------------------------------------------------- #
# Routes                                                                       #
# --------------------------------------------------------------------------- #
@app.get("/")
def index():
    return _render_form()


@app.post("/predict")
def predict_form():
    if MODEL is None:
        return _render_form(values=request.form, banner=MODEL_ERROR + " Please try again later.", status=503)

    clean, errors = validate(request.form)
    if errors:
        return _render_form(values=request.form, errors=errors,
                            banner="Please correct the highlighted fields and try again.", status=400)
    try:
        result = predict(clean)
    except Exception:  # noqa: BLE001
        logger.exception("Prediction failed")
        return _render_form(values=request.form,
                            banner="Something went wrong while generating the prediction. Please try again.",
                            status=500)
    return render_template("result.html", result=result, summary=describe_inputs(clean), info=model_info())


@app.post("/api/predict")
def predict_api():
    """JSON endpoint: POST the 13 feature values, receive the prediction."""
    if MODEL is None:
        return jsonify(error=MODEL_ERROR), 503
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Request body must be a JSON object."), 400
    clean, errors = validate(payload)
    if errors:
        return jsonify(error="Invalid input.", fields=errors), 400
    try:
        result = predict(clean)
    except Exception:  # noqa: BLE001
        logger.exception("API prediction failed")
        return jsonify(error="Prediction failed."), 500
    return jsonify(prediction=result["prediction"], label=result["label"],
                   probability_heart_disease=round(result["probability"], 4),
                   model=METADATA.get("model_name"), version=METADATA.get("model_version"),
                   disclaimer="Educational demonstration only; not a medical diagnosis.")


@app.get("/health")
def health():
    ok = MODEL is not None
    return jsonify(status="ok" if ok else "unavailable", model_loaded=ok,
                   model=METADATA.get("model_name") if ok else None), (200 if ok else 503)


@app.after_request
def security_headers(response):
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    return response


# --------------------------------------------------------------------------- #
# Error handlers: friendly pages, never a stack trace                          #
# --------------------------------------------------------------------------- #
def _error_response(code: int, title: str, message: str):
    if request.path.startswith("/api/"):
        return jsonify(error=message), code
    return render_template("error.html", code=code, title=title, message=message), code


@app.errorhandler(400)
def bad_request(_e):
    return _error_response(400, "Bad request", "We could not understand that request.")


@app.errorhandler(404)
def not_found(_e):
    return _error_response(404, "Page not found", "The page you are looking for does not exist.")


@app.errorhandler(405)
def method_not_allowed(_e):
    return _error_response(405, "Method not allowed", "That action is not supported on this page.")


@app.errorhandler(413)
def too_large(_e):
    return _error_response(413, "Request too large", "The submitted data was too large.")


@app.errorhandler(500)
def server_error(_e):
    return _error_response(500, "Something went wrong", "An unexpected error occurred. Please try again later.")


if __name__ == "__main__":
    # Local development only. In production the app is served by gunicorn (see Procfile / render.yaml).
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")
