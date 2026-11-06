"""
Streamlit app
"""

import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

MODEL_DIR = "model"
SAMPLE_TEST_FILE = "test_data.csv"

st.set_page_config(page_title="Classifier evaluation", page_icon="🧪", layout="wide")


@st.cache_resource
def load_artifacts():
    with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
        meta = json.load(f)
    encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))
    models = {name: joblib.load(os.path.join(MODEL_DIR, fname))
              for name, fname in meta["models"].items()}
    return meta, encoder, models


def evaluate(model, X, y_true, n_classes):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)

    # AUC can't be computed if a class is missing from the uploaded file (e.g. a small sample), show n/a instead of crashing
    try:
        if n_classes == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr",
                                average="weighted", labels=list(range(n_classes)))
    except ValueError:
        auc = np.nan

    avg = "binary" if n_classes == 2 else "weighted"
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=avg, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    return metrics, y_pred


def plot_confusion(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    size = max(4, 0.7 * len(labels) + 2)
    fig, ax = plt.subplots(figsize=(size, size * 0.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------
try:
    meta, encoder, models = load_artifacts()
except FileNotFoundError:
    st.error("Model files not found in model/. Run train_models.py first and commit the model folder.")
    st.stop()

target = meta["target"]
features = meta["features"]
class_labels = meta["classes"]
n_classes = len(class_labels)

# ---------------------------------------------------------------------------
# Sidebar: data and model choice
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Setup")
    uploaded = st.file_uploader("Upload test data (CSV)", type="csv")
    st.caption(f"The file needs the {len(features)} feature columns and the "
               f"**{target}** column.")

    if os.path.exists(SAMPLE_TEST_FILE):
        with open(SAMPLE_TEST_FILE, "rb") as f:
            st.download_button("Download sample test_data.csv", f,
                               file_name="test_data.csv", mime="text/csv")

    model_name = st.selectbox("Model", list(models.keys()))

st.title("Student outcome early-warning")
st.write(
    "Universities lose many students before they finish their degree, and the earlier "
    "a struggling student is spotted, the more time there is to help. This app predicts "
    "whether a student will drop out, stay enrolled, or graduate, using only what is "
    "known at enrollment and after the first semester. Upload the held-out test data in "
    "the sidebar to see how five pre-trained classifiers perform."
)

with st.expander("What do the outcomes and metrics mean?"):
    st.markdown(
        "**Dropout**: the student left the course before completing it.  \n"
        "**Enrolled**: the student was still enrolled at the end of the normal course "
        "duration, so they had not finished on time but had not left either.  \n"
        "**Graduate**: the student completed the degree within the normal duration.\n\n"
        "**AUC**: how well the model ranks students, i.e. how often it gives a higher "
        "probability to a student who really belongs to a class than to one who doesn't. "
        "0.5 is random guessing, 1.0 is perfect. For three classes it is averaged "
        "one-vs-rest.  \n"
        "**MCC**: a single score from -1 to 1 that uses every cell of the confusion matrix. "
        "About half the students graduate, so a model can reach decent accuracy by "
        "favouring that class while doing poorly on the smaller Enrolled group. MCC "
        "punishes that, which makes it a fairer summary than accuracy for this data."
    )

if uploaded is None:
    st.info("Upload a test CSV in the sidebar to see results.")
    st.stop()

data = pd.read_csv(uploaded)

missing = [c for c in features + [target] if c not in data.columns]
if missing:
    st.error(f"These columns are missing from the uploaded file: {', '.join(missing)}")
    st.stop()

# drop rows whose label the model has never seen
known = data[target].astype(str).isin(class_labels)
if not known.all():
    st.warning(f"Skipped {(~known).sum()} rows with labels not seen during training.")
    data = data[known]

X = data[features].copy()
# category codes like Course were trained as text; pandas reads them back as numbers
cat_cols = meta.get("categorical_features", [])
X[cat_cols] = X[cat_cols].astype(str)
# compare labels as strings so numeric (0/1) and text labels both work
label_to_index = {str(c): i for i, c in enumerate(encoder.classes_)}
y_true = data[target].astype(str).map(label_to_index).to_numpy()

with st.expander(f"Preview of uploaded data ({len(data)} rows)"):
    st.dataframe(data.head(20), width="stretch")

# ---------------------------------------------------------------------------
# Selected model
# ---------------------------------------------------------------------------
metrics, y_pred = evaluate(models[model_name], X, y_true, n_classes)

st.subheader(model_name)
cols = st.columns(6)
for col, (k, v) in zip(cols, metrics.items()):
    col.metric(k, "n/a" if pd.isna(v) else f"{v:.3f}")

left, right = st.columns([1, 1])
with left:
    st.markdown("**Confusion matrix**")
    st.pyplot(plot_confusion(y_true, y_pred, class_labels))
with right:
    st.markdown("**Classification report**")
    report = classification_report(y_true, y_pred, labels=list(range(n_classes)),
                                   target_names=class_labels, output_dict=True,
                                   zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(4), width="stretch")

# ---------------------------------------------------------------------------
# All models side by side
# ---------------------------------------------------------------------------
st.subheader("All five models on this data")
rows = []
all_preds = {}   # kept for the dropout panel below
for name, m in models.items():
    scores, preds = evaluate(m, X, y_true, n_classes)
    rows.append({"Model": name, **scores})
    all_preds[name] = preds
comparison = pd.DataFrame(rows).set_index("Model").round(4)

st.dataframe(comparison.style.highlight_max(axis=0, color="#2f6b4f").format("{:.4f}"),
             width="stretch")

metric_to_plot = st.radio("Compare by", list(metrics.keys()), horizontal=True, index=4)
st.bar_chart(comparison[metric_to_plot])

# ---------------------------------------------------------------------------
# Dropout early-warning view: which model catches the students at risk?
# ---------------------------------------------------------------------------
DROPOUT_LABEL = "Dropout"
if DROPOUT_LABEL in label_to_index:
    st.subheader("Who gets flagged as a likely dropout?")
    dropout_idx = label_to_index[DROPOUT_LABEL]
    actual_dropouts = int((y_true == dropout_idx).sum())

    risk_rows = []
    for name, preds in all_preds.items():
        flagged = preds == dropout_idx
        caught = int((flagged & (y_true == dropout_idx)).sum())
        risk_rows.append({
            "Model": name,
            "Flagged as dropout": int(flagged.sum()),
            "Real dropouts caught": caught,
            "Real dropouts missed": actual_dropouts - caught,
            "Share of dropouts caught": f"{caught / actual_dropouts:.0%}" if actual_dropouts else "n/a",
            "False alarms": int((flagged & (y_true != dropout_idx)).sum()),
        })

    st.write(f"The uploaded file has **{actual_dropouts}** students who actually dropped out.")
    st.dataframe(pd.DataFrame(risk_rows).set_index("Model"), width="stretch")
    st.caption(
        "For an early-warning system, a missed dropout is usually the costlier mistake: "
        "that student gets no support at all. A false alarm mostly costs an advisor's "
        "time, so a model that catches more real dropouts can be worth a few extra alarms."
    )

# ---------------------------------------------------------------------------
# Encoding experiment from training (one-hot codes vs codes as plain numbers)
# ---------------------------------------------------------------------------
ONE_HOT_RESULTS = os.path.join(MODEL_DIR, "metrics_train_run.csv")
NUMERIC_CODE_RESULTS = os.path.join(MODEL_DIR, "metrics_codes_as_numbers.csv")

if os.path.exists(ONE_HOT_RESULTS) and os.path.exists(NUMERIC_CODE_RESULTS):
    st.subheader("Experiment: how the category codes were encoded")
    one_hot = pd.read_csv(ONE_HOT_RESULTS).set_index("Model")
    as_numbers = pd.read_csv(NUMERIC_CODE_RESULTS).set_index("Model")

    experiment = pd.DataFrame({
        "Accuracy (one-hot)": one_hot["Accuracy"],
        "Accuracy (as numbers)": as_numbers["Accuracy"],
        "MCC (one-hot)": one_hot["MCC"],
        "MCC (as numbers)": as_numbers["MCC"],
    })
    experiment["MCC change"] = (experiment["MCC (one-hot)"] - experiment["MCC (as numbers)"]).round(4)
    st.dataframe(experiment, width="stretch")
    st.caption(
        "Columns such as Course and parents' occupation are category codes, not real "
        "quantities. One-hot encoding them helps the distance- and weight-based models, "
        "but Gaussian Naive Bayes struggles with the many sparse 0/1 columns it creates, "
        "especially for rare categories. The deployed models use one-hot encoding."
    )