"""
This is for training the five classifiers required for Assignment and saving them to model.

"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------------------
# CONFIG -  Settings for the student dropout dataset.
# ---------------------------------------------------------------------------
DATA_PATH = "data/data.csv"   # UCI "Predict Students' Dropout and Academic Success"
CSV_SEP = ";"                 # this file is semicolon-separated
TARGET_COL = "Target"         # Dropout / Enrolled / Graduate

# Early-warning setup: drop everything from the 2nd semester so the models
# only using what is known at enrollment plus the 1st semester.
SECOND_SEM_COLS = [
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
]

# These are stored as numbers but are really category codes (course 171 is not
# "bigger" than course 33), so they get one-hot encoded instead of scaled.
# Setting ONE_HOT_CODED_COLS = False to treat them as plain numbers for comparison.
ONE_HOT_CODED_COLS = True
CODED_CATEGORY_COLS = [
    "Marital status", "Application mode", "Course", "Previous qualification",
    "Nacionality", "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation",
]

TEST_SIZE = 0.2
RANDOM_STATE = 42

MODEL_DIR = "model"


def build_preprocessor(X):
    """Preparatory features: numeric columns are imputed and scaled; the coded category columns (Course, occupations, etc.) are imputed and one-hot encoded"""
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        # dense output because GaussianNB does not accept sparse matrices
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, num_cols),
        ("cat", categorical, cat_cols),
    ])


def get_models():
    # Five classifiers compared on the same split; depth and k limited to reduce overfitting.
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(kernel="rbf", probability=True,
                                      random_state=RANDOM_STATE),
    }


def score_model(y_true, y_pred, y_prob, n_classes):
    """Six metrics required by the assignment. Works for binary and multi-class."""
    if n_classes == 2:
        auc = roc_auc_score(y_true, y_prob[:, 1])
        avg = "binary"
    else:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
        avg = "weighted"
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=avg, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def file_name(model_name):
    return model_name.lower().replace(" ", "_") + ".joblib"


def data_load():
    """Read the dropout CSV and prepare it for the first-semester early-warning setup."""
    df = pd.read_csv(DATA_PATH, sep=CSV_SEP)
    df.columns = df.columns.str.strip()   # original CSV has a hidden tab after "Daytime/evening attendance"

    missing = [c for c in SECOND_SEM_COLS if c not in df.columns]
    if missing:
        print("Warning - these SECOND_SEM_COLS were not found:", missing)
    df = df.drop(columns=SECOND_SEM_COLS, errors="ignore")
    df = df.dropna(subset=[TARGET_COL])

    if ONE_HOT_CODED_COLS:
        df[CODED_CATEGORY_COLS] = df[CODED_CATEGORY_COLS].astype(str)
    return df

def tune_hyperparameters(X_train, y_train):
    print("Tuning Decision Tree max_depth (5-fold CV, weighted F1):")
    for depth in [4, 6, 8, 10, 12, None]:
        pipe = Pipeline([
            ("prep", build_preprocessor(X_train)),
            ("clf", DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)),
        ])
        scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="f1_weighted")
        print(f"  max_depth={depth}: {scores.mean():.4f}")

    print("\nTuning KNN n_neighbors (5-fold CV, weighted F1):")
    for k in [3, 5, 7, 9, 11, 15, 21]:
        pipe = Pipeline([
            ("prep", build_preprocessor(X_train)),
            ("clf", KNeighborsClassifier(n_neighbors=k)),
        ])
        scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="f1_weighted")
        print(f"  n_neighbors={k}: {scores.mean():.4f}")
    print()

def main():
    df = data_load()
    print(f"Loaded {df.shape[0]} rows, {df.shape[1] - 1} features")
    print("Class distribution:\n", df[TARGET_COL].value_counts(), "\n")

    X = df.drop(columns=[TARGET_COL])
    y_raw = df[TARGET_COL]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    n_classes = len(encoder.classes_)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    tune_hyperparameters(X_train, y_train)

    os.makedirs(MODEL_DIR, exist_ok=True)

    for name, clf in get_models().items():
        pipe = Pipeline([("prep", build_preprocessor(X_train)), ("clf", clf)])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)
        scores = score_model(y_test, y_pred, y_prob, n_classes)
        results.append({"Model": name, **scores})

        joblib.dump(pipe, os.path.join(MODEL_DIR, file_name(name)))
        print(f"{name:<25} acc={scores['Accuracy']:.4f}  f1={scores['F1']:.4f}")

    joblib.dump(encoder, os.path.join(MODEL_DIR, "label_encoder.joblib"))

    metadata = {
        "target": TARGET_COL,
        "features": X.columns.tolist(),
       # lets the app convert coded columns back to text after reading the uploaded CSV
        "categorical_features": X.select_dtypes(exclude=np.number).columns.tolist(),
        "classes": [str(c) for c in encoder.classes_],
        "models": {name: file_name(name) for name in get_models()},
    }
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    table = pd.DataFrame(results).round(4)
    table.to_csv(os.path.join(MODEL_DIR, "metrics_train_run.csv"), index=False)
    print("\n", table.to_string(index=False))

    # Save the untouched test split with original labels, for uploading in the app
    test_df = X_test.copy()
    test_df[TARGET_COL] = encoder.inverse_transform(y_test)
    test_df.to_csv("test_data.csv", index=False)
    print(f"\nSaved test_data.csv with {len(test_df)} rows")


if __name__ == "__main__":
    main()