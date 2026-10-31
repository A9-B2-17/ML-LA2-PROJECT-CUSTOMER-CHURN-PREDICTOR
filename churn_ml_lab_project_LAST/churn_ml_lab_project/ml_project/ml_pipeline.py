"""Core ML pipeline: cleaning, leakage-free preprocessing, tuned model comparison.

Leakage rules followed:
  * duplicates are removed BEFORE the split (so identical rows can't land in both sets)
  * 80:20 stratified split is done BEFORE any statistic is learned
  * imputer, outlier caps, encoder and scaler live inside a sklearn Pipeline, so they are
    fitted on the training fold only (also inside every GridSearchCV fold)
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ARTIFACTS = Path("artifacts")
RANDOM_STATE = 42


class IQRCapper(BaseEstimator, TransformerMixin):
    """Winsorise outliers: learn Q1/Q3 on train data, clip to [Q1-k*IQR, Q3+k*IQR]."""

    def __init__(self, k=1.5):
        self.k = k

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1, q3 = np.nanpercentile(X, 25, axis=0), np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_, self.upper_ = q1 - self.k * iqr, q3 + self.k * iqr
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, names=None):
        return names


# ---------------------------------------------------------------- data cleaning
def basic_clean(df, target, id_cols=()):
    """Steps that are safe before splitting. Returns (clean_df, report dict)."""
    report = {"rows_before": len(df), "cols": df.shape[1]}
    report["missing_before"] = df.isna().sum()[lambda s: s > 0].to_dict()
    df = df.drop(columns=[c for c in id_cols if c in df.columns])
    report["duplicates_removed"] = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.dropna(subset=[target])            # rows without a label are unusable
    report["rows_after"] = len(df)
    return df, report


def outlier_summary(df, num_cols, k=1.5):
    out = {}
    for c in num_cols:
        q1, q3 = df[c].quantile([.25, .75])
        iqr = q3 - q1
        n = int(((df[c] < q1 - k * iqr) | (df[c] > q3 + k * iqr)).sum())
        if n:
            out[c] = n
    return out


def build_preprocessor(X):
    num = X.select_dtypes(include="number").columns.tolist()
    cat = X.select_dtypes(exclude="number").columns.tolist()
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("cap", IQRCapper()),
                         ("scale", StandardScaler())])
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                         ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num_pipe, num), ("cat", cat_pipe, cat)]), num, cat


# ---------------------------------------------------------------- models + grids
def model_zoo():
    return {
        "Logistic Regression": (LogisticRegression(max_iter=2000), {"clf__C": [0.01, 0.1, 1, 10]}),
        "Naive Bayes": (GaussianNB(), {"clf__var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3]}),
        "KNN": (KNeighborsClassifier(), {"clf__n_neighbors": [3, 5, 7, 11, 15],
                                         "clf__weights": ["uniform", "distance"]}),
        "SVM": (SVC(probability=True, random_state=RANDOM_STATE),
                {"clf__kernel": ["linear", "rbf", "poly"], "clf__C": [0.1, 1, 10]}),
        "Decision Tree": (DecisionTreeClassifier(random_state=RANDOM_STATE),
                          {"clf__max_depth": [3, 5, 8, None], "clf__min_samples_leaf": [1, 5, 15],
                           "clf__criterion": ["gini", "entropy"]}),
        "Random Forest": (RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
                          {"clf__n_estimators": [100, 200], "clf__max_depth": [5, 10, None],
                           "clf__min_samples_leaf": [1, 5]}),
        "AdaBoost": (AdaBoostClassifier(random_state=RANDOM_STATE),
                     {"clf__n_estimators": [50, 100, 200], "clf__learning_rate": [0.1, 0.5, 1.0]}),
    }


def evaluate(model, X_te, y_te):
    pred = model.predict(X_te)
    proba = model.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, proba)
    return {
        "accuracy": accuracy_score(y_te, pred),
        "precision": precision_score(y_te, pred, zero_division=0),
        "recall": recall_score(y_te, pred, zero_division=0),
        "f1": f1_score(y_te, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_te, proba),
        "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
    }


def run_experiment(csv_path, target, id_cols=("CustomerID",), cv=5, models=None, progress=None):
    ARTIFACTS.mkdir(exist_ok=True)
    raw = pd.read_csv(csv_path)
    df, report = basic_clean(raw, target, id_cols)
    X, y = df.drop(columns=[target]), df[target]
    if y.dtype == object:
        y = (y == y.value_counts().index[-1]).astype(int)   # minority label -> 1

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    report["train_rows"], report["test_rows"] = len(X_tr), len(X_te)
    report["outliers_in_train"] = outlier_summary(X_tr, X_tr.select_dtypes("number").columns)

    pre, num_cols, cat_cols = build_preprocessor(X_tr)
    skf = StratifiedKFold(cv, shuffle=True, random_state=RANDOM_STATE)
    results, best_est = {}, {}
    zoo = {k: v for k, v in model_zoo().items() if models is None or k in models}
    for i, (name, (est, grid)) in enumerate(zoo.items()):
        t0 = time.time()
        pipe = Pipeline([("prep", pre), ("clf", est)])
        gs = GridSearchCV(pipe, grid, scoring="f1", cv=skf, n_jobs=-1).fit(X_tr, y_tr)
        res = evaluate(gs.best_estimator_, X_te, y_te)
        res.update(best_params={k.replace("clf__", ""): v for k, v in gs.best_params_.items()},
                   cv_f1=gs.best_score_, fit_seconds=round(time.time() - t0, 1))
        results[name], best_est[name] = res, gs.best_estimator_
        if progress:
            progress((i + 1) / len(zoo), f"{name} done")

    best_name = max(results, key=lambda k: results[k]["f1"])
    joblib.dump(best_est, ARTIFACTS / "models.joblib")
    meta = {"target": target, "num_cols": num_cols, "cat_cols": cat_cols, "best_model": best_name,
            "cat_options": {c: sorted(X[c].dropna().unique().tolist()) for c in cat_cols},
            "num_ranges": {c: [float(X[c].min()), float(X[c].max()), float(X[c].median())] for c in num_cols},
            "report": report}
    (ARTIFACTS / "results.json").write_text(json.dumps({"meta": meta, "results": results}, default=str))
    return meta, results, best_est
