"""Loading, pipelines and evaluation helpers for the UCI Cleveland heart disease analysis."""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
           "exang", "oldpeak", "slope", "ca", "thal", "num"]
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
BINARY = ["sex", "fbs", "exang"]
CATEGORICAL = ["cp", "restecg", "slope", "thal"]
FEATURES = NUMERIC + BINARY + CATEGORICAL

# UCI codes, from heart-disease.names
LABELS = {
    "cp": {1: "typical angina", 2: "atypical angina", 3: "non-anginal pain", 4: "asymptomatic"},
    "restecg": {0: "normal", 1: "ST-T abnormality", 2: "LV hypertrophy"},
    "slope": {1: "upsloping", 2: "flat", 3: "downsloping"},
    "thal": {3: "normal", 6: "fixed defect", 7: "reversible defect"},
}


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_cleveland(path: str | Path) -> pd.DataFrame:
    """Read processed.cleveland.data from the UCI archive; add binary `disease` (num > 0)."""
    with zipfile.ZipFile(path) as z:
        raw = z.read("processed.cleveland.data")
    df = pd.read_csv(io.BytesIO(raw), header=None, names=COLUMNS, na_values="?")
    for col, mapping in LABELS.items():
        df[col] = df[col].map(mapping)
    df["disease"] = (df["num"] > 0).astype(int)
    return df


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
        ("bin", "passthrough", BINARY),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"))]), CATEGORICAL),
    ])


def make_model(estimator) -> Pipeline:
    return Pipeline([("pre", make_preprocessor()), ("model", estimator)])


def threshold_for_sensitivity(y, proba, target: float) -> float:
    """Highest probability threshold whose sensitivity on (y, proba) reaches `target`."""
    y = np.asarray(y)
    pos = np.sort(np.asarray(proba)[y == 1])[::-1]
    k = int(np.ceil(target * len(pos)))
    return float(pos[k - 1])


def screening_metrics(y, flagged) -> dict[str, float]:
    y = np.asarray(y).astype(bool)
    f = np.asarray(flagged).astype(bool)
    tp, fn, tn, fp = (f & y).sum(), (~f & y).sum(), (~f & ~y).sum(), (f & ~y).sum()
    div = lambda a, b: a / b if b else float("nan")
    return {"sensitivity": div(tp, tp + fn), "specificity": div(tn, tn + fp),
            "PPV": div(tp, tp + fp), "NPV": div(tn, tn + fn)}


def repeated_cv(model_factory, X, y, n_repeats=20, n_splits=5, target_sensitivity=0.90, seed=0):
    """Repeated stratified CV returning out-of-fold probabilities and two sets of flags per repeat.

    `flag_05` uses a fixed 0.5 threshold. `flag_target` uses a threshold chosen
    on the training fold only (from inner 5-fold out-of-fold probabilities) to
    reach `target_sensitivity`, then applied unchanged to the test fold.
    """
    y = np.asarray(y)
    results = []
    for r in range(n_repeats):
        proba = np.zeros(len(y))
        flag_target = np.zeros(len(y), dtype=bool)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        for train, test in cv.split(X, y):
            Xtr, ytr = X.iloc[train], y[train]
            inner = cross_val_predict(model_factory(), Xtr, ytr, method="predict_proba",
                                      cv=StratifiedKFold(5, shuffle=True, random_state=seed + r))[:, 1]
            t = threshold_for_sensitivity(ytr, inner, target_sensitivity)
            model = model_factory().fit(Xtr, ytr)
            proba[test] = model.predict_proba(X.iloc[test])[:, 1]
            flag_target[test] = proba[test] >= t
        results.append({"proba": proba, "flag_05": proba >= 0.5, "flag_target": flag_target})
    return results


def bootstrap_ci(metric, y_true, y_score, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].min() == y_true[idx].max():
            continue
        stats.append(metric(y_true[idx], y_score[idx]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(metric(y_true, y_score)), float(lo), float(hi)
