from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from heart import (FEATURES, NUMERIC, load_cleveland, make_model, repeated_cv, screening_metrics,
                   sha256, threshold_for_sensitivity)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "heart_disease_uci45.zip"
EXPECTED_SHA256 = "b17cd273da9ce1caa4710fce80227ea454d4dbf9fcbc8e6a9121672751563adc"


def _toy(n=120, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.normal(55, 9, n), "trestbps": rng.normal(130, 15, n), "chol": rng.normal(240, 40, n),
        "thalach": rng.normal(150, 20, n), "oldpeak": rng.exponential(1, n), "ca": rng.integers(0, 4, n).astype(float),
        "sex": rng.integers(0, 2, n), "fbs": rng.integers(0, 2, n), "exang": rng.integers(0, 2, n),
        "cp": rng.choice(["typical angina", "asymptomatic"], n), "restecg": rng.choice(["normal", "LV hypertrophy"], n),
        "slope": rng.choice(["upsloping", "flat"], n), "thal": rng.choice(["normal", "reversible defect"], n),
    })
    df.loc[:4, "ca"] = np.nan
    df.loc[5:6, "thal"] = np.nan
    y = ((df["oldpeak"] + df["exang"] + rng.normal(0, 0.5, n)) > 1.5).astype(int).values
    return df, y


def test_threshold_reaches_target_sensitivity():
    y = np.array([1, 1, 1, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.4, 0.3, 0.2, 0.6])
    t = threshold_for_sensitivity(y, p, 0.75)
    assert t == 0.4
    assert screening_metrics(y, p >= t)["sensitivity"] >= 0.75


def test_pipeline_handles_missing_and_categories():
    df, y = _toy()
    model = make_model(LogisticRegression(max_iter=1000)).fit(df[FEATURES], y)
    proba = model.predict_proba(df[FEATURES])[:, 1]
    assert np.isfinite(proba).all() and ((proba >= 0) & (proba <= 1)).all()


def test_repeated_cv_shapes():
    df, y = _toy()
    res = repeated_cv(lambda: make_model(LogisticRegression(max_iter=1000)), df[FEATURES], y, n_repeats=2)
    assert len(res) == 2 and res[0]["proba"].shape == (len(y),)
    assert res[0]["flag_target"].dtype == bool


@pytest.mark.skipif(not RAW.exists(), reason="UCI archive not present")
class TestRealData:
    def test_checksum(self):
        assert sha256(RAW) == EXPECTED_SHA256

    def test_cleveland_structure(self):
        df = load_cleveland(RAW)
        assert df.shape[0] == 303
        assert df["disease"].sum() == 139
        assert df[NUMERIC].isna().sum().to_dict()["ca"] == 4
        assert df["thal"].isna().sum() == 2
