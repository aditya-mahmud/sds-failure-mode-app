"""
Model logic for the SDS failure-mode web tool.

The four best classifiers of the nested cross-validation in the paper
(CatBoost, Gradient Boosting, XGBoost, ANN) are retrained on all 278 specimens
with the hyper-parameters selected most often across the 15 outer folds.
"""
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

SEED = 2026

# order of the columns in the database ("For ML" sheet)
FEATURES = ["e1/d", "e2/d", "p1/d", "p2/d", "t1/t2", "be/b", "Nr", "fu1/fy1", "fu2/fy2", "Nb", "N"]

FEATURE_NAMES = {
    "e1/d": "End distance ratio e1/d", "e2/d": "Edge distance ratio e2/d",
    "p1/d": "Pitch ratio p1/d", "p2/d": "Pitch ratio p2/d", "t1/t2": "Thickness ratio t1/t2",
    "be/b": "Width ratio be/b", "Nr": "Number of rows Nr", "fu1/fy1": "Strength ratio fu1/fy1",
    "fu2/fy2": "Strength ratio fu2/fy2", "Nb": "Screws in widest row Nb", "N": "Total number of screws N",
}

MODE_NAMES = {
    "T": "Tilting (T)", "T+B": "Tilting and bearing (T+B)", "T+B+PO": "Tilting, bearing and pull-out (T+B+PO)",
    "T+NSF": "Tilting and net-section fracture (T+NSF)", "NSF": "Net-section fracture (NSF)",
}


def build_models():
    """Hyper-parameters selected most often in the nested cross-validation (no class weighting)."""
    from catboost import CatBoostClassifier
    from xgboost import XGBClassifier
    return {
        "CatBoost": CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05,
                                       random_seed=SEED, verbose=0, allow_writing_files=False),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=2,
                                                        random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=1.0,
                                 eval_metric="mlogloss", random_state=SEED, n_jobs=1),
        "ANN": Pipeline([("scale", MinMaxScaler()),
                         ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, max_iter=3000,
                                               random_state=SEED))]),
    }


def load_data(path):
    df = pd.read_excel(path, sheet_name="For ML")
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")])
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Database is missing columns: {missing}")
    return df


def train(df, models=None):
    """Fit every model on all specimens. Returns (models, label encoder, feature ranges)."""
    X = df[FEATURES].to_numpy(float)
    le = LabelEncoder().fit(df["failure mode"])
    y = le.transform(df["failure mode"])
    models = models if models is not None else build_models()
    for m in models.values():
        m.fit(X, y)
    ranges = {f: (float(df[f].min()), float(df[f].max())) for f in FEATURES}
    return models, le, ranges


def engineer_features(t1, t2, d, e1, e2, p1, p2, b, be, N, Nb, Nr, fu1, fu2, fy1, fy2):
    """The 16 physical inputs -> the 11 dimensionless model inputs."""
    return {
        "e1/d": e1 / d, "e2/d": e2 / d, "p1/d": p1 / d, "p2/d": p2 / d, "t1/t2": t1 / t2,
        "be/b": be / b, "Nr": float(Nr), "fu1/fy1": fu1 / fy1, "fu2/fy2": fu2 / fy2,
        "Nb": float(Nb), "N": float(N),
    }


def check_inputs(v):
    """Return (errors, warnings) for a dict of the 16 physical inputs."""
    errors, warnings = [], []
    positive = ["t1", "t2", "d", "e1", "e2", "b", "be", "fu1", "fu2", "fy1", "fy2"]
    for k in positive:
        if v.get(k) is None or v[k] <= 0:
            errors.append(f"{k} must be greater than zero.")
    for k in ["p1", "p2"]:
        if v.get(k) is None or v[k] < 0:
            errors.append(f"{k} cannot be negative (use 0 for a single row or column).")
    for k in ["N", "Nb", "Nr"]:
        if v.get(k) is None or v[k] < 1:
            errors.append(f"{k} must be at least 1.")
    if errors:
        return errors, warnings
    if v["Nb"] > v["N"]:
        errors.append("Nb cannot exceed the total number of screws N.")
    if v["Nr"] > v["N"]:
        errors.append("Nr cannot exceed the total number of screws N.")
    if v["Nb"] * v["Nr"] < v["N"]:
        errors.append("Nb × Nr must be at least N (Nb is the largest number of screws in one row).")
    if v["be"] > v["b"]:
        errors.append("The effective width be cannot exceed the sheet width b.")
    if v["Nr"] > 1 and v["p1"] == 0:
        warnings.append("p1 = 0 but more than one row is given; p1 is the spacing between rows.")
    if v["Nb"] > 1 and v["p2"] == 0:
        warnings.append("p2 = 0 but more than one screw per row is given; p2 is the spacing within a row.")
    for i in ("1", "2"):
        if v[f"fu{i}"] < v[f"fy{i}"]:
            warnings.append(f"fu{i} is lower than fy{i}; please check the material properties.")
    return errors, warnings


def out_of_range(feats, ranges):
    """Inputs outside the ranges of the database (Table 3 of the paper)."""
    out = []
    for f in FEATURES:
        lo, hi = ranges[f]
        if feats[f] < lo - 1e-9 or feats[f] > hi + 1e-9:
            out.append((FEATURE_NAMES[f], feats[f], lo, hi))
    return out


def predict(models, le, feats):
    """Per-model prediction and class probabilities, majority vote and mean probabilities."""
    x = np.array([[feats[f] for f in FEATURES]], dtype=float)
    labels = list(le.classes_)
    rows, probs = [], []
    for name, m in models.items():
        p = np.asarray(m.predict_proba(x))[0]
        k = int(np.argmax(p))
        rows.append({"Model": name, "Predicted failure mode": labels[k], "Probability": p[k]})
        probs.append(p)
    mean_p = np.mean(probs, axis=0)
    votes = Counter(r["Predicted failure mode"] for r in rows)
    top = max(votes.values())
    tied = [lab for lab, n in votes.items() if n == top]
    final = max(tied, key=lambda lab: mean_p[labels.index(lab)])     # ties broken by mean probability
    return {
        "per_model": pd.DataFrame(rows),
        "mean_proba": pd.Series(mean_p, index=labels).sort_values(ascending=False),
        "final": final,
        "votes": votes[final],
        "n_models": len(rows),
    }
