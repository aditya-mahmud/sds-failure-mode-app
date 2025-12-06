# failure_mode_app.py

import streamlit as st
import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# ================== CONFIG ==================
st.set_page_config(page_title="SDS Failure Mode Predictor", layout="wide")

# ================== STYLES (AESTHETICS) ==================
st.markdown("""
<style>
:root{ --primary:#0ea5a4; --accent:#2563eb; --bg1:#f6f9ff; --card:#ffffff; --ink:#0f172a; --muted:#64748b; }
html, body, [class*="css"] { font-size: 13px !important; }
h1{ font-size:22px !important; color:var(--ink); margin-bottom:10px !important; }
h1:after{ content:""; display:block; width:260px; height:3px; background:linear-gradient(90deg,var(--accent),var(--primary)); border-radius:6px; margin-top:6px; }
.main{ background: linear-gradient(180deg, var(--bg1) 0%, #ffffff 100%); }
.section{ background:var(--card); border:1px solid #e5e7eb; border-radius:12px; padding:14px; box-shadow:0 6px 24px rgba(15,23,42,.06); margin-bottom:12px; }
.stNumberInput>div{ background:#fff !important; border:1px solid #d1d5db !important; border-radius:10px !important; box-shadow:0 1px 2px rgba(0,0,0,.03) inset; }
.stNumberInput input{ padding:8px 10px !important; font-size:13px !important; }
.stNumberInput>div:focus-within{ border-color:var(--accent) !important; box-shadow:0 0 0 3px rgba(37,99,235,.18) !important; }
div[data-baseweb="input"] > div { width: 190px !important; }
.label {
    font-weight:400;
    color:var(--ink);
    margin-bottom:2px;
    font-size:13px;
}
.label b {
    font-weight:700;
}
.stButton>button{
    background:linear-gradient(90deg,var(--primary),#0891b2);
    color:#fff; border:0; border-radius:10px;
    padding:10px 16px; font-weight:800; font-size:13px;
    box-shadow:0 6px 18px rgba(8,145,178,.35);
}
.result{
    background:linear-gradient(90deg,#ecfeff,#eff6ff);
    border:1px solid #bfdbfe;
    color:#1e3a8a;
    border-radius:12px;
    padding:12px 14px;
    font-size:13px;
    font-weight:700;
    margin-top:8px;
}
.caption{ font-size:11px; color:#64748b; text-align:center; margin-top:6px; }
.sds-img img{ width:320px !important; max-width:100% !important; height:auto !important; }
@media (max-width:1200px){ .sds-img img{ width:420px !important; } }
@media (max-width:640px){ .sds-img img{ width:100% !important; } }
</style>
""", unsafe_allow_html=True)

# ================== HELPERS ==================
def safe_div(numer, denom, eps=1e-9):
    """Safe division to avoid zero-division crashes."""
    denom = float(denom)
    if abs(denom) < eps:
        denom = eps
    return float(numer) / denom

def labeled_number_input(html_label, key, *, min_value=0.0, step=0.1, value=None, fmt=None, int_mode=False):
    """Label with HTML (subs, bold) + matching styled number_input."""
    st.markdown(f'<div class="label">{html_label}</div>', unsafe_allow_html=True)

    if int_mode:
        return st.number_input(
            " ",
            key=key,
            min_value=int(min_value),
            step=int(step),
            value=(int(value) if value is not None else None),
            format="%d",
            label_visibility="collapsed",
        )
    else:
        return st.number_input(
            " ",
            key=key,
            min_value=min_value,
            step=step,
            value=value,
            format=(fmt or None),
            label_visibility="collapsed",
        )

# ================== PATHS ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXCEL_PATH = os.path.join(BASE_DIR, "screw_database_ML.xlsx")
CONN_IMG_PATH = os.path.join(BASE_DIR, "Connection_Configuration.png")
SCREW_IMG_PATH = os.path.join(BASE_DIR, "Screw.jpg")

# ================== MODEL TRAINING (CACHED) ==================
@st.cache_resource
def load_and_train_models():
    df = pd.read_excel(EXCEL_PATH, sheet_name="For ML")

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    X = df.drop(columns=["failure mode"])
    y = df["failure mode"]

    feature_order = X.columns.tolist()  # expected: [e1/d, e2/d, p1/d, ..., Nb, N, Nr]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=51, stratify=y_encoded
    )

    models = {
        "Random Forest": RandomForestClassifier(
            random_state=51,
            max_depth=10,
            min_samples_split=4,
            min_samples_leaf=2,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=51),
        "XGBoost": XGBClassifier(
            eval_metric="mlogloss",
            random_state=51,
            max_depth=5,
            learning_rate=0.1,
            n_estimators=100,
        ),
        "CatBoost": CatBoostClassifier(
            verbose=0,
            depth=5,
            learning_rate=0.1,
            iterations=200,
        ),
    }

    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models, scaler, le, feature_order

with st.spinner("Preparing ML models…"):
    best_models, scaler, le, feature_order = load_and_train_models()

# ================== FEATURE ENGINEERING ==================
def engineer_features(t1, t2, d, e1, e2, p1, p2, b, be, N, Nb, Nr, fu1, fu2, fy1, fy2):
    """
    Convert the 16 physical inputs into the 11 ML features:
    e1/d, e2/d, p1/d, p2/d, t1/t2, be/b, fu1/fy1, fu2/fy2, Nb, N, Nr
    """
    return {
        "e1/d":    safe_div(e1, d),
        "e2/d":    safe_div(e2, d),
        "p1/d":    safe_div(p1, d),
        "p2/d":    safe_div(p2, d),
        "t1/t2":   safe_div(t1, t2),
        "be/b":    safe_div(be, b),
        "fu1/fy1": safe_div(fu1, fy1),
        "fu2/fy2": safe_div(fu2, fy2),
        "Nb":      float(Nb),
        "N":       float(N),
        "Nr":      float(Nr),
    }

def build_vector(feature_order, feat_dict):
    """Return feature vector in the exact order used during training."""
    missing = [name for name in feature_order if name not in feat_dict]
    if missing:
        raise ValueError(f"Missing engineered features: {missing}")
    return np.array([feat_dict[name] for name in feature_order], dtype=np.float64).reshape(1, -1)

# ================== UI ==================
st.title("🔩 Failure Mode Predictor for Self Drilling Screw Connection")
st.write("Enter geometric and material parameters maintaining consistent units.")

col_inputs, col_image = st.columns([3, 1])

# ---------- INPUTS ----------
with col_inputs:
    st.markdown('<div class="section">', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        t1 = labeled_number_input("<b>t<sub>1</sub></b> (Thickness of sheet in contact with screw head)", "t1")
        t2 = labeled_number_input("<b>t<sub>2</sub></b> (Thickness of sheet not in contact with screw head)", "t2")
        d  = labeled_number_input("<b>d</b> (Screw diameter)", "d")
        e1 = labeled_number_input("<b>e<sub>1</sub></b> (End distance)", "e1")
        e2 = labeled_number_input("<b>e<sub>2</sub></b> (Edge distance)", "e2")
        p1 = labeled_number_input("<b>p<sub>1</sub></b> (Pitch parallel to loading)", "p1")
        p2 = labeled_number_input("<b>p<sub>2</sub></b> (Pitch perpendicular to loading)", "p2")
        b  = labeled_number_input("<b>b</b> (Sheet width)", "b")

    with c2:
        be  = labeled_number_input("<b>b<sub>e</sub></b> (Effective width)", "be")
        N   = labeled_number_input("<b>N</b> (Total number of screws)", "N", int_mode=True, step=1)

        Nb  = labeled_number_input(
            "<b>N<sub>b</sub></b> (Number of screws in the row with the highest screws)",
            "Nb", int_mode=True, step=1
        )

        Nr  = labeled_number_input(
            "<b>N<sub>r</sub></b> (Number of rows)",
            "Nr", int_mode=True, step=1
        )

        fu1 = labeled_number_input(
            "<b>f<sub>u1</sub></b> (Tensile strength of sheet in contact with screw head)",
            "fu1", step=1.0
        )
        fu2 = labeled_number_input(
            "<b>f<sub>u2</sub></b> (Tensile strength of sheet not in contact with screw head)",
            "fu2", step=1.0
        )
        fy1 = labeled_number_input(
            "<b>f<sub>y1</sub></b> (Yield strength of sheet in contact with screw head)",
            "fy1", step=1.0
        )
        fy2 = labeled_number_input(
            "<b>f<sub>y2</sub></b> (Yield strength of sheet not in contact with screw head)",
            "fy2", step=1.0
        )

    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Predict Failure Mode"):
        try:
            # 1) Engineer features
            feats = engineer_features(t1, t2, d, e1, e2, p1, p2, b, be, N, Nb, Nr, fu1, fu2, fy1, fy2)

            # 2) Build feature vector in correct order
            X_vec = build_vector(feature_order, feats)

            # 3) Scale and predict
            X_scaled = scaler.transform(X_vec)

            predictions = {}
            for name, model in best_models.items():
                pred_enc = model.predict(X_scaled)
                pred_label = le.inverse_transform(pred_enc)[0]
                predictions[name] = pred_label

            from collections import Counter
            final_label, votes = Counter(predictions.values()).most_common(1)[0]

            # 4) Show results
            st.markdown(
                '<div class="result">Model-wise Predicted Failure Modes:</div>',
                unsafe_allow_html=True,
            )
            res_df = pd.DataFrame(
                {
                    "Model": list(predictions.keys()),
                    "Predicted Failure Mode": list(predictions.values()),
                }
            )
            st.table(res_df)

            st.markdown(
                f'<div class="result">✅ Final Failure Mode (Majority Vote): '
                f'<b>{final_label}</b> (votes: {votes}/{len(predictions)})</div>',
                unsafe_allow_html=True,
            )

        except Exception as e:
            st.error(f"⚠️ Error: {e}")

# ---------- IMAGES ----------
with col_image:
    st.markdown('<div class="section sds-img">', unsafe_allow_html=True)

    if os.path.exists(CONN_IMG_PATH):
        st.image(CONN_IMG_PATH, caption=None, use_container_width=True)
    else:
        st.warning(f"Connection_Configuration.png not found at:\n{CONN_IMG_PATH}")

    if os.path.exists(SCREW_IMG_PATH):
        st.image(SCREW_IMG_PATH, caption=None, use_container_width=True)
    else:
        st.warning(f"Screw.png not found at:\n{SCREW_IMG_PATH}")

    st.markdown(
        '<div class="caption">Screw Connection & Self-Drilling Screw Geometry</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)
