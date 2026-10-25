# failure_mode_app.py
# Web tool accompanying: "Machine Learning based Failure Mode Classification for
# Self-Drilling Screw Connections in Cold-formed Steel".
# The four best models of the paper's nested cross-validation (CatBoost, Gradient Boosting,
# XGBoost, ANN) are retrained on all 278 specimens with the selected hyper-parameters.

import os

import pandas as pd
import streamlit as st

import sds_model as sm

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
    df = sm.load_data(EXCEL_PATH)
    return sm.train(df)

with st.spinner("Preparing ML models…"):
    best_models, le, ranges = load_and_train_models()

# ================== UI ==================
st.title("🔩 Failure Mode Predictor for Self Drilling Screw Connection")
st.write("Enter geometric and material parameters maintaining consistent units "
         "(for example mm and MPa). The tool applies to self-drilling screw connections "
         "under quasi-static single-lap shear.")

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
        vals = dict(t1=t1, t2=t2, d=d, e1=e1, e2=e2, p1=p1, p2=p2, b=b, be=be,
                    N=N, Nb=Nb, Nr=Nr, fu1=fu1, fu2=fu2, fy1=fy1, fy2=fy2)
        errors, warnings = sm.check_inputs(vals)
        if errors:
            for e in errors:
                st.error(e)
        else:
            for wmsg in warnings:
                st.warning(wmsg)
            feats = sm.engineer_features(**vals)
            outside = sm.out_of_range(feats, ranges)
            if outside:
                st.warning("Some inputs lie outside the range of the test database, so the prediction "
                           "is an extrapolation and less reliable:\n\n" +
                           "\n".join(f"- {n} = {v:.2f} (database range {lo:.2f}–{hi:.2f})"
                                      for n, v, lo, hi in outside))
            res = sm.predict(best_models, le, feats)

            st.markdown(
                f'<div class="result">Predicted failure mode: <b>{sm.MODE_NAMES.get(res["final"], res["final"])}</b> '
                f'(majority vote: {res["votes"]}/{res["n_models"]} models)</div>',
                unsafe_allow_html=True,
            )
            per = res["per_model"].copy()
            per["Probability"] = (100 * per["Probability"]).round(0).astype(int).astype(str) + "%"
            st.markdown('<div class="result">Model-wise predictions</div>', unsafe_allow_html=True)
            st.table(per.set_index("Model"))

            st.markdown('<div class="result">Mean probability of each failure mode (four models)</div>',
                        unsafe_allow_html=True)
            mp = (100 * res["mean_proba"]).round(1)
            st.bar_chart(pd.DataFrame({"Probability (%)": mp.values}, index=mp.index), horizontal=True)
            if res["mean_proba"].iloc[0] < 0.5:
                st.info("No failure mode reaches a mean probability of 50%; the connection lies near the "
                        "boundary between mechanisms, so consider the two most probable modes.")

            with st.expander("Dimensionless inputs used by the models"):
                st.table(pd.DataFrame({"Value": [round(feats[f], 3) for f in sm.FEATURES]},
                                      index=[sm.FEATURE_NAMES[f] for f in sm.FEATURES]))

st.caption("Models: CatBoost, Gradient Boosting, XGBoost and ANN, retrained on all 278 single-lap shear tests "
           "with the hyper-parameters selected by configuration-grouped nested cross-validation. "
           "Predictions are only as reliable as the database; use them together with the design-code checks.")

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
