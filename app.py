# app.py
# Structural Number Prediction — Flexible Pavement Design Tool
# Autores: Ing. Leonardo Zambrano & Claude (Anthropic)
# TICEC 2026 | Springer CCIS
# Dataset: https://zenodo.org/records/20585688 | DOI: 10.5281/zenodo.20585688

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import joblib
import shap
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="SN Predictor — Flexible Pavement",
    page_icon="🛣️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# =============================================================================
# STYLE
# =============================================================================
st.markdown("""
<style>
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .result-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 1.5rem 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .result-label {
        color: #aaa;
        font-size: 0.95rem;
        margin-bottom: 0.3rem;
    }
    .result-value {
        color: #ffffff;
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: 0.05rem;
    }
    .result-unit {
        color: #aaa;
        font-size: 0.9rem;
        margin-top: 0.2rem;
    }
    .interp-box {
        background: #f8f9fa;
        border-left: 4px solid #1a1a2e;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.2rem;
        margin: 0.8rem 0;
        font-size: 0.95rem;
        color: #333;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    .badge-low  { background:#d4edda; color:#155724; }
    .badge-med  { background:#fff3cd; color:#856404; }
    .badge-high { background:#f8d7da; color:#721c24; }
    .footer {
        font-size: 0.78rem;
        color: #999;
        text-align: center;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# LOAD MODEL
# =============================================================================
@st.cache_resource
def load_model():
    model     = joblib.load('modelo_XGBR.pkl')
    explainer = shap.TreeExplainer(model)
    return model, explainer

try:
    model, explainer = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"⚠️ Could not load model: {e}")
    st.stop()

# =============================================================================
# HEADER
# =============================================================================
st.markdown('<div class="main-title">🛣️ Structural Number Predictor</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Flexible Pavement Design Tool — '
    'Machine Learning + SHAP Interpretability</div>',
    unsafe_allow_html=True)

st.markdown(
    "Enter the subgrade soil properties below. The model predicts the "
    "**Structural Number (SN)** required for flexible pavement design "
    "using an XGBoost model trained on a synthetic dataset generated "
    "with the equation of Mohammad et al. (1999) and AASHTO 93."
)

# =============================================================================
# INPUT FORM
# =============================================================================
st.markdown("---")
st.subheader("📋 Subgrade Soil Properties")

col1, col2, col3 = st.columns(3)

with col1:
    w_pct = st.number_input(
        label="Moisture Content — w (%)",
        min_value=10.0,
        max_value=40.0,
        value=20.0,
        step=0.5,
        format="%.1f",
        help="Compaction moisture content. Valid range: 10–40%"
    )

with col2:
    gd_knm3 = st.number_input(
        label="Dry Unit Weight — γd (kN/m³)",
        min_value=12.6,
        max_value=18.1,
        value=17.0,
        step=0.1,
        format="%.2f",
        help="Dry unit weight of the subgrade soil. Valid range: 12.6–18.1 kN/m³ (80–115 pcf)"
    )

with col3:
    PI = st.number_input(
        label="Plasticity Index — PI",
        min_value=4.0,
        max_value=50.0,
        value=15.0,
        step=1.0,
        format="%.1f",
        help="Plasticity Index. Valid range: 4–50. A-4: 4–10, A-6: 11–25, A-7-6: >25"
    )

# AASHTO parameters info
with st.expander("⚙️ Fixed AASHTO 93 Design Parameters"):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("W₁₈ (ESALs)", "5 × 10⁶")
    c2.metric("ZR", "−1.282")
    c3.metric("So", "0.45")
    c4.metric("ΔPSI", "2.5")
    st.caption(
        "These parameters correspond to a reliability level of R = 95%, "
        "consistent with urban freeways and principal arterials (AASHTO 1993)."
    )

st.markdown("---")

# =============================================================================
# PREDICTION
# =============================================================================
predict_btn = st.button("🔍 Predict Structural Number", type="primary",
                        use_container_width=True)

if predict_btn:
    X_input = np.array([[w_pct, gd_knm3, PI]])

    # Prediction
    SN_pred = model.predict(X_input)[0]

    # SHAP values
    shap_vals = explainer.shap_values(X_input)[0]
    base_val  = explainer.expected_value
    feat_names = ['w (%)', 'γd (kN/m³)', 'PI']

    # ==========================================================================
    # RESULT BOX
    # ==========================================================================
    st.markdown("---")
    st.subheader("📊 Prediction Result")

    st.markdown(f"""
    <div class="result-box">
        <div class="result-label">Predicted Structural Number</div>
        <div class="result-value">SN = {SN_pred:.3f}</div>
        <div class="result-unit">Dimensionless | AASHTO 93 Compatible</div>
    </div>
    """, unsafe_allow_html=True)

    # SN interpretation badge
    if SN_pred < 3.5:
        badge = '<span class="badge badge-low">Low structural demand</span>'
        sn_interp = "The predicted SN indicates a **low structural demand**, typical of lightly trafficked roads or high-quality subgrades."
    elif SN_pred < 5.0:
        badge = '<span class="badge badge-med">Moderate structural demand</span>'
        sn_interp = "The predicted SN indicates a **moderate structural demand**, typical of collector and arterial roads under medium traffic."
    else:
        badge = '<span class="badge badge-high">High structural demand</span>'
        sn_interp = "The predicted SN indicates a **high structural demand**, requiring thicker pavement layers typical of heavily trafficked roads."

    st.markdown(badge, unsafe_allow_html=True)
    st.markdown(sn_interp)

    # ==========================================================================
    # SHAP INTERPRETATION
    # ==========================================================================
    st.markdown("---")
    st.subheader("🔬 SHAP Interpretability Analysis")

    # Identify most influential feature
    abs_shap    = np.abs(shap_vals)
    top_idx     = np.argmax(abs_shap)
    top_feat    = feat_names[top_idx]
    top_val     = shap_vals[top_idx]
    top_dir     = "increases" if top_val > 0 else "decreases"
    top_input   = [w_pct, gd_knm3, PI][top_idx]

    # Second most influential
    second_idx  = np.argsort(abs_shap)[-2]
    second_feat = feat_names[second_idx]
    second_val  = shap_vals[second_idx]
    second_dir  = "increases" if second_val > 0 else "decreases"

    st.markdown(f"""
    <div class="interp-box">
    <b>Most influential variable:</b> {top_feat} = {top_input:.2f}<br>
    This feature <b>{top_dir}</b> the predicted SN by <b>{abs(top_val):.4f}</b> units
    relative to the model baseline (E[f(X)] = {base_val:.3f}).<br><br>
    <b>Second most influential:</b> {second_feat}, which <b>{second_dir}</b> SN
    by <b>{abs(second_val):.4f}</b> units.<br><br>
    Together, these two variables explain most of the deviation from the
    average prediction.
    </div>
    """, unsafe_allow_html=True)

    # Feature contribution table
    contrib_df = pd.DataFrame({
        'Feature'     : feat_names,
        'Input Value' : [f"{w_pct:.2f}", f"{gd_knm3:.2f}", f"{PI:.2f}"],
        'SHAP Value'  : [f"{v:+.4f}" for v in shap_vals],
        'Direction'   : ["↑ Increases SN" if v > 0 else "↓ Decreases SN"
                         for v in shap_vals],
    })
    st.dataframe(contrib_df, use_container_width=True, hide_index=True)

    # ==========================================================================
    # WATERFALL PLOT
    # ==========================================================================
    st.subheader("📉 SHAP Waterfall Plot")
    st.caption("Shows how each feature pushes the prediction above or below the baseline.")

    matplotlib.rcParams.update({
        'font.family'    : 'serif',
        'font.size'      : 11,
        'axes.facecolor' : 'white',
        'figure.facecolor': 'white',
    })

    exp_single = shap.Explanation(
        values        = shap_vals,
        base_values   = base_val,
        data          = X_input[0],
        feature_names = feat_names)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    plt.sca(ax)
    shap.waterfall_plot(exp_single, show=False)
    ax.set_title(
        f"SHAP Waterfall — w={w_pct:.1f}%, γd={gd_knm3:.2f} kN/m³, PI={PI:.0f}  |  "
        f"Predicted SN = {SN_pred:.3f}",
        fontsize=10, fontweight='bold', pad=8)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # ==========================================================================
    # DESIGN GUIDANCE
    # ==========================================================================
    st.markdown("---")
    st.subheader("📐 Pavement Design Guidance")

    # Approximate layer thickness for reference
    a1 = 0.44  # asphalt layer coefficient (typical HMA)
    a2 = 0.14  # base layer coefficient (crushed stone)
    m2 = 1.0   # drainage coefficient

    D1_min = 3.0  # inches (AASHTO minimum for high-traffic)
    SN1    = a1 * D1_min
    SN2    = SN_pred - SN1
    D2     = max(0, SN2 / (a2 * m2))

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Predicted SN", f"{SN_pred:.3f}")
        st.metric("Asphalt Layer Coefficient (a₁)", "0.44")
        st.metric("Base Layer Coefficient (a₂)", "0.14")
    with col_b:
        st.metric("Minimum Asphalt Thickness (D₁)", f"{D1_min:.1f} in")
        st.metric("Estimated Base Thickness (D₂)", f"{D2:.1f} in",
                  help="Approximate. Final design requires full AASHTO 93 procedure.")

    st.info(
        "ℹ️ **Note:** Layer thicknesses shown above are approximate estimates "
        "based on typical layer coefficients. Final pavement design must follow "
        "the complete AASHTO 1993 procedure, including drainage coefficients, "
        "minimum thickness requirements, and material-specific layer coefficients."
    )

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
<div class="footer">
    <b>Structural Number Predictor</b> | TICEC 2026 | Springer CCIS<br>
    Model: XGBoost (R² = 0.9995 | RMSE = 0.0201) trained on synthetic dataset<br>
    Dataset DOI: <a href="https://zenodo.org/records/20585688" target="_blank">10.5281/zenodo.20585688</a> |
    Equation: Mohammad et al. (1999). <i>Transp. Res. Rec.</i>, 1687, 47–54.
    <a href="https://doi.org/10.3141/1687-06" target="_blank">DOI: 10.3141/1687-06</a><br>
    Developed by Ing. Leonardo Zambrano (UTI Ecuador) &amp; Claude (Anthropic) — 2026
</div>
""", unsafe_allow_html=True)