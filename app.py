"""
app.py  –  Parkinson's Tremor Detection from Voice
Run:  streamlit run app.py
"""

import os
import pickle
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from feature_extractor import extract_features

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Parkinson's Voice Detector",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-title  { text-align:center; font-size:2.2rem; font-weight:700; color:#1a1a2e; }
    .sub-title   { text-align:center; color:#555; margin-bottom:1.5rem; }
    .result-box  { border-radius:12px; padding:1.4rem 2rem; margin:1.2rem 0;
                   text-align:center; font-size:1.3rem; font-weight:600; }
    .positive    { background:#ffe0e0; border:2px solid #e53935; color:#c62828; }
    .negative    { background:#e0f2f1; border:2px solid #00897b; color:#00695c; }
    .warning-box { background:#fff8e1; border-left:4px solid #f9a825;
                   padding:.8rem 1rem; border-radius:6px; font-size:.9rem; }
    .feature-tag { display:inline-block; background:#f0f4ff; color:#3949ab;
                   border-radius:20px; padding:2px 10px; margin:2px;
                   font-size:.78rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load model artifacts  (cached so they load only once)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_artifacts():
    import os
    import pandas as pd
    from xgboost import XGBClassifier
    from imblearn.over_sampling import SMOTE
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    FEATURE_COLS = [
        "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
        "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
        "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
        "MDVP:APQ", "Shimmer:DDA",
        "NHR", "HNR",
        "RPDE", "DFA", "spread1", "spread2", "D2", "PPE",
    ]

    # Load pkl files if they exist, otherwise train fresh
    if os.path.exists("model.pkl"):
        with open("model.pkl",        "rb") as f: model        = pickle.load(f)
        with open("scaler.pkl",       "rb") as f: scaler       = pickle.load(f)
        with open("feature_cols.pkl", "rb") as f: feature_cols = pickle.load(f)
    else:
        # Auto-train if pkl files are missing (e.g. on Streamlit Cloud)
        df = pd.read_csv("parkinsons_dataset.csv")
        X  = df[FEATURE_COLS].values
        y  = df["status"].values

        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)

        smote = SMOTE(random_state=42)
        X_bal, y_bal = smote.fit_resample(X_train_s, y_train)

        model = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42,
        )
        model.fit(X_bal, y_bal)
        feature_cols = FEATURE_COLS

    return model, scaler, feature_cols

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<p class="main-title">🧠 Parkinson\'s Voice Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Upload a short voice recording to screen for Parkinson\'s tremor indicators.</p>', unsafe_allow_html=True)

if not model_loaded:
    st.error("⚠️  Model not found. Run `python train_model.py` first to generate model.pkl.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Instructions
# ─────────────────────────────────────────────────────────────────────────────

with st.expander("📋 How to record your voice sample", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**✅ Do this**")
        st.markdown("""
- Record a sustained **'ahh'** vowel (3–6 seconds)
- Sit in a quiet room
- Hold microphone 10–15 cm from mouth
- Breathe naturally before starting
- Save as **WAV** format (16kHz or higher)
        """)
    with col2:
        st.markdown("**❌ Avoid this**")
        st.markdown("""
- Background noise (fans, TV, traffic)
- Very short recordings (< 1 second)
- Clipping / distortion
- MP3 files (convert to WAV first)
- Whispering or shouting
        """)

# ─────────────────────────────────────────────────────────────────────────────
# File uploader
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
uploaded_file = st.file_uploader(
    "Upload your voice recording (WAV)",
    type=["wav"],
    help="Supported format: WAV (mono or stereo). Minimum 1 second.",
)

# ─────────────────────────────────────────────────────────────────────────────
# Prediction pipeline
# ─────────────────────────────────────────────────────────────────────────────

if uploaded_file is not None:

    # Save to temp file so parselmouth / librosa can read it
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.audio(uploaded_file, format="audio/wav")

    with st.spinner("🔬 Extracting vocal biomarkers …"):
        try:
            features = extract_features(tmp_path)
        except ValueError as e:
            st.error(f"🚫 {e}")
            os.unlink(tmp_path)
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error during feature extraction: {e}")
            os.unlink(tmp_path)
            st.stop()
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # Build input vector in the correct column order
    X_input  = np.array([[features[col] for col in feature_cols]])
    X_scaled = scaler.transform(X_input)

    proba    = model.predict_proba(X_scaled)[0]   # [P(healthy), P(parkinsons)]
    pred     = model.predict(X_scaled)[0]
    conf     = proba[pred] * 100

    # ── Result banner ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔍 Prediction Result")

    if pred == 1:
        st.markdown(
            f'<div class="result-box positive">⚠️  Possible Parkinson\'s indicators detected<br>'
            f'<span style="font-size:.95rem;font-weight:400">Confidence: {conf:.1f}%</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="result-box negative">✅  No significant Parkinson\'s indicators detected<br>'
            f'<span style="font-size:.95rem;font-weight:400">Confidence: {conf:.1f}%</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="warning-box">⚠️  <strong>Medical disclaimer:</strong> '
        'This tool is for educational and screening purposes only. '
        'It is <strong>not a substitute for professional medical diagnosis</strong>. '
        'Please consult a neurologist for any health concerns.</div>',
        unsafe_allow_html=True,
    )

    # ── Confidence gauge ──────────────────────────────────────────────────────
    st.markdown("#### Prediction Confidence")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(proba[1] * 100, 1),
        title={"text": "Parkinson's Probability (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": "#e53935" if pred == 1 else "#00897b"},
            "steps": [
                {"range": [0,  40], "color": "#e0f2f1"},
                {"range": [40, 60], "color": "#fff8e1"},
                {"range": [60, 100], "color": "#ffebee"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.75,
                "value": 50,
            },
        },
        number={"suffix": "%"},
    ))
    fig_gauge.update_layout(height=280, margin=dict(t=40, b=10))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # ── Feature breakdown ─────────────────────────────────────────────────────
    with st.expander("📊 Extracted vocal biomarkers", expanded=False):
        feat_df = pd.DataFrame({
            "Feature":     list(features.keys()),
            "Value":       [round(v, 6) for v in features.values()],
        })

        # Load reference ranges from training data
        try:
            ref_df   = pd.read_csv("parkinsons_dataset.csv")
            means    = ref_df[feature_cols].mean()
            stds     = ref_df[feature_cols].std()
            feat_df["Healthy mean ± std"] = [
                f"{means[c]:.4f} ± {stds[c]:.4f}" for c in feature_cols
            ]
        except Exception:
            pass

        st.dataframe(feat_df, use_container_width=True, hide_index=True)

    # ── Feature importance bar chart ──────────────────────────────────────────
    with st.expander("🏆 Top features driving this prediction", expanded=False):
        importances = model.feature_importances_
        sorted_idx  = np.argsort(importances)[::-1][:10]

        fig_bar = go.Figure(go.Bar(
            x=[importances[i] for i in sorted_idx],
            y=[feature_cols[i] for i in sorted_idx],
            orientation="h",
            marker_color="#3949ab",
        ))
        fig_bar.update_layout(
            title="Top-10 Feature Importances ",
            xaxis_title="Importance",
            yaxis={"autorange": "reversed"},
            height=380,
            margin=dict(l=150, r=20, t=50, b=40),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<small>Model: Random Forest · Dataset: UCI Parkinson's Voice (195 samples) · "
    "Built with Streamlit + parselmouth + librosa</small>",
    unsafe_allow_html=True,
)
