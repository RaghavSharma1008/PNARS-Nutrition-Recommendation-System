import streamlit as st

st.set_page_config(
    page_title="Home | Personalized Nutrition Analysis & Recommendation System - PNARS",
    page_icon="🍏",
    layout="wide"
)

# ----------------------------- PAGE STYLES -----------------------------
st.markdown(
    """
    <style>
        /* Main title */
        .main-title {
            font-size: 3rem;
            font-weight: 800;
            text-align: center;
            margin-top: -20px;
            background: linear-gradient(90deg, #4ade80, #22d3ee);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        /* Subtitle */
        .subtitle {
            text-align: center;
            font-size: 1.25rem;
            color: #cccccc;
            margin-top: -10px;
        }

        /* Feature badges */
        .feature-box {
            border-radius: 12px;
            padding: 18px;
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.15);
            backdrop-filter: blur(10px);
            transition: 0.3s;
        }
        .feature-box:hover {
            transform: translateY(-4px);
            background: rgba(255,255,255,0.1);
        }

        /* Button spacing */
        .center-btn {
            display: flex;
            justify-content: center;
            margin-top: 25px;
        }

        /* Footer */
        .footer {
            text-align: center;
            margin-top: 80px;
            padding-bottom: 15px;
            font-size: 0.9rem;
            color: #999999;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------- MAIN CONTENT -----------------------------

st.markdown("<h1 class='main-title'>🍎 Personalized Nutrition Analysis & Recommendation System - PNARS</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Your intelligent assistant for healthier choices — powered by data, ML & nutrition science.</p>", unsafe_allow_html=True)

st.write("")

# --------------------------- FEATURE GRID -----------------------------
feature_cols = st.columns(3)

with feature_cols[0]:
    st.markdown(
        """
        <div class="feature-box">
            <h3>📊 Smart Nutrition Analysis</h3>
            <p>Deep insights into calories, macros, micros & food quality scoring.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with feature_cols[1]:
    st.markdown(
        """
        <div class="feature-box">
            <h3>🤖 ML-Based Food Recommendations</h3>
            <p>AI-powered suggestions tailored to your health profile.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with feature_cols[2]:
    st.markdown(
        """
        <div class="feature-box">
            <h3>🍽 Meal Clustering</h3>
            <p>Discover patterns in foods you eat using unsupervised ML.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# --------------------------- CTA BUTTON -----------------------------
st.markdown("<div class='center-btn'>", unsafe_allow_html=True)

user_input_page = st.Page(
    page="pages/2_🧍_User_Input.py" if hasattr(st, "Page") else None
)


if st.button("🚀 Start Analysis", use_container_width=False):
    st.switch_page("pages/2_🧍_User_Input.py")

st.markdown("</div>", unsafe_allow_html=True)

# --------------------------- FOOTER -----------------------------
st.markdown(
    """
    <div class="footer">
        Built with ❤️ using Streamlit & Machine Learning<br>
        © 2025 Personalized Nutrition Analysis & Recommendation System
    </div>
    """,
    unsafe_allow_html=True
)
