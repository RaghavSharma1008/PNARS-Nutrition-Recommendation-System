import streamlit as st

st.set_page_config(
    page_title="About | Personalized Nutrition Analysis & Recommendation System",
    page_icon="ℹ️",
    layout="wide"
)

# ----------------------------- PAGE STYLES -----------------------------
st.markdown("""
<style>

.page-title {
    font-size: 2.8rem;
    font-weight: 800;
    text-align: center;
    margin-top: -10px;
    background: linear-gradient(90deg, #4ade80, #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.section {
    margin-top: 40px;
}

.card {
    padding: 22px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 14px;
    backdrop-filter: blur(10px);
    transition: 0.3s;
}

.card:hover {
    background: rgba(255,255,255,0.1);
}

.footer {
    text-align: center;
    margin-top: 70px;
    color: #bbbbbb;
    font-size: 0.95rem;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------- TITLE ----------------------------
st.markdown("<div class='page-title'>ℹ️ About The Project</div>", unsafe_allow_html=True)
st.write("")
st.write("A modern, AI-driven nutrition system built to help users understand their food, improve their habits, and make smarter dietary choices — powered by data science & machine learning.")

# ---------------------------- SECTIONS ----------------------------

# Mission Section
st.markdown("## 🌱 Our Mission")
st.markdown(
    """
    The goal of the **Personalized Nutrition Analysis & Recommendation System** is simple:  
    to make nutrition easy, scientific, and accessible.

    This project analyzes foods, identifies nutrient patterns, predicts recommendations,
    and helps users make healthier decisions — all through intelligent automation.
    """
)

# What the System Does
st.markdown("## 🚀 What This System Offers")
st.markdown(
    """
    <div class="card">
        <ul>
            <li><b>Smart Nutrition Analysis</b> — Understand calories, macros, & food composition in detail.</li>
            <li><b>AI Recommendation Engine</b> — ML model suggests ideal foods based on user requirements.</li>
            <li><b>Meal Clustering</b> — Group foods using unsupervised learning to reveal eating patterns.</li>
            <li><b>User-Friendly Dashboard</b> — Clean, modern UI for quick access to insights.</li>
            <li><b>Pattern-Based Understanding</b> — Detect trends in food choices to help improve consistency.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)

# Tech Stack
st.markdown("## 🧠 Technology Behind the Project")
col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        <div class='card'>
            <h4>📘 Machine Learning</h4>
            <ul>
                <li>K-Means Meal Clustering</li>
                <li>Food Recommendation Model</li>
                <li>Calorie & Macro Prediction Pipeline</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class='card'>
            <h4>🛠 Tech Stack</h4>
            <ul>
                <li>Python</li>
                <li>Pandas, Scikit-Learn</li>
                <li>Streamlit (Front-End UI)</li>
                <li>Plotly & Matplotlib Visualizations</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

# Developer Info
st.markdown("## 👤 Developed By")
st.markdown(
    """
    <div class="card">
        <h4>🧑‍💻 Project Lead & Developer </h4>
        <p>
            <b> Raghav Sharma </b><br>
            
            Designed, developed & engineered the entire system from UI to ML models.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Footer
st.markdown(
    """
    <div class="footer">
        Built with ❤️ and a passion for AI & nutrition.<br>
        © 2025 Personalized Nutrition Analysis & Recommendation System
    </div>
    """,
    unsafe_allow_html=True
)
