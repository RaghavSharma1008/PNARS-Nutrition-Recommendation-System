# pages/3_📊_Analytics_Dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import os
from io import BytesIO

import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ============================================================
# PAGE CONFIG + THEME
# ============================================================
st.set_page_config(layout="wide", page_title="PNARS — Analytics Dashboard", page_icon="📊")

st.markdown("""
<style>
.main { background-color: #0d1117; }

/* headings */
h1, h2, h3, h4, h5, h6 {
    color: #E2E8F0 !important;
    font-weight: 700 !important;
}

/* metric cards */
.css-1wivap2, .stMetric {
    background: rgba(255,255,255,0.03);
    padding: 16px !important;
    border-radius: 15px !important;
    border: 1px solid rgba(255,255,255,0.08);
}

/* card container */
.card {
    background: rgba(255,255,255,0.05);
    padding: 25px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 30px;
}

/* tables */
table {
    background: rgba(255,255,255,0.05) !important;
}
</style>
""", unsafe_allow_html=True)



# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data
def load_food_df():
    base = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(base, ".."))
    return pd.read_csv(os.path.join(root, "data", "cleaned_nutrition_dataset.csv"))

try:
    df_food = load_food_df()
except:
    st.error("Food dataset missing.")
    st.stop()



# ============================================================
# CHECK SESSION
# ============================================================
if "profile" not in st.session_state or "food_log" not in st.session_state:
    st.warning("⚠ Please complete the User Input page first.")
    if st.button("Go to User Input"):
        st.switch_page("pages/2_🧍_User_Input.py")
    st.stop()

profile = st.session_state["profile"]
food_log = st.session_state["food_log"]



# ============================================================
# TOTALS + ENERGY CALC
# ============================================================
def compute_totals(profile, food_log, df_food):
    total = {"calories":0,"protein":0,"carbs":0,"fat":0}
    rows = []

    for food, grams in food_log.items():
        row = df_food[df_food.food_name == food].iloc[0]
        f = grams/100
        c = row.calories*f
        p = row.protein*f
        cb = row.carbs*f
        ft = row.fat*f

        total["calories"] += c
        total["protein"] += p
        total["carbs"] += cb
        total["fat"] += ft

        rows.append({
            "food_name": food,
            "grams": grams,
            "calories": round(c,1),
            "protein": round(p,1),
            "carbs": round(cb,1),
            "fat": round(ft,1)
        })

    # BMI
    bmi = round(profile["weight"] / ((profile["height_cm"]/100)**2), 2)

    # BMR
    if profile["gender"] == "male":
        bmr = 10*profile["weight"] + 6.25*profile["height_cm"] - 5*profile["age"] + 5
    else:
        bmr = 10*profile["weight"] + 6.25*profile["height_cm"] - 5*profile["age"] - 161
    bmr = round(bmr,1)

    act = profile["activity"]
    activity_map = {"sedentary":1.2,"light":1.375,"moderate":1.55,"active":1.725,"very active":1.9}
    tdee = round(bmr * activity_map.get(act,1.55),1)

    return total, rows, bmi, bmr, tdee


total, rows, bmi, bmr, tdee = compute_totals(profile, food_log, df_food)



# ============================================================
# PAGE HEADER
# ============================================================
st.title("📊 Analytics Dashboard")
st.markdown("Deep insights into your nutrition patterns with premium visualization.")
st.markdown("---")



# ============================================================
# SECTION 1 — SUMMARY CARDS
# ============================================================
st.subheader("📌 Today's Summary")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Calories", f"{round(total['calories'],1)} kcal")
c2.metric("Protein", f"{round(total['protein'],1)} g")
c3.metric("Carbs", f"{round(total['carbs'],1)} g")
c4.metric("Fats", f"{round(total['fat'],1)} g")
c5.metric("BMI", bmi)



# ============================================================
# SECTION 2 — PROFILE + ENERGY
# ============================================================
st.markdown("### 🧍 Personal & Energy Stats")

colA, colB = st.columns([1.2,1])

with colA:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Profile Summary")
    st.table(pd.DataFrame({
        "Value":[
            profile["name"], profile["age"], profile["gender"],
            profile["height_cm"], profile["weight"],
            profile["activity"], profile["goal_ui"], profile["diet"]
        ]
    }, index=["Name","Age","Gender","Height","Weight","Activity","Goal","Diet"]))
    st.markdown("</div>", unsafe_allow_html=True)

with colB:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Energy Calculations")
    st.table(pd.DataFrame({
        "Value":[bmi, bmr, tdee]
    }, index=["BMI","BMR","TDEE"]))
    st.markdown("</div>", unsafe_allow_html=True)



st.markdown("---")



# ============================================================
# SECTION 3 — MACRO BREAKDOWN (Improved)
# ============================================================
st.subheader("🍽 Macro Breakdown")

pie_data = pd.Series({
    "Protein": total["protein"],
    "Carbs": total["carbs"],
    "Fats": total["fat"]
})

fig_pie = px.pie(
    names=pie_data.index,
    values=pie_data.values,
    title="Macro Consumption Share",
    hole=0.45
)
st.plotly_chart(fig_pie, width="stretch")

# Premium comparison bar
fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(
    name="Consumed",
    x=["Protein","Carbs","Fats"],
    y=[total["protein"], total["carbs"], total["fat"]],
    marker_color="#8ab4f8"
))
fig_bar.update_layout(
    title="Macro Intake (in grams)",
    template="plotly_dark"
)
st.plotly_chart(fig_bar, width="stretch")

st.markdown("---")



# ============================================================
# SECTION 4 — NUTRIENT DENSITY PLOT (NEW)
# ============================================================
st.subheader("🔥 Nutrient Density Analysis")

density_df = df_food.copy()
density_df["protein_density"] = density_df["protein"] / density_df["calories"]
density_df["carb_density"] = density_df["carbs"] / density_df["calories"]

fig_density = px.scatter(
    density_df,
    x="protein_density",
    y="carb_density",
    size="calories",
    hover_name="food_name",
    title="Protein Density vs Carb Density",
    color="calories"
)
st.plotly_chart(fig_density, width="stretch")

st.markdown("---")



# ============================================================
# SECTION 5 — CALORIE BALANCE SIMULATION (NEW)
# ============================================================
st.subheader("📈 Calorie Balance Insight")

target = tdee
consumed = total["calories"]

fig_sim = go.Figure()
fig_sim.add_trace(go.Indicator(
    mode="gauge+number+delta",
    value=consumed,
    delta={"reference": target},
    gauge={"axis":{"range":[0, max(target*1.5, consumed*1.5)]}},
    title={"text":"Daily Calorie Status"}
))
st.plotly_chart(fig_sim, width="stretch")

st.markdown("---")



# ============================================================
# SECTION 6 — KMEANS CLUSTERING
# ============================================================
st.subheader("🍱 Meal Clustering Insights")

numeric = ["protein","carbs","fat"]
cluster_df = df_food.dropna(subset=numeric)[["food_name"]+numeric]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(cluster_df[numeric])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_df["cluster"] = kmeans.fit_predict(X_scaled)

fig_clusters = px.scatter(
    cluster_df,
    x="carbs",
    y="protein",
    color=cluster_df["cluster"].astype(str),
    size="fat",
    hover_data=["food_name"],
    title="Protein vs Carbs Clusters"
)
st.plotly_chart(fig_clusters, width="stretch")

st.markdown("---")



# ============================================================
# SECTION 7 — RECOMMENDATION BUTTON
# ============================================================
st.markdown("### ➡️ Want next steps?")
if st.button("Go to Personalized Recommendations →"):
    st.switch_page("pages/4_✨_Recommendations.py")



# ============================================================
# BALLOONS ON FIRST VISIT
# ============================================================
if st.session_state.get("_visited_analytics") is False:
    st.session_state["_visited_analytics"] = True
    st.balloons()
