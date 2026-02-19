# pages/4_📋_Recommendations.py
import streamlit as st
import pandas as pd
import numpy as np
import os
from io import BytesIO

import plotly.express as px
import plotly.graph_objects as go

# try import for PDF
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# your engine (should be the upgraded one)
from src.recommendation_engine import generate_recommendations, load_food_data

st.set_page_config(layout="wide", page_title="PNARS — Recommendations", page_icon="📋")

# ---------------- safety / load dataset ----------------
@st.cache_data
def _load_df():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    path = os.path.join(root, "data", "cleaned_nutrition_dataset.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset missing at: {path}")
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    return df

try:
    df_food = _load_df()
except Exception as e:
    st.error(f"Could not load dataset: {e}")
    st.stop()

# ---------------- session checks ----------------
if "profile" not in st.session_state or "food_log" not in st.session_state:
    st.warning("No profile / food log found. Please fill the User Input page first.")
    if st.button("Go to User Input"):
        st.switch_page("pages/2_🧍_User_Input.py")
    st.stop()

profile = st.session_state["profile"]
food_log = st.session_state["food_log"]  # dict food->grams

# ---------------- small helpers ----------------
def compute_totals_from_foodlog(food_log, df_food):
    total = {"calories":0.0, "protein":0.0, "carbs":0.0, "fat":0.0}
    rows = []
    for food, grams in food_log.items():
        row = df_food[df_food["food_name"] == food]
        if row.empty:
            continue
        row = row.iloc[0]
        f = grams/100.0
        c = float(row.get("calories",0)) * f
        p = float(row.get("protein",0)) * f
        cb = float(row.get("carbs",0)) * f
        fat = float(row.get("fat",0)) * f
        total["calories"] += c
        total["protein"] += p
        total["carbs"] += cb
        total["fat"] += fat
        rows.append({
            "food_name": food, "grams": grams,
            "calories": round(c,1), "protein": round(p,1),
            "carbs": round(cb,1), "fat": round(fat,1)
        })
    return total, rows

def create_pdf_report(report_info, macros, top_foods):
    if not REPORTLAB_AVAILABLE:
        return None
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width_page, height_page = letter
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, height_page-40, "PNARS — Recommendations Report")
    c.setFont("Helvetica", 10)
    y = height_page-80
    for k,v in report_info.items():
        c.drawString(40, y, f"{k}: {v}")
        y -= 14
        if y < 90:
            c.showPage()
            y = height_page-40
    y -= 8
    c.drawString(40, y, "Macro Targets:")
    y -= 14
    c.drawString(60, y, f"Protein (g): {macros['protein_g']}")
    y -= 14
    c.drawString(60, y, f"Carbs (g): {macros['carbs_g']}")
    y -= 14
    c.drawString(60, y, f"Fats (g): {macros['fats_g']}")
    y -= 18
    c.drawString(40, y, "Top recommended foods:")
    y -= 14
    for f in top_foods:
        c.drawString(60, y, f"- {f}")
        y -= 12
        if y < 80:
            c.showPage()
            y = height_page-40
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.read()

# ---------------- compute analysis ----------------
total, detailed_rows = compute_totals_from_foodlog(food_log, df_food)

# compute basic energy numbers
def compute_bmr_tdee(profile):
    weight = float(profile['weight'])
    height_cm = float(profile['height_cm'])
    age = int(profile['age'])
    gender = profile['gender']
    if gender == 'male':
        bmr = 10*weight + 6.25*height_cm - 5*age + 5
    else:
        bmr = 10*weight + 6.25*height_cm - 5*age - 161
    bmr = round(bmr,1)
    activity_map = {"sedentary":1.2,"light":1.375,"moderate":1.55,"active":1.725,"very active":1.9}
    tdee = round(bmr * activity_map.get(profile.get('activity','moderate'),1.55),1)
    return bmr, tdee

bmr, tdee = compute_bmr_tdee(profile)

# call engine
try:
    reco = generate_recommendations(
        age=int(profile["age"]),
        gender=profile["gender"],
        weight=float(profile["weight"]),
        height=float(profile["height_cm"]),
        activity_level=profile["activity"],
        goal=profile["goal"],
        calorie_intake_today=round(total["calories"],1),
        diet_preference=profile["diet"]
    )
except Exception as e:
    st.error(f"Recommendation engine error: {e}")
    st.stop()

target_cal = reco.get("calorie_target", None)
macros = reco.get("macro_targets", {})
meal_plan = reco.get("meal_plan", {})
rec_df = reco.get("recommended_foods", pd.DataFrame())
goal_score = reco.get("goal_alignment_score", None)
insights = reco.get("insights", {})

# ---------------- Page UI ----------------
st.title("📋 Smart Suggestions — Recommendations")
st.markdown("Personalized meal & actionable recommendations based on your profile and today's intake.")
st.markdown("---")

# header row
st.subheader(f"Hello, {profile.get('name','User')} — Recommendations")
hcol1, hcol2, hcol3, hcol4 = st.columns([1,1,1,1])
hcol1.metric("Target Calories", f"{target_cal} kcal" if target_cal else "—")
hcol2.metric("Calories eaten", f"{round(total['calories'],1)} kcal")
hcol3.metric("BMR", f"{bmr} kcal/day")
hcol4.metric("TDEE (maintenance)", f"{tdee} kcal/day")

st.markdown("### Macro status")
m1, m2, m3 = st.columns(3)
m1.metric("Protein (consumed / target)", f"{round(total['protein'],1)} g / {macros.get('protein_g','—')} g")
m2.metric("Carbs (consumed / target)", f"{round(total['carbs'],1)} g / {macros.get('carbs_g','—')} g")
m3.metric("Fats (consumed / target)", f"{round(total['fat'],1)} g / {macros.get('fats_g','—')} g")

st.markdown("---")

# Goal alignment gauge
if goal_score is not None:
    st.markdown("### 🎯 Goal Alignment")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=goal_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Alignment Score"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "#00cc96"},
               'steps': [
                   {'range': [0, 40], 'color': "#ff4d4d"},
                   {'range': [40, 70], 'color': "#ffcc00"},
                   {'range': [70, 100], 'color': "#00cc96"}]},
        delta={'reference': 75, 'increasing': {'color': "green"}}
    ))
    st.plotly_chart(fig_gauge, width="stretch")
    st.markdown(f"**Calorie difference:** {insights.get('calorie_diff',0)} kcal — {insights.get('summary_line','')}")
    st.markdown("---")

# ---------------- Meal Cards (beautiful, concise) ----------------
st.subheader("🍽 Meal Suggestions (Indian-aware)")
cols = st.columns([1,1,1])
def render_card(col, title, items, rationale):
    with col:
        st.markdown(f"#### {title}")
        if items:
            for it in items:
                st.markdown(f"- **{it}**")
        else:
            st.markdown("- _No specific suggestion_")
        if rationale:
            st.markdown(f"*{rationale}*")

render_card(cols[0], "🍳 Breakfast", meal_plan.get("breakfast", []),
            "Protein + slow carbs to start strong.")
render_card(cols[1], "🍛 Lunch", meal_plan.get("lunch", []),
            "Balanced energy for the afternoon.")
render_card(cols[2], "🥘 Dinner", meal_plan.get("dinner", []),
            "Light carbs, prioritize protein.")

st.markdown("#### 🍪 Snacks")
if meal_plan.get("snacks"):
    for s in meal_plan.get("snacks", []):
        st.markdown(f"- **{s}**")
else:
    st.markdown("- _Consider nuts, fruit, or yogurt as quick snacks._")

st.markdown("---")

# ---------------- Fix Your Day (actionable) ----------------
st.subheader("Fix Your Day — Quick Actions")
actions = []
if target_cal:
    diff = round(total["calories"] - target_cal, 1)
    if diff < -150:
        actions.append(f"You're ~{abs(diff)} kcal below target — add a calorie-dense snack (nuts, peanut butter).")
    elif diff > 150:
        actions.append(f"You're ~{diff} kcal above target — reduce refined carbs and fried foods.")
    else:
        actions.append("You're close to your calorie target — maintain current plan.")

prot_gap = round(macros.get("protein_g",0) - total["protein"], 1)
if prot_gap > 0:
    actions.append(f"Add ~{prot_gap} g protein today (eggs, paneer, dal).")
else:
    actions.append("Protein target met — maintain balanced meals.")

# simple suggestions for carbs/fat
if total["carbs"] > macros.get("carbs_g",999)*1.2:
    actions.append("Carb intake is high — prefer whole grains & veggies.")
if total["fat"] > macros.get("fats_g",999)*1.2:
    actions.append("Fat intake is high — avoid fried snacks.")

for a in actions:
    st.markdown(f"- {a}")

st.markdown("---")

# ---------------- Top recommended foods table ----------------
st.subheader("Top Recommended Foods")
if rec_df is None or rec_df.empty:
    st.info("No recommended foods returned by engine.")
else:
    # safe display: convert to string to avoid pyarrow issues
    display_df = rec_df.reset_index(drop=True).copy()
    # keep important columns and rename nicely if present
    keep_cols = [c for c in ["food_name","calories","protein","carbs","fat","score"] if c in display_df.columns]
    if not keep_cols:
        keep_cols = display_df.columns.tolist()[:6]
    display_df = display_df[keep_cols].head(20)
    st.dataframe(display_df.astype(str))

st.markdown("---")

# small donut (consumed macro distribution)
st.subheader("Macro Distribution (Consumed)")
try:
    pie_df = pd.DataFrame({
        "macro": ["Protein","Carbs","Fat"],
        "grams": [round(total["protein"],1), round(total["carbs"],1), round(total["fat"],1)]
    })
    fig = px.pie(pie_df, names='macro', values='grams', title='Consumed macros', hole=0.4)
    st.plotly_chart(fig, width="stretch")
except Exception:
    pass

st.markdown("---")

# ---------------- Export & Navigation ----------------
st.subheader("Export Recommendations")
report_info = {
    "Name": profile.get("name",""),
    "Age": profile.get("age",""),
    "Gender": profile.get("gender",""),
    "Height_cm": profile.get("height_cm",""),
    "Weight_kg": profile.get("weight",""),
    "BMI": round(float(profile.get("weight")) / ((float(profile.get("height_cm"))/100)**2), 2) if profile.get("height_cm") else "",
    "BMR_kcal": bmr,
    "TDEE_kcal": tdee,
    "Target_kcal": target_cal
}
report_df = pd.DataFrame([report_info])
csv = report_df.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Download recommendations (CSV)", data=csv, file_name="pnars_recommendations.csv", mime="text/csv")

top_foods_list = []
try:
    if rec_df is not None and "food_name" in rec_df.columns:
        top_foods_list = rec_df["food_name"].head(8).tolist()
    else:
        top_foods_list = display_df.iloc[:,0].head(8).astype(str).tolist()
except Exception:
    top_foods_list = []

pdf_bytes = create_pdf_report(report_info, macros, top_foods_list)
if pdf_bytes:
    st.download_button("⬇️ Download recommendations (PDF)", data=pdf_bytes, file_name="pnars_recommendations.pdf", mime="application/pdf")
else:
    st.info("PDF export not available (reportlab not installed). CSV provided.")

st.markdown("---")

# footer navigation
nav_col1, nav_col2, nav_col3 = st.columns([1,1,1])
with nav_col1:
    if st.button("← Back to Analytics"):
        st.session_state["_visited_analytics"] = True
        st.switch_page("pages/3_📊_Analytics_Dashboard.py")
with nav_col2:
    if st.button("Go to Meal Clusters →"):
        st.switch_page("pages/5_🍱_Meal_Clusters.py")
with nav_col3:
    if st.button("Edit Profile / Food Log"):
        st.switch_page("pages/2_🧍_User_Input.py")

st.markdown("If you'd like different meal options, tweak your profile on the User Input page and re-run Analytics.")
