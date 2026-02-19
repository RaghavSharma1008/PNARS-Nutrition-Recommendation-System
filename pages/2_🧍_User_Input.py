import streamlit as st
import pandas as pd
import os
from io import BytesIO

st.set_page_config(layout="wide", page_title="SNP — Profile & Food Log", page_icon="🥗")

# ---------------------------------- Load dataset ----------------------------------
@st.cache_data
def load_food_df():
    base = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(base, ".."))
    data_path = os.path.join(root, "data", "cleaned_nutrition_dataset.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find dataset at: {data_path}")
    return pd.read_csv(data_path)

try:
    df_food = load_food_df()
    food_names = df_food["food_name"].tolist()
except Exception as e:
    st.error(f"Failed to load food dataset: {e}")
    st.stop()

# ---------------------------------- Helpers ----------------------------------
def height_dropdown_values(min_ft=4, max_ft=7):
    vals = []
    for ft in range(min_ft, max_ft+1):
        for inch in range(0, 12):
            vals.append(f"{ft} ft {inch} in")
    return vals

def height_to_cm(h_str):
    try:
        ft = int(h_str.split()[0])
        inch = int(h_str.split()[2])
        return round(ft*30.48 + inch*2.54, 1)
    except:
        return None

# ---------------------------------- UI ----------------------------------
st.title("🧍 PNARS: Profile & Food Log")
st.markdown("Fill your profile and food log. Analysis will move to the Analytics page.")
st.markdown("---")

col_profile, col_food = st.columns([1, 1.2])

# ---------------- PROFILE ----------------
with col_profile:
    st.header("👤 Profile")

    name = st.text_input("Full name (optional)")

    age_opts = ["Select age"] + [str(x) for x in range(15, 81)]
    age_sel = st.selectbox("Age", options=age_opts, index=0)
    age = int(age_sel) if age_sel != "Select age" else None

    gender_opts = ["Select gender", "male", "female", "other"]
    gender_sel = st.selectbox("Gender", options=gender_opts, index=0)
    gender = gender_sel if gender_sel != "Select gender" else None

    weight_opts = ["Select weight"] + [str(x) for x in range(30, 151)]
    weight_sel = st.selectbox("Weight (kg)", options=weight_opts, index=0)
    weight = float(weight_sel) if weight_sel != "Select weight" else None

    # height dropdown single
    h_vals = height_dropdown_values(4, 7)
    height_sel = st.selectbox("Height", options=["Select height"] + h_vals, index=0)
    height_cm = height_to_cm(height_sel) if height_sel != "Select height" else None

    activity_level = st.selectbox(
        "Activity level",
        ["Select activity", "sedentary", "light", "moderate", "active", "very active"],
        index=0,
    )
    activity_level = activity_level if activity_level != "Select activity" else None

    goal_map = {"Select goal": None, "Maintain": "maintain", "Weight Loss": "weight_loss", "Weight Gain": "weight_gain"}
    goal_ui = st.selectbox("Goal", list(goal_map.keys()), index=0)
    goal = goal_map[goal_ui]

    diet_pref_opts = ["Select diet", "veg", "non-veg", "vegan"]
    diet_pref_sel = st.selectbox("Diet preference", diet_pref_opts, index=0)
    diet_pref = diet_pref_sel if diet_pref_sel != "Select diet" else None

# ---------------- FOOD LOG ----------------
with col_food:
    st.header("🍽️ Food Log")

    selected_foods = st.multiselect("Select foods", options=food_names, default=None)

    quantities = {}
    if selected_foods:
        st.markdown("**Enter grams for each selected food**")
        for food in selected_foods:
            quantities[food] = st.number_input(f"{food} (g)", min_value=1, max_value=5000, value=100)
    else:
        st.write("No foods selected yet.")

st.markdown("---")

# ---------------- ANALYZE BUTTON (Redirect only) ----------------
analyze = st.button("🔎 Analyze & View Dashboard")

if analyze:
    missing = []
    if age is None: missing.append("Age")
    if gender is None: missing.append("Gender")
    if weight is None: missing.append("Weight")
    if height_cm is None: missing.append("Height")
    if activity_level is None: missing.append("Activity level")
    if goal is None: missing.append("Goal")
    if diet_pref is None: missing.append("Diet preference")
    if not selected_foods: missing.append("Food log (select at least 1 food)")

    if missing:
        st.error("Please provide: " + ", ".join(missing))
    else:
        # Store everything in session_state for Analytics page
        st.session_state["profile"] = {
            "name": name,
            "age": age,
            "gender": gender,
            "weight": weight,
            "height_cm": height_cm,
            "activity": activity_level,
            "goal": goal,
            "goal_ui": goal_ui,
            "diet": diet_pref,
        }
        st.session_state["food_log"] = quantities

        st.success("Inputs saved! Redirecting…")
        st.balloons()

        # Navigation hack
        st.switch_page("pages/3_📊_Analytics_Dashboard.py")