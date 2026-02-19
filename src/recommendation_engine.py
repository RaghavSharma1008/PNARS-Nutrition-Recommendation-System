# src/recommendation_engine.py
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os

# -----------------------------------------
# FIX PYARROW STRICT MODE (Universal Fix)
# -----------------------------------------
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"


# -------------------------------------------------------------
# LOAD CLEANED DATA
# -------------------------------------------------------------
def load_food_data(path="data/cleaned_nutrition_dataset.csv"):
    df = pd.read_csv(path)

    # Ensure consistent column names
    df.columns = [c.strip().lower() for c in df.columns]

    # fallback diet type if missing
    if "diet_type" not in df.columns:
        df["diet_type"] = "veg"

    # normalize diet_type strings for robust filtering later
    df["diet_type"] = df["diet_type"].astype(str).str.lower().str.replace("-", " ").str.replace("_", " ").str.strip()

    # ensure food_name exists
    if "food_name" not in df.columns:
        df["food_name"] = df.index.astype(str)

    return df


# -------------------------------------------------------------
# DAILY CALORIE TARGET
# -------------------------------------------------------------
def calculate_daily_calorie_target(age, gender, weight, height, activity_level, goal):
    if gender.lower() == "male":
        bmr = (10 * weight) + (6.25 * height) - (5 * age) + 5
    else:
        bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161

    activity_factors = {
        "sedentary": 1.2,
        "light": 1.375,
        "moderate": 1.55,
        "active": 1.725,
        "very active": 1.9
    }

    maintenance = bmr * activity_factors.get(activity_level.lower(), 1.2)

    if goal == "weight_loss":
        return round(maintenance - 300)
    elif goal == "weight_gain":
        return round(maintenance + 300)
    return round(maintenance)


# -------------------------------------------------------------
# MACRO TARGETS
# -------------------------------------------------------------
def get_macro_targets(calorie_target):
    protein = 0.30 * calorie_target / 4
    carbs = 0.45 * calorie_target / 4
    fats = 0.25 * calorie_target / 9

    return {
        "protein_g": round(protein),
        "carbs_g": round(carbs),
        "fats_g": round(fats)
    }


# -------------------------------------------------------------
# RULE-BASED TIPS
# -------------------------------------------------------------
def rule_based_tips(intake, target):
    diff = intake - target
    tips = []

    if diff > 200:
        tips.append("You're in a calorie surplus — reduce refined carbs & sugary foods.")
    elif diff < -200:
        tips.append("You're in a calorie deficit — add nutrient-dense foods like nuts & paneer.")

    tips.append("Drink 2–3 liters of water daily.")
    tips.append("Add at least 20–30g protein to each meal.")
    tips.append("Prefer whole grains over refined carbs.")

    return tips


# -------------------------------------------------------------
# CONTENT-BASED FOOD RECOMMENDER
# -------------------------------------------------------------
def recommend_foods(user_preference, goal, df):
    # work on a copy to avoid surprising side-effects
    df = df.copy()

    features = ["calories", "protein", "carbs", "fat"]

    # fallback if any column missing
    for col in features:
        if col not in df.columns:
            df[col] = 0

    # fillna and prepare nutrient matrix
    nutrient_matrix = df[features].fillna(0).values

    # SMART GOAL VECTORS (more realistic)
    if goal == "weight_loss":
        user_vector = np.array([[180, 30, 25, 6]])
    elif goal == "weight_gain":
        user_vector = np.array([[450, 25, 60, 12]])
    else:
        user_vector = np.array([[300, 22, 35, 8]])

    # compute cosine similarity (handle tiny datasets gracefully)
    try:
        sim = cosine_similarity(user_vector, nutrient_matrix)[0]
    except Exception:
        # fallback: similarity zeros
        sim = np.zeros(len(df))

    df["similarity"] = sim

    # --- Robust diet filtering ---
    # ensure diet_type column exists & normalized (load_food_data normally does this)
    if "diet_type" not in df.columns:
        df["diet_type"] = "veg"
    df["diet_type"] = df["diet_type"].astype(str).str.lower().str.replace("-", " ").str.replace("_", " ").str.strip()

    if user_preference == "veg":
        # include vegetarian and vegan entries, exclude explicit non-veg labels
        df = df[~df["diet_type"].str.contains(r"\b(non|non ?veg|non ?vegetarian|non ?vegetarian)\b", na=False)]
    elif user_preference == "vegan":
        # include only entries that explicitly mention vegan
        df = df[df["diet_type"].str.contains(r"\bvegan\b", na=False)]
    elif user_preference == "non-veg":
        # keep all entries
        pass

    # Weighted score (avoid division by zero)
    # convert calories/protein to numeric safe
    df["calories"] = pd.to_numeric(df["calories"].fillna(0), errors="coerce").fillna(0)
    df["protein"] = pd.to_numeric(df["protein"].fillna(0), errors="coerce").fillna(0)

    df["score"] = (
        df["similarity"] * 0.60 +
        (df["protein"] / (df["calories"] + 1)) * 0.40
    )

    # final safe sort and return top N (keep relevant columns)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    keep_cols = [c for c in ["food_name", "calories", "protein", "carbs", "fat", "diet_type", "score"] if c in df.columns]
    return df[keep_cols].head(60)


# -------------------------------------------------------------
# INDIAN MEAL HEURISTIC KEYWORDS
# -------------------------------------------------------------
BREAKFAST_INDIAN = [
    "poha", "upma", "idli", "dosa", "uttapam", "dalia", "porridge",
    "omelette", "eggs", "sprouts", "oats", "paratha", "chai", "milk",
    "curd", "paneer", "moong dal", "chilla"
]

LUNCH_INDIAN = [
    "roti", "rice", "dal", "sabzi", "rajma", "chole", "paneer",
    "chicken", "fish", "sambar", "curd", "salad", "khichdi"
]

DINNER_INDIAN = [
    "khichdi", "dalia", "roti", "dal", "paneer", "chicken",
    "veg soup", "spinach", "light sabzi"
]

SNACKS_INDIAN = [
    "banana", "apple", "almonds", "peanuts", "sprouts",
    "yogurt", "tea", "coffee", "fruit"
]


def best_match(keywords, foods):
    matches = []
    for f in foods:
        fname = str(f).lower()
        if any(k in fname for k in keywords):
            matches.append(f)
    return matches


# -------------------------------------------------------------
# INDIAN MEAL PLAN GENERATOR
# -------------------------------------------------------------
def generate_meal_plan(rec_df, diet, goal):
    foods = rec_df["food_name"].astype(str).tolist()

    breakfast = best_match(BREAKFAST_INDIAN, foods)
    lunch = best_match(LUNCH_INDIAN, foods)
    dinner = best_match(DINNER_INDIAN, foods)
    snacks = best_match(SNACKS_INDIAN, foods)

    # If empty, fallback to top-ranked foods
    if not breakfast:
        breakfast = foods[:3]
    if not lunch:
        lunch = foods[3:6]
    if not dinner:
        dinner = foods[6:9]
    if not snacks:
        snacks = foods[9:12]

    # Diet filtering (string-based, defensive)
    if diet == "veg":
        avoid = ["chicken", "fish", "egg", "meat"]
        breakfast = [f for f in breakfast if not any(a in f.lower() for a in avoid)]
        lunch = [f for f in lunch if not any(a in f.lower() for a in avoid)]
        dinner = [f for f in dinner if not any(a in f.lower() for a in avoid)]

    if diet == "vegan":
        avoid = ["milk", "curd", "paneer", "butter", "cheese", "yogurt", "lassi", "ghee"]
        breakfast = [f for f in breakfast if not any(a in f.lower() for a in avoid)]
        lunch = [f for f in lunch if not any(a in f.lower() for a in avoid)]
        dinner = [f for f in dinner if not any(a in f.lower() for a in avoid)]
        snacks = [f for f in snacks if not any(a in f.lower() for a in avoid)]

    # Ensure lists are not empty — fallback to top-ranked
    def ensure_list(lst, start_idx):
        if lst:
            return lst[:3]
        else:
            return foods[start_idx:start_idx + 3]

    return {
        "breakfast": ensure_list(breakfast, 0),
        "lunch": ensure_list(lunch, 3),
        "dinner": ensure_list(dinner, 6),
        "snacks": ensure_list(snacks, 9)
    }


# -------------------------------------------------------------
# GOAL ALIGNMENT SCORE (Beautiful logic)
# -------------------------------------------------------------
def compute_goal_alignment_score(intake, target, protein_gap):
    score = 100

    # Penalize calorie mismatch
    score -= min(abs(intake - target) / 20, 35)

    # Penalize protein shortfall
    score -= min(protein_gap / 4, 35)

    return int(max(5, min(score, 100)))


# -------------------------------------------------------------
# MASTER ENGINE
# -------------------------------------------------------------
def generate_recommendations(age, gender, weight, height, activity_level,
                             goal, calorie_intake_today, diet_preference):

    df = load_food_data()

    calorie_target = calculate_daily_calorie_target(age, gender, weight, height, activity_level, goal)
    macro_targets = get_macro_targets(calorie_target)

    rec_df = recommend_foods(diet_preference, goal, df)

    # If recommend_foods returned empty, fallback to top dataset foods (avoid empty meal plans)
    if rec_df is None or rec_df.empty:
        fallback_df = df.copy()
        # pick top 30 by a simple heuristic: high protein/calorie ratio
        fallback_df["ratio"] = fallback_df.apply(lambda r: (float(r.get("protein", 0)) / (float(r.get("calories", 1)) + 1)), axis=1)
        rec_df = fallback_df.sort_values("ratio", ascending=False).head(60)[[
            c for c in ["food_name", "calories", "protein", "carbs", "fat", "diet_type"] if c in fallback_df.columns
        ]].reset_index(drop=True)

    meal_plan = generate_meal_plan(rec_df, diet_preference, goal)

    # compute protein mean safely
    prot_mean = 0.0
    if "protein" in rec_df.columns and not rec_df["protein"].dropna().empty:
        prot_mean = float(rec_df["protein"].head(10).dropna().mean())

    protein_gap = max(macro_targets["protein_g"] - prot_mean, 0)
    goal_score = compute_goal_alignment_score(calorie_intake_today, calorie_target, protein_gap)

    insights = {
        "calorie_diff": round(calorie_intake_today - calorie_target, 1),
        "macro_targets": macro_targets,
        "protein_gap": round(protein_gap, 1),
        "summary_line": f"{macro_targets['protein_g']}g protein, {macro_targets['carbs_g']}g carbs, {macro_targets['fats_g']}g fats"
    }

    return {
        "calorie_target": calorie_target,
        "macro_targets": macro_targets,
        "tips": rule_based_tips(calorie_intake_today, calorie_target),
        "recommended_foods": rec_df,
        "meal_plan": meal_plan,
        "goal_alignment_score": goal_score,
        "insights": insights
    }
