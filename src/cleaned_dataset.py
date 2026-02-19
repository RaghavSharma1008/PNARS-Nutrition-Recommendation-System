import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")     
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -----------------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------------

DATA_PATH = "../data/nutrition_dataset.csv"

print("📌 Loading dataset...")
df = pd.read_csv(DATA_PATH)

print("✔ Dataset Loaded Successfully!")
print("\n--- HEAD ---")
print(df.head())

print("\n--- INFO ---")
print(df.info())

print("\n--- NULL VALUES ---")
print(df.isnull().sum())


# -----------------------------------------------------------
# 2. STANDARDIZE COLUMN NAMES & DATA TYPES
# -----------------------------------------------------------

print("\n📌 Standardizing column formats...")

# Clean column names
df.columns = df.columns.str.strip().str.lower()

# Correct column naming (ensures uniform naming)
df = df.rename(columns={
    "food_id": "food_id",
    "food_name": "food_name",
    "category": "category",
    "calories": "calories",
    "carbs": "carbs",
    "protein": "protein",
    "fat": "fat",
    "fibre": "fibre",
    "sugar": "sugar",
    "sodium": "sodium",
    "vitamin_c": "vitamin_c",
    "iron": "iron",
    "calcium": "calcium",
    "diet_type": "diet_type"
})

# Format text-based columns
df["diet_type"] = df["diet_type"].astype(str).str.title().str.strip()
df["category"] = df["category"].astype(str).str.title().str.strip()

# Numeric columns
numeric_cols = [
    "calories", "carbs", "protein", "fat", "fibre", "sugar",
    "sodium", "vitamin_c", "iron", "calcium"
]

df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

print("\n--- AFTER TYPE FIXING ---")
print(df.info())


# -----------------------------------------------------------
# 3. HANDLE MISSING VALUES
# -----------------------------------------------------------

print("\n📌 Handling Missing Values...")

# Fill numeric missing values with median
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Fix blanks in diet_type
df["diet_type"] = df["diet_type"].replace("", np.nan)
df["diet_type"] = df["diet_type"].fillna("Unknown")

print("✔ Missing values fixed!")
print(df.isnull().sum())


# -----------------------------------------------------------
# 4. REMOVE OUTLIERS
# -----------------------------------------------------------

print("\n📌 Removing unrealistic outliers...")

df = df[(df["calories"] > 0) & (df["calories"] < 1500)]
df = df[df["sodium"] >= 0]
df = df[df["protein"] >= 0]

print("✔ Outliers removed!")
print(f"📊 Total Rows After Cleaning: {len(df)}")


# -----------------------------------------------------------
# 5. BASIC EDA VISUALIZATIONS (Saved as Images)
# -----------------------------------------------------------

print("\n📌 Generating EDA visualizations...")

EDA_DIR = "../data/eda_charts"

if not os.path.exists(EDA_DIR):
    os.makedirs(EDA_DIR)

# 1. Calorie Distribution
plt.figure(figsize=(8, 5))
plt.hist(df["calories"], bins=40)
plt.title("Calorie Distribution")
plt.xlabel("Calories")
plt.ylabel("Frequency")
plt.savefig(f"{EDA_DIR}/calorie_distribution.png")
plt.close()


# 2. Protein vs Calories
plt.figure(figsize=(7, 5))
plt.scatter(df["protein"], df["calories"])
plt.title("Protein vs Calories")
plt.xlabel("Protein (g)")
plt.ylabel("Calories")
plt.savefig(f"{EDA_DIR}/protein_vs_calories.png")
plt.close()


# 3. Average calories by category
plt.figure(figsize=(10, 6))
df.groupby("category")["calories"].mean().plot(kind="bar")
plt.title("Average Calories by Category")
plt.ylabel("Calories")
plt.savefig(f"{EDA_DIR}/calories_by_category.png")
plt.close()


# 4. Correlation heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig(f"{EDA_DIR}/correlation_heatmap.png")
plt.close()

print("✔ All charts saved in folder:", EDA_DIR)


# -----------------------------------------------------------
# 6. SAVE CLEANED DATASET
# -----------------------------------------------------------

CLEAN_PATH = "../data/cleaned_nutrition_dataset.csv"
df.to_csv(CLEAN_PATH, index=False)

print("\n🎉 PHASE 3 COMPLETED SUCCESSFULLY!")
print(f"📁 Cleaned dataset saved at: {CLEAN_PATH}")
print("📊 EDA charts saved at:", EDA_DIR)
