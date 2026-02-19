import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pickle
import os

# --------------------------------------------------------
# 1. LOAD CLEANED DATASET
# --------------------------------------------------------

DATA_PATH = "../data/cleaned_nutrition_dataset.csv"

print("📌 Loading cleaned dataset...")

df = pd.read_csv(DATA_PATH)
print("✔ Dataset Loaded Successfully!")
print(df.head())


# --------------------------------------------------------
# 2. MACHINE LEARNING MODEL 1:
#    CALORIE BALANCE PREDICTOR (Linear Regression)
# --------------------------------------------------------

print("\n📌 Building Calorie Balance Predictor (ML-1)...")

# Synthetic training dataset for calorie balance
# (This model predicts weekly calorie surplus/deficit based on lifestyle)
data = {
    "weight": [50, 60, 70, 80, 90, 100],
    "activity": [1, 2, 3, 4, 5, 3],  # 1 = sedentary, 5 = athlete
    "calorie_intake": [1500, 1800, 2200, 2500, 2800, 3200],
    "expected_change": [-400, -200, 0, 200, 350, 500]
}

train_df = pd.DataFrame(data)

X = train_df[["weight", "activity", "calorie_intake"]]
y = train_df["expected_change"]

calorie_model = LinearRegression()
calorie_model.fit(X, y)

print("✔ ML-1 Model Trained!")


# --------------------------------------------------------
# 3. MACHINE LEARNING MODEL 2:
#    MEAL CLUSTERING (KMeans)
# --------------------------------------------------------

print("\n📌 Building Meal Clustering Model (ML-2)...")

cluster_features = df[["calories", "protein", "carbs", "fat"]]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(cluster_features)

kmeans = KMeans(n_clusters=3, random_state=42)
df["cluster"] = kmeans.fit_predict(scaled_features)

cluster_labels = {
    0: "High Protein Meals",
    1: "High Carb Meals",
    2: "Balanced Meals"
}

df["cluster_name"] = df["cluster"].map(cluster_labels)

print("✔ ML-2 Clustering Completed!")
print(df[["food_name", "cluster_name"]].head())


# --------------------------------------------------------
# 4. MACHINE LEARNING MODEL 3:
#    CONTENT-BASED RECOMMENDER
# --------------------------------------------------------

print("\n📌 Building Recommendation Model (ML-3)...")

recommend_features = df[["protein", "carbs", "fat", "calories"]]

rec_scaler = StandardScaler()
scaled_rec = rec_scaler.fit_transform(recommend_features)

similarity_matrix = np.dot(scaled_rec, scaled_rec.T)

print("✔ ML-3 Recommendation System Ready!")


# --------------------------------------------------------
# 5. SAVE ALL MODELS
# --------------------------------------------------------

MODEL_DIR = "../models"

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

pickle.dump(calorie_model, open(f"{MODEL_DIR}/calorie_balance_model.pkl", "wb"))
pickle.dump(kmeans, open(f"{MODEL_DIR}/meal_cluster_model.pkl", "wb"))
pickle.dump(scaler, open(f"{MODEL_DIR}/cluster_scaler.pkl", "wb"))
pickle.dump(similarity_matrix, open(f"{MODEL_DIR}/similarity_matrix.pkl", "wb"))
pickle.dump(rec_scaler, open(f"{MODEL_DIR}/recommend_scaler.pkl", "wb"))
df.to_csv("../data/dataset_with_clusters.csv", index=False)

print("\n🎉 PHASE 4 COMPLETED SUCCESSFULLY!")
print("📁 Saved Models in: /models/")
print("📊 Saved cluster-enhanced dataset in: /data/dataset_with_clusters.csv")
