from recommendation_engine import generate_recommendations

out = generate_recommendations(
    age=25, gender="male", weight=70, height=175,
    activity_level="moderate", goal="weight_loss",
    calorie_intake_today=1800, diet_preference="veg"
)

print(out["calorie_target"])
print(out["macro_targets"])
print(out["tips"])
print(out["recommended_foods"])
