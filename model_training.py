import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib



CSV_PATH = "Book2.csv"
MODEL_PATH = "bowler_model.pkl"

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv(CSV_PATH)

# Standardise column names (removes hidden spaces)
df.columns = df.columns.str.strip()

# Drop fully empty rows (your CSV has separator rows like ",,,,,,,,,,,")
df = df.dropna(how="all").copy()

# Required columns check
required_cols = [
    "match_type", "pitch_type", "weather", "ground",
    "your_team", "opposition_team",
    "bowler_type",   # keep type
    "wickets", "runs", "economy",
    "did_bowler_do_well"
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in CSV: {missing}")

# Ensure numeric columns are numeric (bad strings -> NaN)
numeric_cols = ["wickets", "runs", "economy", "did_bowler_do_well"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Fill missing numeric values (economy sometimes missing)
df["economy"] = df["economy"].fillna(0)
df["wickets"] = df["wickets"].fillna(0)
df["runs"] = df["runs"].fillna(0)

# Drop rows where target is missing
df = df.dropna(subset=["did_bowler_do_well"]).copy()
df["did_bowler_do_well"] = df["did_bowler_do_well"].astype(int)

# ----------------------------
#  ENCODE CATEGORICALS
# ----------------------------
categorical_cols = [
    "match_type",
    "pitch_type",
    "weather",
    "ground",
    "your_team",
    "opposition_team",
    "bowler_type"
]

# Use one-hot encoding 
X_cat = pd.get_dummies(df[categorical_cols], drop_first=False)

# Add numeric features
X_num = df[["wickets", "runs", "economy"]].copy()

# Final feature matrix
X = pd.concat([X_cat, X_num], axis=1)
y = df["did_bowler_do_well"]

print("✅ Rows used for training:", len(df))
print("✅ Feature columns:", X.shape[1])
print("✅ Class balance:\n", y.value_counts())

# ----------------------------
# TRAIN / TEST SPLIT 
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ----------------------------
# TRAIN MODEL
# ----------------------------
model = RandomForestClassifier(
    n_estimators=400,
    random_state=42,
    class_weight="balanced",  # helps if 0/1 are imbalanced
    n_jobs=-1
)

model.fit(X_train, y_train)

# ----------------------------
# EVALUATION
# ----------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n✅ Model Training Complete!")
print("Accuracy:", round(acc, 4))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))

# ----------------------------
# SAVE MODEL + FEATURE COLUMNS
# ----------------------------
bundle = {
    "model": model,
    "feature_columns": X.columns.tolist()
}

joblib.dump(bundle, MODEL_PATH)
print(f"\n✅ Saved model bundle to: {MODEL_PATH}")
