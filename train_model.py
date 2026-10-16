
import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load cleaned datasets
matches = pd.read_csv("clean_matches.csv")
deliveries = pd.read_csv("clean_deliveries.csv")

# IPL teams used in the model
teams = [
    "Sunrisers Hyderabad",
    "Mumbai Indians",
    "Royal Challengers Bangalore",
    "Kolkata Knight Riders",
    "Kings XI Punjab",
    "Chennai Super Kings",
    "Rajasthan Royals",
    "Delhi Daredevils",
    "Gujarat Lions",
    "Rising Pune Supergiant",
    "Delhi Capitals",
    "Punjab Kings",
    "Lucknow Super Giants",
    "Gujarat Titans",
    "Royal Challengers Bengaluru",
]

# Retain matches with known winners and supported teams
matches = matches.dropna(subset=["winner", "team1", "team2"])
matches = matches[
    matches["team1"].isin(teams)
    & matches["team2"].isin(teams)
    & matches["winner"].isin(teams)
].copy()

matches["team1_win"] = (
    matches["winner"] == matches["team1"]
).astype(int)

# One row per match
data = matches[
    ["team1", "team2", "city", "team1_win"]
].copy()

data["city"] = data["city"].fillna("Unknown")

# Input features and target
X = data[["team1", "team2", "city"]]
y = data["team1_win"]

if len(data) < 50 or y.nunique() < 2:
    raise ValueError(
        "Not enough valid matches to train the model. "
        "Check the team names in matches.csv."
    )

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Encode team names and cities
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            ["team1", "team2", "city"],
        )
    ]
)

# Build model pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced",
            ),
        ),
    ]
)

# Train and evaluate
model.fit(X_train, y_train)
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("IPL WIN PREDICTOR")
print("-" * 40)
print("Matches used:", len(data))
print("Training matches:", len(X_train))
print("Testing matches:", len(X_test))
print(f"Test accuracy: {accuracy:.2%}")
print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0,
    )
)

# Save trained model
with open("ipl_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel saved successfully as ipl_model.pkl")
