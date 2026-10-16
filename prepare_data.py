
import pandas as pd
from pathlib import Path

# Load IPL datasets
matches_path = Path("matches.csv")
deliveries_path = Path("deliveries.csv")

if not matches_path.exists() or not deliveries_path.exists():
    raise FileNotFoundError(
        "Please put matches.csv and deliveries.csv "
        "in the project folder."
    )

matches = pd.read_csv(matches_path)
deliveries = pd.read_csv(deliveries_path)

print("Matches dataset:", matches.shape)
print("Deliveries dataset:", deliveries.shape)

print("\nMatch columns:")
print(matches.columns.tolist())

print("\nDelivery columns:")
print(deliveries.columns.tolist())

# Check required columns
required_matches = {"id", "winner", "team1", "team2"}
required_deliveries = {
    "match_id",
    "inning",
    "batting_team",
    "bowling_team",
    "over",
    "ball",
    "total_runs",
}

missing_matches = required_matches - set(matches.columns)
missing_deliveries = required_deliveries - set(deliveries.columns)

if missing_matches or missing_deliveries:
    print("\nDataset format needs adjustment.")
    print("Missing match columns:", missing_matches)
    print("Missing delivery columns:", missing_deliveries)
else:
    print("\nDataset column checks passed!")

    matches = matches.dropna(subset=["winner"])
    deliveries = deliveries.dropna(
        subset=["match_id", "batting_team", "bowling_team"]
    )

    matches.to_csv("clean_matches.csv", index=False)
    deliveries.to_csv("clean_deliveries.csv", index=False)

    print("Saved clean_matches.csv")
    print("Saved clean_deliveries.csv")
    print("Data preparation completed!")
