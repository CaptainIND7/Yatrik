from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import joblib
import pandas as pd

app = FastAPI(title="Yatrik API", version="1.0.0")

# ----------------------------
# FILE PATHS
# ----------------------------
MODEL_PATH = "rf_tourist_model.pkl"
DATA_PATH = "tourist_places.csv"

# ----------------------------
# MODEL / DATA CONFIG
# ----------------------------
FEATURES = ["Lat", "Lng", "Ideal_Hours", "Popularity_Score", "City_enc"]

TARGETS = [
    "Is_Museum",
    "Is_Nature",
    "Is_Beach",
    "Is_History",
    "Is_Temple",
    "Is_Wildlife",
    "Is_Shopping",
    "Is_Foodie",
]

DEFAULT_HOURS_PER_DAY = 8

# ----------------------------
# LOAD MODEL
# ----------------------------
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model file '{MODEL_PATH}': {e}")

# ----------------------------
# LOAD DATASET
# ----------------------------
try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load dataset file '{DATA_PATH}': {e}")

required_columns = {
    "State",
    "City",
    "Place_Name",
    "Lat",
    "Lng",
    "Ideal_Hours",
    "Popularity_Score",
    "City_enc",
    *TARGETS,
}

missing_columns = required_columns - set(df.columns)
if missing_columns:
    raise RuntimeError(
        f"Dataset missing required columns: {sorted(missing_columns)}"
    )


# ----------------------------
# REQUEST MODEL
# ----------------------------
class TripRequest(BaseModel):
    place: str = Field(..., example="Raipur")
    days: int = Field(..., ge=1, le=30, example=2)
    preferences: List[str] = Field(..., example=["Is_Nature", "Is_History"])
    hours_per_day: int = Field(DEFAULT_HOURS_PER_DAY, ge=1, le=24, example=8)


# ----------------------------
# HELPERS
# ----------------------------
def validate_preferences(preferences: List[str]) -> List[str]:
    valid_preferences = [p for p in preferences if p in TARGETS]
    return valid_preferences


def get_filtered_dataframe(user_input: str) -> tuple[pd.DataFrame, str]:
    if user_input in df["State"].unique():
        return df[df["State"] == user_input].copy(), "state"
    if user_input in df["City"].unique():
        return df[df["City"] == user_input].copy(), "city"
    return pd.DataFrame(), "none"


def score_places(city_df: pd.DataFrame, user_preferences: List[str]) -> pd.DataFrame:
    if city_df.empty:
        return city_df

    X_city = city_df[FEATURES]

    try:
        probas = model.predict_proba(X_city)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    city_df = city_df.copy()
    city_df["Match_Score"] = 0.0

    for pref in user_preferences:
        idx = TARGETS.index(pref)
        city_df["Match_Score"] += [p[1] for p in probas[idx]]

    recommendations = city_df.sort_values(
        by=["Match_Score", "Popularity_Score"],
        ascending=False,
    )

    return recommendations


def build_itinerary(
    recommendations: pd.DataFrame,
    user_duration_days: int,
    hours_per_day: int,
) -> tuple[List[dict], int, bool]:
    itinerary = []
    current_day = 1
    remaining_hours = hours_per_day
    spots_added = 0
    exceeded_duration = False

    for _, spot in recommendations.iterrows():
        spot_time = float(spot["Ideal_Hours"])

        if spot_time > hours_per_day:
            continue

        if remaining_hours < spot_time:
            current_day += 1
            remaining_hours = hours_per_day

            if current_day > user_duration_days:
                exceeded_duration = True
                break

        actual_tags = [
            target.replace("Is_", "")
            for target in TARGETS
            if int(spot[target]) == 1
        ]

        itinerary.append(
            {
                "day": int(current_day),
                "place_name": str(spot["Place_Name"]),
                "city": str(spot["City"]),
                "state": str(spot["State"]),
                "ideal_hours": float(spot["Ideal_Hours"]),
                "popularity_score": float(spot["Popularity_Score"]),
                "match_score": float(spot["Match_Score"]),
                "categories": actual_tags,
                "lat": float(spot["Lat"]),
                "lng": float(spot["Lng"]),
            }
        )

        remaining_hours -= spot_time
        spots_added += 1

    return itinerary, spots_added, exceeded_duration


# ----------------------------
# ROUTES
# ----------------------------
@app.get("/")
def root():
    return {"message": "Yatrik backend is running"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "dataset_rows": int(len(df)),
    }


@app.get("/targets")
def get_targets():
    return {"targets": TARGETS}


@app.post("/predict")
def predict_trip(request: TripRequest):
    user_input = request.place.strip()
    user_duration_days = request.days
    user_preferences = validate_preferences(request.preferences)
    hours_per_day = request.hours_per_day

    if not user_input:
        raise HTTPException(status_code=400, detail="Place cannot be empty.")

    if not user_preferences:
        raise HTTPException(
            status_code=400,
            detail=f"No valid preferences provided. Valid preferences: {TARGETS}",
        )

    city_df, mode = get_filtered_dataframe(user_input)

    if city_df.empty:
        return {
            "success": False,
            "message": f"No data found for: {user_input}",
            "place": user_input,
            "days": user_duration_days,
            "hours_per_day": hours_per_day,
            "preferences": user_preferences,
            "itinerary": [],
        }

    recommendations = score_places(city_df, user_preferences)

    itinerary, spots_added, exceeded_duration = build_itinerary(
        recommendations=recommendations,
        user_duration_days=user_duration_days,
        hours_per_day=hours_per_day,
    )

    if spots_added == 0:
        return {
            "success": False,
            "message": "No spots found matching your constraints.",
            "mode": mode,
            "place": user_input,
            "days": user_duration_days,
            "hours_per_day": hours_per_day,
            "preferences": user_preferences,
            "spots_found": int(len(city_df)),
            "itinerary": [],
        }

    response = {
        "success": True,
        "mode": mode,
        "place": user_input,
        "days": user_duration_days,
        "hours_per_day": hours_per_day,
        "preferences": user_preferences,
        "spots_found": int(len(city_df)),
        "spots_added": int(spots_added),
        "itinerary": itinerary,
    }

    if exceeded_duration:
        response["note"] = "Some spots were omitted as they exceeded the trip duration."

    return response
