from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import joblib
import pandas as pd
import os
import requests

app = FastAPI(title="Yatrik API", version="1.0.0")

# ----------------------------
# CORS CONFIG
# ----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# FILE PATHS
# ----------------------------
MODEL_PATH = "rf_tourist_model.pkl"
DATA_PATH = "chhattisgarh_tourist_places.csv"

# ----------------------------
# MODEL / DATA CONFIG
# ----------------------------
DEFAULT_FEATURES = ["Lat", "Lng", "Ideal_Hours", "Popularity_Score", "City_enc"]

TARGETS = [
    "Is_Museum",
    "Is_Nature",
    "Is_Beach",
    "Is_History",
    "Is_Temple",
    "Is_Wildlife",
    "Is_Shopping",
]

DEFAULT_HOURS_PER_DAY = 14

# ----------------------------
# IMAGE CONFIG
# ----------------------------
IMAGE_REQUEST_TIMEOUT = 5
PEXELS_IMAGES_PER_SEARCH = 8

# ----------------------------
# LOAD MODEL
# ----------------------------
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model file '{MODEL_PATH}': {e}")

# ----------------------------
# GET EXACT MODEL FEATURE ORDER
# ----------------------------
try:
    FEATURES = list(model.feature_names_in_)
except Exception:
    FEATURES = DEFAULT_FEATURES

# ----------------------------
# LOAD DATASET
# ----------------------------
try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load dataset file '{DATA_PATH}': {e}")

# ----------------------------
# CREATE City_enc IF MISSING
# ----------------------------
if "City_enc" not in df.columns:
    df["City_enc"] = pd.factorize(df["City"].astype(str))[0]

required_columns = {
    "State",
    "City",
    "Place_Name",
    "Lat",
    "Lng",
    "Ideal_Hours",
    "Popularity_Score",
    *FEATURES,
    *TARGETS,
}

missing_columns = required_columns - set(df.columns)
if missing_columns:
    raise RuntimeError(
        f"Dataset missing required columns: {sorted(missing_columns)}"
    )

# ----------------------------
# REQUEST MODELS
# ----------------------------
class TripRequest(BaseModel):
    place: str = Field(..., example="Raipur")
    days: int = Field(..., ge=1, le=30, example=2)
    preferences: List[str] = Field(..., example=["Is_Nature", "Is_History"])
    hours_per_day: int = Field(DEFAULT_HOURS_PER_DAY, ge=1, le=24, example=8)


class FlutterTripRequest(BaseModel):
    City: str = Field(..., example="Raipur")
    State: str = Field("", example="Chhattisgarh")
    Days: int = Field(..., ge=1, le=30, example=2)

    Is_Museum: int = 0
    Is_Nature: int = 0
    Is_Beach: int = 0
    Is_History: int = 0
    Is_Temple: int = 0
    Is_Wildlife: int = 0
    Is_Shopping: int = 0
    Is_Foodie: int = 0

    hours_per_day: int = Field(DEFAULT_HOURS_PER_DAY, ge=1, le=24, example=8)


# ----------------------------
# HELPERS
# ----------------------------
def validate_preferences(preferences: List[str]) -> List[str]:
    return [p for p in preferences if p in TARGETS]


def get_filtered_dataframe(user_input: str):
    user_input = user_input.strip()

    state_match = df[df["State"].astype(str).str.lower() == user_input.lower()]
    if not state_match.empty:
        return state_match.copy(), "state"

    city_match = df[df["City"].astype(str).str.lower() == user_input.lower()]
    if not city_match.empty:
        return city_match.copy(), "city"

    return pd.DataFrame(), "none"


def flutter_request_to_trip_request(request: FlutterTripRequest) -> TripRequest:
    preferences = []

    if request.Is_Museum == 1:
        preferences.append("Is_Museum")

    if request.Is_Nature == 1:
        preferences.append("Is_Nature")

    if request.Is_Beach == 1:
        preferences.append("Is_Beach")

    if request.Is_History == 1:
        preferences.append("Is_History")

    if request.Is_Temple == 1:
        preferences.append("Is_Temple")

    if request.Is_Wildlife == 1:
        preferences.append("Is_Wildlife")

    if request.Is_Shopping == 1:
        preferences.append("Is_Shopping")

    # Is_Foodie is ignored for now because TARGETS does not contain Is_Foodie.

    place_to_search = request.City.strip()

    if not place_to_search:
        place_to_search = request.State.strip()

    return TripRequest(
        place=place_to_search,
        days=request.Days,
        preferences=preferences,
        hours_per_day=request.hours_per_day,
    )


def get_wikimedia_image_url(place_name: str, city: str = "") -> str:
    query = f"{place_name} {city}".strip()

    try:
        api_url = "https://en.wikipedia.org/w/api.php"

        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": 1,
        }

        search_response = requests.get(
            api_url,
            params=search_params,
            timeout=IMAGE_REQUEST_TIMEOUT,
        )
        search_response.raise_for_status()
        search_data = search_response.json()

        search_results = search_data.get("query", {}).get("search", [])

        if not search_results:
            return ""

        page_title = search_results[0].get("title", "")

        if not page_title:
            return ""

        image_params = {
            "action": "query",
            "prop": "pageimages",
            "titles": page_title,
            "format": "json",
            "pithumbsize": 600,
        }

        image_response = requests.get(
            api_url,
            params=image_params,
            timeout=IMAGE_REQUEST_TIMEOUT,
        )
        image_response.raise_for_status()
        image_data = image_response.json()

        pages = image_data.get("query", {}).get("pages", {})

        for _, page in pages.items():
            thumbnail = page.get("thumbnail", {})
            image_url = thumbnail.get("source", "")
            if image_url:
                return image_url

        return ""

    except Exception:
        return ""


def get_pexels_image_urls(place_name: str, city: str = "") -> List[str]:
    api_key = os.getenv("PEXELS_API_KEY", "").strip()

    if not api_key:
        return []

    query = f"{place_name} {city} Chhattisgarh India tourist place".strip()

    try:
        api_url = "https://api.pexels.com/v1/search"

        headers = {
            "Authorization": api_key,
        }

        params = {
            "query": query,
            "per_page": PEXELS_IMAGES_PER_SEARCH,
            "orientation": "landscape",
        }

        response = requests.get(
            api_url,
            headers=headers,
            params=params,
            timeout=IMAGE_REQUEST_TIMEOUT,
        )
        response.raise_for_status()

        data = response.json()
        photos = data.get("photos", [])
        image_urls = []

        for photo in photos:
            src = photo.get("src", {})
            image_url = (
                src.get("medium")
                or src.get("large")
                or src.get("original")
                or ""
            )

            if image_url:
                image_urls.append(image_url)

        return image_urls

    except Exception:
        return []


def get_pexels_image_url(place_name: str, city: str = "") -> str:
    image_urls = get_pexels_image_urls(place_name, city)

    if image_urls:
        return image_urls[0]

    return ""


def get_spot_image_url(place_name: str, city: str = "") -> str:
    image_url = get_wikimedia_image_url(place_name, city)

    if image_url:
        return image_url

    image_url = get_pexels_image_url(place_name, city)

    if image_url:
        return image_url

    return ""


def get_unique_spot_image_url(
    place_name: str,
    city: str = "",
    used_image_urls: set | None = None,
) -> str:
    if used_image_urls is None:
        used_image_urls = set()

    wikimedia_url = get_wikimedia_image_url(place_name, city)

    if wikimedia_url and wikimedia_url not in used_image_urls:
        used_image_urls.add(wikimedia_url)
        return wikimedia_url

    pexels_urls = get_pexels_image_urls(place_name, city)

    for image_url in pexels_urls:
        if image_url and image_url not in used_image_urls:
            used_image_urls.add(image_url)
            return image_url

    return ""


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
):
    itinerary = []
    current_day = 1
    remaining_hours = hours_per_day
    spots_added = 0
    exceeded_duration = False
    used_image_urls = set()

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

        place_name = str(spot["Place_Name"])
        city_name = str(spot["City"])
        state_name = str(spot["State"])

        image_url = get_unique_spot_image_url(
            place_name=place_name,
            city=city_name,
            used_image_urls=used_image_urls,
        )

        itinerary.append(
            {
                "day": int(current_day),
                "place_name": place_name,
                "name": place_name,
                "city": city_name,
                "state": state_name,
                "description": f"{city_name}, {state_name}",
                "ideal_hours": float(spot["Ideal_Hours"]),
                "popularity_score": float(spot["Popularity_Score"]),
                "match_score": float(spot["Match_Score"]),
                "categories": actual_tags,
                "lat": float(spot["Lat"]),
                "lng": float(spot["Lng"]),
                "image_url": image_url,
            }
        )

        remaining_hours -= spot_time
        spots_added += 1

    return itinerary, spots_added, exceeded_duration


def generate_trip_response(request: TripRequest):
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
            "recommendations": [],
            "spots": [],
            "places": [],
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
            "recommendations": [],
            "spots": [],
            "places": [],
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

        # Old key kept.
        "itinerary": itinerary,

        # New keys for Flutter.
        "recommendations": itinerary,
        "spots": itinerary,
        "places": itinerary,
    }

    if exceeded_duration:
        response["note"] = "Some spots were omitted as they exceeded the trip duration."

    return response


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
        "features_used": FEATURES,
    }


@app.get("/targets")
def get_targets():
    return {"targets": TARGETS}


@app.post("/predict")
def predict_trip(request: TripRequest):
    return generate_trip_response(request)


@app.post("/recommend")
def recommend_trip(request: FlutterTripRequest):
    converted_request = flutter_request_to_trip_request(request)
    return generate_trip_response(converted_request)
