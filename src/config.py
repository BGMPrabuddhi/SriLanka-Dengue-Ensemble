"""Project-wide configuration: paths, constants and district coordinates.

Every other module imports from here so that a single change (e.g. the split
years) propagates to notebooks, training code and the dashboard.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
# Resolve paths relative to this file so code works from any working dir
# (notebooks/, app/, repo root, Streamlit Cloud).
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"

DENGUE_RAW_CSV = DATA_RAW / "dengue.csv"
WEATHER_RAW_CSV = DATA_RAW / "weather.csv"

# ---------------------------------------------------------------- reproducibility
RANDOM_STATE = 42

# ---------------------------------------------------------------- chronological split
# Train <= TRAIN_END_YEAR, validation == VAL_YEAR, test >= TEST_START_YEAR.
# Never shuffle: the model must only learn from the past.
TRAIN_END_YEAR = 2021
VAL_YEAR = 2022
TEST_START_YEAR = 2023

# ---------------------------------------------------------------- features
TARGET = "target"           # cases in week t+1 (what we forecast)
N_LAGS = 4                  # cases_lag1 .. cases_lag4
WEATHER_VARS = ["rain_mm", "temp_c", "humidity_pct"]
WEATHER_LAGS = [1, 2, 3, 4]

# Risk levels: percentiles of each district's TRAINING-period weekly cases.
RISK_PERCENTILES = {"medium": 50, "high": 75}

# ---------------------------------------------------------------- districts
# Approximate district-centre (main town) coordinates, used only to query
# Open-Meteo weather. Kalmunai RDHS lies inside Ampara district, so it reuses
# Ampara's coordinates (as instructed in the project brief).
DISTRICT_COORDS = {
    "Colombo": (6.9271, 79.8612),
    "Gampaha": (7.0840, 80.0098),
    "Kalutara": (6.5854, 79.9607),
    "Kandy": (7.2906, 80.6337),
    "Matale": (7.4675, 80.6234),
    "Nuwara Eliya": (6.9497, 80.7891),
    "Galle": (6.0535, 80.2210),
    "Matara": (5.9549, 80.5550),
    "Hambantota": (6.1241, 81.1185),
    "Jaffna": (9.6615, 80.0255),
    "Kilinochchi": (9.3803, 80.3770),
    "Mannar": (8.9810, 79.9044),
    "Vavuniya": (8.7514, 80.4971),
    "Mullaitivu": (9.2671, 80.8142),
    "Batticaloa": (7.7310, 81.6747),
    "Ampara": (7.2975, 81.6820),
    "Kalmunai": (7.2975, 81.6820),  # mapped to Ampara
    "Trincomalee": (8.5874, 81.2152),
    "Kurunegala": (7.4863, 80.3647),
    "Puttalam": (8.0362, 79.8283),
    "Anuradhapura": (8.3114, 80.4037),
    "Polonnaruwa": (7.9403, 81.0188),
    "Badulla": (6.9934, 81.0550),
    "Monaragala": (6.8728, 81.3507),
    "Ratnapura": (6.6828, 80.3992),
    "Kegalle": (7.2513, 80.3464),
}

GITHUB_USER = "BGMPrabuddhi"
GITHUB_REPO_URL = f"https://github.com/{GITHUB_USER}/sl-dengue-ensemble"
