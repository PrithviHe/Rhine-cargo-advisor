import logging
from pathlib import Path

import parsenvy

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

try:
    TRACKING_URI = (ROOT / ".mlflow_uri").read_text().strip()
except FileNotFoundError:
    TRACKING_URI = parsenvy.str("MLFLOW_URI")

if TRACKING_URI and TRACKING_URI.startswith("sqlite:///") \
        and not TRACKING_URI.startswith("sqlite:////"):
    TRACKING_URI = f"sqlite:///{ROOT / TRACKING_URI.removeprefix('sqlite:///')}"

EXPERIMENT_NAME = "hydroload-baseline"

DATE_COL = "date"
LEVEL_COL = "water_level_mean"
DATA_PATH = ROOT / "data" / "processed" / "kaub_water_weather_daily.csv"

HORIZON = 7
TEST_YEARS = 3
LAGS = [0, 1, 2, 3, 4, 5, 6, 7, 14, 21, 28]

REGIME = {"lowwater_100": 100, "glw_78": 78, "critical_40": 40}