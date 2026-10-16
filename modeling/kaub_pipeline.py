"""Training pipeline for the notebook-04 XGBoost model.

Does the same steps as the original `archive/notebooks/04_eda_and_multivariate_main.ipynb`, but in one place:

    1. load the daily data
    2. create features and the 7-day-ahead target
    3. split by time (train / validation / test)
    4. train XGBoost and evaluate it
    5. save the model

Run it with:

    uv run python -m modeling.kaub_pipeline

The saved file has the same format the Streamlit app expects
(model, features, horizon), and the same features realtime_pipeline.py creates.
"""

import joblib
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from xgboost import XGBRegressor

from modeling import hydro_pipeline as hp
from modeling.data_pipeline import REPO_ROOT

# ---------------------------------------------------------------
# Settings
# ---------------------------------------------------------------
MODEL_PATH = REPO_ROOT / "models" / "kaub_xgboost_04.joblib"

FEATURE_GROUPS = hp.FEATURE_GROUPS
ALL_FEATURES = FEATURE_GROUPS["All selected features"]

# The model that is saved for the app
FEATURE_GROUP = "Kaub + precipitation + temperature"
FEATURES = FEATURE_GROUPS[FEATURE_GROUP]
HORIZON = hp.HORIZON


# ---------------------------------------------------------------
# Step 1 + 2: data and features
# ---------------------------------------------------------------
def load_data():
    """Daily Kaub + weather table (one row per day)."""
    return hp.load_hydro_daily()


def make_features(daily):
    """Features + target, exactly like the original notebook 04 (inputs rounded first)."""
    return hp.build_features(daily, horizon=HORIZON, round_inputs=True)


# ---------------------------------------------------------------
# Step 3 + 4: train and evaluate
# ---------------------------------------------------------------
def make_model():
    """The XGBoost settings from the original notebook 04."""
    return XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )


def make_models():
    """All models from the original notebook 04, with the same settings."""
    return {
        "Linear regression": LinearRegression(),
        "LightGBM": LGBMRegressor(
            n_estimators=500, learning_rate=0.01, max_depth=6, num_leaves=7,
            min_child_samples=50, subsample=0.8, subsample_freq=1,
            reg_alpha=0.2, reg_lambda=5.0, random_state=42, n_jobs=1, verbosity=-1,
        ),
        "Random forest": RandomForestRegressor(
            n_estimators=300, max_depth=10, min_samples_split=10,
            min_samples_leaf=5, max_features=0.8, random_state=42, n_jobs=-1,
        ),
        "XGBoost": make_model(),
        "Polynomial ridge": make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False),
            StandardScaler(),
            Ridge(alpha=10.0),
        ),
    }


def split_data(features):
    """Train / validation / test split on the features of all groups."""
    return hp.time_split(features[ALL_FEATURES + ["target"]])


def train_model_groups(model_name, splits):
    """Train one model type on every feature group and score it on validation.

    Returns a results table and the trained models (one per feature group).
    """
    train_fit = splits["train_fit"]
    validation = splits["validation"]

    results = []
    models = {}

    for group, columns in FEATURE_GROUPS.items():
        model = make_models()[model_name]
        model.fit(train_fit[columns], train_fit["target"])

        predictions = model.predict(validation[columns])

        row = hp.metrics(validation["target"], predictions, validation["lag_0"])
        row["Feature group"] = group
        results.append(row)
        models[group] = model

    return pd.DataFrame(results).set_index("Feature group"), models


def refit_and_test(model_name, group, splits):
    """Train again on all training years and predict the test years."""
    columns = FEATURE_GROUPS[group]

    model = make_models()[model_name]
    model.fit(splits["train"][columns], splits["train"]["target"])

    test_predictions = model.predict(splits["test"][columns])
    return model, test_predictions


def evaluate(model, part):
    """Metrics for one data split, compared with persistence."""
    predictions = model.predict(part[FEATURES])
    return hp.metrics(part["target"], predictions, part["lag_0"])


def train(features):
    """Train the app model, check it on validation and test.

    Returns the trained model and a table with the results.
    """
    splits = split_data(features)

    # Validation: train on train_fit, check on validation
    model = make_model()
    model.fit(splits["train_fit"][FEATURES], splits["train_fit"]["target"])
    validation_result = evaluate(model, splits["validation"])

    # Test: train again on all training years, check once on test
    model = make_model()
    model.fit(splits["train"][FEATURES], splits["train"]["target"])
    test_result = evaluate(model, splits["test"])

    report = pd.DataFrame(
        [validation_result, test_result], index=["Validation", "Test"]
    )
    return model, report, splits


# ---------------------------------------------------------------
# Step 5: save, load, predict
# ---------------------------------------------------------------
def save_model(model, features=FEATURES, path=MODEL_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": features, "horizon": HORIZON}, path)


def load_model(path=MODEL_PATH):
    return joblib.load(path)


def predict(saved, daily):
    """Forecast from daily data. Returns one row per forecast day.

    `daily` must have one row per day (use hp.load_hydro_daily()).
    """
    features = make_features(daily)
    X = features[saved["features"]].dropna()

    forecast = pd.DataFrame(index=X.index)
    forecast["today_level"] = X["lag_0"]
    forecast["forecast_date"] = X.index + pd.Timedelta(days=saved["horizon"])
    forecast["forecast_level"] = saved["model"].predict(X)
    forecast["low_water_alarm"] = forecast["forecast_level"] < hp.LOW_WATER_CM
    return forecast


# ---------------------------------------------------------------
# Run everything
# ---------------------------------------------------------------
def run():
    print("1. Loading data ...")
    daily = load_data()

    print("2. Creating features ...")
    features = make_features(daily)

    print("3-4. Training and evaluating ...")
    model, report, splits = train(features)
    print(report[["Samples", "MAE", "RMSE", "Bias", "Persistence MAE"]].round(2))

    print("5. Saving model ...")
    save_model(model)
    print(f"Saved to {MODEL_PATH}")

    return model, report


if __name__ == "__main__":
    run()
