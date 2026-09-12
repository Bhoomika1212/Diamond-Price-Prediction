import os
import json
import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# --------------------------------------------------
# PATH CONFIGURATION
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "diamonds.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "model"
)


# --------------------------------------------------
# CATEGORY ENCODING
# --------------------------------------------------

CUT_MAP = {
    "Ideal": 5,
    "Premium": 4,
    "Very Good": 3,
    "Good": 2,
    "Fair": 1
}

COLOR_MAP = {
    "D": 7,
    "E": 6,
    "F": 5,
    "G": 4,
    "H": 3,
    "I": 2,
    "J": 1
}

CLARITY_MAP = {
    "IF": 8,
    "VVS1": 7,
    "VVS2": 6,
    "VS1": 5,
    "VS2": 4,
    "SI1": 3,
    "SI2": 2,
    "I1": 1
}


# --------------------------------------------------
# FEATURES
# --------------------------------------------------

FEATURES = [
    "carat",
    "cut",
    "color",
    "clarity",
    "depth",
    "table",
    "x",
    "y",
    "z"
]


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# --------------------------------------------------
# REMOVE MISSING VALUES
# --------------------------------------------------

df = df.dropna(
    subset=FEATURES + ["price"]
)


# --------------------------------------------------
# REMOVE INVALID DIMENSIONS
# --------------------------------------------------

df = df[
    (df["x"] > 0) &
    (df["y"] > 0) &
    (df["z"] > 0)
]


# --------------------------------------------------
# ENCODE CATEGORICAL FEATURES
# --------------------------------------------------

df["cut"] = df["cut"].map(CUT_MAP)

df["color"] = df["color"].map(COLOR_MAP)

df["clarity"] = df["clarity"].map(CLARITY_MAP)


# Remove rows that could not be encoded

df = df.dropna(
    subset=FEATURES
)


# --------------------------------------------------
# INPUT AND TARGET
# --------------------------------------------------

X = df[FEATURES]

y = df["price"]


# --------------------------------------------------
# TRAIN TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print(
    "Training records:",
    len(X_train)
)

print(
    "Testing records:",
    len(X_test)
)


# --------------------------------------------------
# RANDOM FOREST MODEL
# --------------------------------------------------

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    max_features="sqrt"
)


# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------

print("Training Random Forest model...")

model.fit(
    X_train,
    y_train
)


print("Training completed.")


# --------------------------------------------------
# PREDICTIONS
# --------------------------------------------------

y_pred = model.predict(
    X_test
)


# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n-----------------------------")
print("MODEL PERFORMANCE")
print("-----------------------------")

print(
    "MAE:",
    round(mae, 2)
)

print(
    "RMSE:",
    round(rmse, 2)
)

print(
    "R2 Score:",
    round(r2, 4)
)


# --------------------------------------------------
# CREATE MODEL DIRECTORY
# --------------------------------------------------

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "diamond_price_model.pkl"
)

with open(
    MODEL_PATH,
    "wb"
) as file:

    pickle.dump(
        model,
        file
    )


# --------------------------------------------------
# SAVE METRICS
# --------------------------------------------------

metrics = {

    "MAE": float(mae),

    "RMSE": float(rmse),

    "R2": float(r2),

    "train_rows": int(
        len(X_train)
    ),

    "test_rows": int(
        len(X_test)
    ),

    "dataset_rows": int(
        len(df)
    )
}


METRICS_PATH = os.path.join(
    MODEL_DIR,
    "metrics.json"
)


with open(
    METRICS_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


print("\nModel saved successfully.")

print(
    "Model:",
    MODEL_PATH
)

print(
    "Metrics:",
    METRICS_PATH
)