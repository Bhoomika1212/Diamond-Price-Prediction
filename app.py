from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify

import os
import json
import pickle

import pandas as pd


# --------------------------------------------------
# FLASK APPLICATION
# --------------------------------------------------

app = Flask(__name__)


# --------------------------------------------------
# BASE DIRECTORY
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "diamond_price_model.pkl"
)


with open(
    MODEL_PATH,
    "rb"
) as file:

    model = pickle.load(
        file
    )


# --------------------------------------------------
# LOAD MODEL METRICS
# --------------------------------------------------

METRICS_PATH = os.path.join(
    BASE_DIR,
    "model",
    "metrics.json"
)


with open(
    METRICS_PATH,
    "r",
    encoding="utf-8"
) as file:

    metrics = json.load(
        file
    )


# --------------------------------------------------
# CATEGORY MAPPINGS
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
# FEATURE CREATION
# --------------------------------------------------

def create_features(data):

    carat = float(
        data["carat"]
    )

    depth = float(
        data["depth"]
    )

    table = float(
        data["table"]
    )

    x = float(
        data["x"]
    )

    y = float(
        data["y"]
    )

    z = float(
        data["z"]
    )


    cut = data["cut"]

    color = data["color"]

    clarity = data["clarity"]


    # --------------------------------------------------
    # VALIDATION
    # --------------------------------------------------

    if carat < 0.20 or carat > 5.10:

        raise ValueError(
            "Carat must be between 0.20 and 5.10."
        )


    if depth < 43 or depth > 79:

        raise ValueError(
            "Depth must be between 43 and 79."
        )


    if table < 43 or table > 95:

        raise ValueError(
            "Table must be between 43 and 95."
        )


    if x <= 0:

        raise ValueError(
            "X must be greater than 0."
        )


    if y <= 0:

        raise ValueError(
            "Y must be greater than 0."
        )


    if z <= 0:

        raise ValueError(
            "Z must be greater than 0."
        )


    if cut not in CUT_MAP:

        raise ValueError(
            "Invalid cut value."
        )


    if color not in COLOR_MAP:

        raise ValueError(
            "Invalid color value."
        )


    if clarity not in CLARITY_MAP:

        raise ValueError(
            "Invalid clarity value."
        )


    # --------------------------------------------------
    # CREATE DATAFRAME
    # --------------------------------------------------

    input_data = [

        carat,

        CUT_MAP[cut],

        COLOR_MAP[color],

        CLARITY_MAP[clarity],

        depth,

        table,

        x,

        y,

        z

    ]


    dataframe = pd.DataFrame(
        [input_data],
        columns=FEATURES
    )


    return dataframe


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/")
def home():

    return render_template(
        "index.html",
        metrics=metrics
    )


# --------------------------------------------------
# PREDICTION ROUTE
# --------------------------------------------------

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        data = request.get_json(
            silent=True
        )

        if not data:

            data = request.form


        features = create_features(
            data
        )


        prediction = model.predict(
            features
        )[0]


        prediction = max(
            float(prediction),
            0
        )


        return jsonify({

            "success": True,

            "price": round(
                prediction,
                2
            ),

            "currency": "USD"

        })


    except Exception as error:

        return jsonify({

            "success": False,

            "error": str(error)

        }), 400


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.route("/health")
def health():

    return jsonify({

        "status": "ok",

        "model_loaded": True

    })


# --------------------------------------------------
# RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )