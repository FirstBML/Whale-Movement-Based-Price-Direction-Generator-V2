"""
prediction/predict.py
Lambda-safe prediction entrypoint
SINGLE SOURCE OF TRUTH: CoreTrendEngine
"""

import json
import os
import pandas as pd

from loader.data_loader import load_cached_data
from pipeline.pipeline_builder import build_pipeline_complete
from engines.core_trend_engine import CoreTrendEngine


def load_pipeline():
    if os.path.exists("data/pipeline_complete.csv"):
        return pd.read_csv("data/pipeline_complete.csv", parse_dates=["block_date"])

    df_whales, df_market, df_btc, df_eth, df_funding = load_cached_data()
    return build_pipeline_complete(
        df_whales=df_whales,
        df_market=df_market,
        df_btc=df_btc,
        df_eth=df_eth,
        df_funding=df_funding
    )


def run_prediction():
    df = load_pipeline()

    engine = CoreTrendEngine()
    engine.load_models()

    latest_row = df.iloc[-1]

    signal = engine.generate_core_signal(
        latest_row,
        df
    )

    return {
        "date": str(latest_row["block_date"]),
        "regime": latest_row["regime_code"],
        "signal": signal,
        "engine": "core_trend"
    }


# 🔌 AWS LAMBDA HANDLER
def lambda_handler(event=None, context=None):
    try:
        result = run_prediction()
        return {
            "statusCode": 200,
            "body": json.dumps(result)
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
