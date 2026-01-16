"""
ETH WHALE ALPHA - Main Orchestrator
"""

import os
import json
import pandas as pd
import joblib

from loader.data_loader import load_cached_data
from pipeline.pipeline_builder import build_pipeline_complete
from engines.core_trend_engine import CoreTrendEngine
from shadow_trading.shadow import run_shadow_trading
#from training.train import train_multi_models
from training.train_long_model import train_long_model
from training.train_short_model import train_short_model
from evaluation.inspection import (
    export_signals_for_manual_inspection,
    create_manual_review_template
)

# PIPELINE
def build_pipeline(save=True):
    print("\n" + "=" * 70)
    print("BUILDING DATA PIPELINE")
    print("=" * 70)

    df_whales, df_market, df_btc, df_eth, df_funding = load_cached_data()

    if any(d.empty for d in [df_whales, df_market, df_btc, df_eth]):
        raise RuntimeError("Missing essential input data")

    df_pipeline = build_pipeline_complete(
        df_whales=df_whales,
        df_market=df_market,
        df_btc=df_btc,
        df_eth=df_eth,
        df_funding=df_funding
    )

    if save:
        os.makedirs("data", exist_ok=True)
        df_pipeline.to_csv("data/pipeline_complete.csv", index=False)
        print(f"✅ Pipeline saved: {len(df_pipeline)} rows")

    return df_pipeline

def load_or_build_pipeline():
    if os.path.exists("data/pipeline_complete.csv"):
        return pd.read_csv("data/pipeline_complete.csv", parse_dates=["block_date"])
    return build_pipeline()

# SIGNAL GENERATION
def generate_daily_signal():
    print("\n" + "=" * 70)
    print("GENERATING DAILY SIGNAL")
    print("=" * 70)

    df = load_or_build_pipeline()
    engine = CoreTrendEngine()
    engine.load_models()

    signal = engine.generate_daily_signal(df)

    os.makedirs("data", exist_ok=True)
    with open("data/latest_signal.json", "w") as f:
        json.dump(signal, f, indent=2)

    print(json.dumps(signal, indent=2))
    return signal

# ML PREDICTION
def run_prediction():
    print("\n" + "=" * 70)
    print("RUNNING ML PREDICTION")
    print("=" * 70)

    df = load_or_build_pipeline()
    latest_row = df.iloc[-1]

    engine = CoreTrendEngine()
    engine.load_models()

    core_signal = engine.generate_core_signal(latest_row, df)
    predictions = {}

    for direction in ["LONG", "SHORT"]:
        model_path = f"models/best_{direction.lower()}_model.pkl"
        if not os.path.exists(model_path):
            continue

        model, metadata = load_model_with_metadata(model_path)
        features = metadata.get("features", [])

        X = latest_row.reindex(features, fill_value=0).values.reshape(1, -1)
        prob = model.predict_proba(X)[0, 1]

        predictions[direction] = {
            "probability": float(prob),
            "threshold": metadata.get("threshold"),
            "auc": metadata.get("auc"),
            "signal": "BUY" if prob >= metadata.get("threshold", 1) else "HOLD"
        }

    combined = {
        "date": str(latest_row["block_date"]),
        "regime": latest_row["regime_code"],
        "eth_price": float(latest_row["eth_price"]),
        "core_signal": core_signal,
        "ml_predictions": predictions
    }

    with open("data/latest_prediction.json", "w") as f:
        json.dump(combined, f, indent=2)

    print(json.dumps(combined, indent=2))
    return combined

def load_model_with_metadata(path):
    model = joblib.load(path)
    meta_path = path.replace(".pkl", "_metadata.json")
    metadata = {}

    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            metadata = json.load(f)

    return model, metadata

# TRAINING
def train_models():
    print("\n" + "=" * 70)
    print("TRAINING ML MODELS")
    print("=" * 70)

    df = load_or_build_pipeline()

    print("\n🔹 Training LONG models")
    train_multi_models(df, direction="LONG")

    print("\n🔹 Training SHORT models")
    train_multi_models(df, direction="SHORT")

    print("\n✅ Training complete")

# CLI
def main_menu():
    while True:
        print("\n" + "=" * 70)
        print("ETH WHALE ALPHA - MAIN MENU")
        print("=" * 70)
        print("1. Build pipeline")
        print("2. Generate daily signal")
        print("3. Run ML prediction")
        print("4. Run shadow trading")
        print("5. Train ML models")
        print("6. Manual inspection tools")
        print("0. Exit")

        choice = input("\nSelect option: ").strip()

        if choice == "1":
            build_pipeline()

        elif choice == "2":
            generate_daily_signal()

        elif choice == "3":
            run_prediction()

        elif choice == "4":
            df = load_or_build_pipeline()
            engine = CoreTrendEngine()
            engine.load_models()
            run_shadow_trading(df, engine)

        elif choice == "5":
            train_models()

        elif choice == "6":
            export_signals_for_manual_inspection(
                df=load_or_build_pipeline(),
                engine=CoreTrendEngine(),
                days=60
            )
            create_manual_review_template()

        elif choice == "0":
            print("👋 Exiting")
            break

        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main_menu()