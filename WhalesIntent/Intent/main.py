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
from shadow_trading.run_shadow_trading import run_shadow_trading
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
    """Run ML prediction with full output"""
    print("\n" + "=" * 70)
    print("RUNNING ML PREDICTION")
    print("=" * 70)

    # Load pipeline
    df = load_or_build_pipeline()
    
    # Initialize predictor
    from prediction.predict import PredictionOrchestrator
    predictor = PredictionOrchestrator(model_dir='models')
    
    # Load models
    models_loaded = predictor.load_models()
    
    if not any(models_loaded.values()):
        print("❌ No models loaded - train models first")
        return None

    # Generate prediction
    prediction = predictor.predict_signal(df)
    
    # Save to file
    os.makedirs("data", exist_ok=True)
    with open("data/latest_prediction.json", "w") as f:
        json.dump(prediction, f, indent=2)

    # Print formatted output
    print("\n" + "=" * 70)
    print("PREDICTION OUTPUT")
    print("=" * 70)
    
    print(f"\n📅 Date: {prediction['date']}")
    print(f"🏷️  Regime: {prediction['regime']}")
    print(f"💰 ETH Price: ${prediction['eth_price']:.2f}")
    
    signal = prediction['signal']
    print(f"\n🎯 Signal:")
    print(f"   Action: {signal['action']}")
    print(f"   Direction: {signal['direction']}")
    print(f"   Confidence: {signal['confidence']:.3f}")
    print(f"   Position Size: {signal['position_size']:.2f}x")
    print(f"   Model Probability: {signal['model_probability']:.3f}")
    print(f"   Reasons: {', '.join(signal['reasons'])}")
    
    if 'model_info' in prediction:
        print(f"\n📊 Model Info:")
        for key, value in prediction['model_info'].items():
            print(f"   {key}: {value}")
    
    print(f"\n📈 Market Context:")
    for key, value in prediction['market_context'].items():
        print(f"   {key}: {value:.4f}")
    
    print(f"\n💾 Saved to: data/latest_prediction.json")
    
    return prediction

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

    print("\n🔹 Training SHORT model")
    from training.train_short_model import train_short_model
    train_short_model(df)

    print("\n🔹 Training LONG model")
    train_long_model(df)

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