# main.py - WITH PREDICTION OPTION
"""
ETH WHALE ALPHA - SIMPLE ORCHESTRATOR
Now includes prediction functionality
"""

import os
import json
import pandas as pd
import joblib

# Import only what we need
from data.data_loader import load_data_from_files
from data.features import engineer_features
from data.targets import create_targets_two_tier
from regimes.regime_detector import define_all_regimes
from core_trend_engine import CoreTrendEngine
from evaluation.shadow import run_shadow_trading
from training.train_models import train_multi_models

def build_pipeline():
    """Build complete pipeline"""
    print("\n" + "="*70)
    print("BUILDING PIPELINE")
    print("="*70)
    
    # 1. Load data
    df_whales, df_market, df_btc, df_eth, df_funding = load_data_from_files()
    
    # 2. Engineer features
    df_features = engineer_features(df_whales, df_market, df_btc, df_eth, df_funding)
    
    # 3. Create targets
    df_targets = create_targets_two_tier(df_features)
    
    # 4. Detect regimes
    df_complete = define_all_regimes(df_targets)
    
    # Save
    df_complete.to_csv('data/pipeline_complete.csv', index=False)
    
    print(f"\n✅ Pipeline saved: {len(df_complete)} rows")
    return df_complete

def generate_signal():
    """Generate daily signal using CoreTrendEngine"""
    print("\n" + "="*70)
    print("GENERATING DAILY SIGNAL (CORE TREND ENGINE)")
    print("="*70)
    
    # Load data
    if not os.path.exists('data/pipeline_complete.csv'):
        df = build_pipeline()
    else:
        df = pd.read_csv('data/pipeline_complete.csv', parse_dates=['block_date'])
    
    # Load engine
    engine = CoreTrendEngine()
    engine.load_models()
    
    # Generate signal
    signal = engine.generate_daily_signal(df)
    
    # Save
    with open('data/latest_signal.json', 'w') as f:
        json.dump(signal, f, indent=2)
    
    print(json.dumps(signal, indent=2))
    return signal

def run_prediction():
    """Run prediction using trained models"""
    print("\n" + "="*70)
    print("RUNNING PREDICTION WITH TRAINED MODELS")
    print("="*70)
    
    # Load data
    if not os.path.exists('data/pipeline_complete.csv'):
        df = build_pipeline()
    else:
        df = pd.read_csv('data/pipeline_complete.csv', parse_dates=['block_date'])
    
    # Get latest row
    latest_row = df.iloc[-1].copy()
    print(f"📅 Latest date: {latest_row['block_date']}")
    print(f"💰 ETH Price: ${latest_row['eth_price']:.2f}")
    print(f"📊 Current regime: {latest_row['regime_code']}")
    
    # Check for trained models
    long_model_path = 'models/best_long_model.pkl'
    short_model_path = 'models/best_short_model.pkl'
    
    predictions = {}
    
    # Load and run LONG model if available
    if os.path.exists(long_model_path):
        print(f"\n🤖 Running LONG model prediction...")
        try:
            long_model, metadata = load_model_with_metadata(long_model_path)
            features = metadata.get('features', [])
            
            # Prepare features
            X = latest_row.reindex(features, fill_value=0).values.reshape(1, -1)
            
            # Predict probability
            prob = long_model.predict_proba(X)[0, 1]
            predictions['LONG'] = {
                'probability': float(prob),
                'threshold': 0.35,  # From core_trend_engine
                'signal': 'BUY' if prob >= 0.35 else 'HOLD',
                'model_type': metadata.get('model_type', 'unknown'),
                'auc': metadata.get('auc', 0)
            }
            
            print(f"   Probability: {prob:.3f}")
            print(f"   Signal: {predictions['LONG']['signal']}")
            print(f"   Model: {metadata.get('model_type', 'unknown')} (AUC: {metadata.get('auc', 0):.3f})")
            
        except Exception as e:
            print(f"❌ Error with LONG model: {e}")
            predictions['LONG'] = {'error': str(e)}
    else:
        print("⚠️  No LONG model found. Run training first.")
    
    # Load and run SHORT model if available
    if os.path.exists(short_model_path):
        print(f"\n🤖 Running SHORT model prediction...")
        try:
            short_model, metadata = load_model_with_metadata(short_model_path)
            features = metadata.get('features', [])
            
            # Prepare features
            X = latest_row.reindex(features, fill_value=0).values.reshape(1, -1)
            
            # Predict probability
            prob = short_model.predict_proba(X)[0, 1]
            predictions['SHORT'] = {
                'probability': float(prob),
                'threshold': 0.55,  # From core_trend_engine
                'signal': 'SELL' if prob >= 0.55 else 'HOLD',
                'model_type': metadata.get('model_type', 'unknown'),
                'auc': metadata.get('auc', 0)
            }
            
            print(f"   Probability: {prob:.3f}")
            print(f"   Signal: {predictions['SHORT']['signal']}")
            print(f"   Model: {metadata.get('model_type', 'unknown')} (AUC: {metadata.get('auc', 0):.3f})")
            
        except Exception as e:
            print(f"❌ Error with SHORT model: {e}")
            predictions['SHORT'] = {'error': str(e)}
    else:
        print("⚠️  No SHORT model found. Run training first.")
    
    # Combine with core engine signal
    print(f"\n" + "="*70)
    print("COMBINED PREDICTION SUMMARY")
    print("="*70)
    
    # Get core engine signal
    engine = CoreTrendEngine()
    engine.load_models()
    core_signal = engine.generate_core_signal(latest_row, df)
    
    combined = {
        'date': str(latest_row['block_date']),
        'regime': latest_row['regime_code'],
        'eth_price': float(latest_row['eth_price']),
        'core_engine_signal': core_signal,
        'ml_predictions': predictions,
        'final_recommendation': determine_final_recommendation(core_signal, predictions)
    }
    
    # Save combined prediction
    with open('data/latest_prediction.json', 'w') as f:
        json.dump(combined, f, indent=2)
    
    print(f"\n📊 CORE ENGINE SIGNAL:")
    print(f"   Action: {core_signal.get('action', 'NO_TRADE')}")
    print(f"   Direction: {core_signal.get('direction', 'NONE')}")
    print(f"   Confidence: {core_signal.get('adjusted_confidence', 0):.3f}")
    
    print(f"\n🎯 FINAL RECOMMENDATION: {combined['final_recommendation']}")
    print(f"\n💾 Saved to: data/latest_prediction.json")
    
    return combined

def determine_final_recommendation(core_signal, ml_predictions):
    """Determine final trading recommendation"""
    core_action = core_signal.get('action', 'NO_TRADE')
    core_direction = core_signal.get('direction')
    
    # If core engine says ENTER, use it
    if core_action == 'ENTER':
        return f"{core_direction} (Core Engine)"
    
    # Check ML predictions if core is unsure
    if 'LONG' in ml_predictions and 'signal' in ml_predictions['LONG']:
        long_signal = ml_predictions['LONG']['signal']
        long_prob = ml_predictions['LONG'].get('probability', 0)
        
        if long_signal == 'BUY' and long_prob > 0.5:
            return "LONG (ML Model)"
    
    if 'SHORT' in ml_predictions and 'signal' in ml_predictions['SHORT']:
        short_signal = ml_predictions['SHORT']['signal']
        short_prob = ml_predictions['SHORT'].get('probability', 0)
        
        if short_signal == 'SELL' and short_prob > 0.6:
            return "SHORT (ML Model)"
    
    return "HOLD (No strong signal)"

def load_model_with_metadata(path):
    """Helper to load model with metadata"""
    model = joblib.load(path)
    
    metadata_path = path.replace('.pkl', '_metadata.json')
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    return model, metadata

def main_menu():
    """Simple menu with prediction option"""
    print("\n" + "="*70)
    print("ETH WHALE ALPHA - MAIN MENU")
    print("="*70)
    print("1. Build pipeline")
    print("2. Generate daily signal (Core Engine)")
    print("3. Run prediction (ML Models)")
    print("4. Run shadow trading")
    print("5. Train ML models")
    print("6. System status check")
    print("="*70)
    
    choice = input("\nSelect option (1-6): ").strip()
    
    if choice == '1':
        build_pipeline()
    
    elif choice == '2':
        generate_signal()
    
    elif choice == '3':
        run_prediction()
    
    elif choice == '4':
        if not os.path.exists('data/pipeline_complete.csv'):
            df = build_pipeline()
        else:
            df = pd.read_csv('data/pipeline_complete.csv', parse_dates=['block_date'])
        
        engine = CoreTrendEngine()
        engine.load_models()
        run_shadow_trading(df, engine)
    
    elif choice == '5':
        if not os.path.exists('data/pipeline_complete.csv'):
            df = build_pipeline()
        else:
            df = pd.read_csv('data/pipeline_complete.csv', parse_dates=['block_date'])
        
        # Train LONG models
        long_results, long_model = train_multi_models(df, direction='LONG')
        
        # Train SHORT models
        short_results, short_model = train_multi_models(df, direction='SHORT')
        
        print("\n✅ Training complete!")
    
    elif choice == '6':
        print("\n" + "="*70)
        print("SYSTEM STATUS CHECK")
        print("="*70)
        
        # Check files
        files_to_check = [
            ('data/pipeline_complete.csv', 'Pipeline data'),
            ('models/best_long_model.pkl', 'LONG model'),
            ('models/best_short_model.pkl', 'SHORT model'),
            ('data/latest_signal.json', 'Latest signal'),
            ('data/latest_prediction.json', 'Latest prediction')
        ]
        
        for filepath, description in files_to_check:
            if os.path.exists(filepath):
                print(f"✅ {description}: Found")
            else:
                print(f"⚠️  {description}: Not found")
        
        # Check modules
        print(f"\n📦 Module Status:")
        try:
            from core_trend_engine import CoreTrendEngine
            print(f"✅ Core Trend Engine: Available")
        except:
            print(f"❌ Core Trend Engine: Missing")
        
        try:
            from regimes.r6_mean_reversion import R6MeanReversionEngine
            print(f"✅ R6 Engine: Available")
        except:
            print(f"⚠️  R6 Engine: Not available (optional)")
    
    else:
        print("❌ Invalid option")

if __name__ == "__main__":
    main_menu()