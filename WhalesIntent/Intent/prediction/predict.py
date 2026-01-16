import joblib
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

from engines.core_trend_engine import CoreTrendEngine

class PredictionOrchestrator:
    """Orchestrates prediction using trained models"""
    
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.core_engine = None
        self.long_metadata = None
        self.short_metadata = None
        
    def load_models(self):
        """Load trained models with metadata"""
        try:
            # Load core engine
            self.core_engine = CoreTrendEngine()
            self.core_engine.load_models(self.model_dir)
            
            # Load metadata
            long_path = f'{self.model_dir}/r1r2_long_final_v1.0.pkl'
            short_path = f'{self.model_dir}/r5_short_final_v1.1.pkl'
            
            models_loaded = {"SHORT": False, "LONG": False}
            
            if os.path.exists(short_path):
                self.short_metadata = joblib.load(short_path)
                models_loaded["SHORT"] = True
            
            if os.path.exists(long_path):
                self.long_metadata = joblib.load(long_path)
                models_loaded["LONG"] = True
            
            print(f"✅ Models loaded:")
            print(f"   SHORT: {'YES' if models_loaded['SHORT'] else 'NO'}")
            print(f"   LONG : {'YES' if models_loaded['LONG'] else 'NO'}")
            
            return models_loaded
                
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return {"SHORT": False, "LONG": False}
    
    def predict_signal(self, df_features):
        """
        Generate prediction with full context
        """
        if self.core_engine is None:
            return {"error": "Engine not loaded"}

        latest_row = df_features.iloc[-1].copy()
        regime = latest_row.get("regime_code", "R0")

        # Get core signal from engine
        signal = self.core_engine.generate_core_signal(
            latest_row,
            df_features
        )

        # Build prediction output
        prediction = {
            "timestamp": datetime.now().isoformat(),
            "date": str(latest_row["block_date"].date()),
            "regime": regime,
            "eth_price": float(latest_row["eth_price"]),
            "btc_price": float(latest_row.get("btc_price", 0)),
            "signal": {
                "action": signal["action"],
                "direction": signal["direction"],
                "confidence": signal["adjusted_confidence"],
                "position_size": signal["position_size"],
                "model_probability": signal["model_probability"],
                "reasons": signal["reasons"],
                "engine": signal["engine"]
            },
            "market_context": {
                "vol_ratio": float(latest_row.get("vol_ratio", 1.0)),
                "btc_ret_lag1": float(latest_row.get("btc_ret_lag1", 0)),
                "eth_ret_lag1": float(latest_row.get("eth_ret_lag1", 0))
            }
        }

        # # Add model-specific details if available
        # if signal["direction"] == "SHORT" and self.short_metadata:
        #     prediction["model_info"] = {
        #         "type": "SHORT",
        #         "threshold": self.short_metadata.get("r5_threshold", 0.55),
        #         "regime_gate": self.short_metadata.get("regime_gate", ["R3", "R5"]),
        #         "version": self.short_metadata.get("version", "unknown")
        #     }
        # elif signal["direction"] == "LONG" and self.long_metadata:
        #     prediction["model_info"] = {
        #         "type": "LONG",
        #         "r1_threshold": self.long_metadata.get("r1_threshold", 0.72),
        #         "r2_threshold": self.long_metadata.get("r2_threshold", 0.65),
        #         "regime_gate": self.long_metadata.get("regime_gate", ["R1", "R2"]),
        #         "version": self.long_metadata.get("version", "unknown")
        #     }

        return prediction