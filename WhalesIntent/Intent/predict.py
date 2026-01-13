# prediction/predict.py
import joblib
import pandas as pd
import numpy as np
import os

from data.features import engineer_features
from regimes.trend_regimes import define_trend_regimes
from regimes.r5_distribution import detect_r5_distribution
from core_trend_engine import CoreTrendEngine

class PredictionOrchestrator:
    """Orchestrates prediction using trained models"""
    
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.long_model = None
        self.short_model = None
        self.core_engine = None
        
    def load_models(self):
        """Load trained models"""
        try:
            # Load core engine
            self.core_engine = CoreTrendEngine()
            self.core_engine.load_models(self.model_dir)
            
            # Load additional models if available
            long_path = f'{self.model_dir}/long_best_model.pkl'
            short_path = f'{self.model_dir}/short_best_model.pkl'
            
            if os.path.exists(long_path):
                self.long_model = joblib.load(long_path)
                print(f"✅ Loaded long model: {long_path}")
            
            if os.path.exists(short_path):
                self.short_model = joblib.load(short_path)
                print(f"✅ Loaded short model: {short_path}")
                
        except Exception as e:
            print(f"⚠️  Error loading models: {e}")
    
    def predict_signal(self, df_features):
        """
        Generate prediction signal using loaded models
        """
        if self.core_engine is None:
            return {"error": "Models not loaded"}
        
        # Get latest data
        latest_row = df_features.iloc[-1].copy()
        
        # Use core engine for prediction
        signal = self.core_engine.generate_core_signal(latest_row, df_features)
        
        # Add model probabilities if available
        if self.long_model and signal['direction'] == 'LONG':
            try:
                features = self.long_model.feature_names_
                X = latest_row.reindex(features, fill_value=0).values.reshape(1, -1)
                prob = self.long_model.predict_proba(X)[0, 1]
                signal['ml_probability'] = float(prob)
            except Exception:
                pass
        
        return signal