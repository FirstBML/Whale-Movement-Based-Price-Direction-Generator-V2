"""
CORE TREND ENGINE
FROZEN R1/R2/R3/R5 Logic - DO NOT MODIFY
This is the capital-protective, high-precision trend engine.
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import json
import os

# ========== CONFIGURATION ==========
LONG_FEATURES = [
    'btc_rsi', 'vol_ratio', 'whale_volume_ratio', 'eth_rsi',  
    'btc_ret_lag1', 'eth_burned_zscore_90d', 'eth_btc_corr_30d',
    'eth_ret_lag1', 'btc_ret_lag7', 'btc_vol30',
    'whale_volume_ratio_delta_1d', 'whale_volume_ratio_delta_3d',
    'exchange_flow_share', 'net_exchange_flow_ratio',
    'whale_exchange_flow_ratio', 'tx_per_active_zscore_90d'
]

SHORT_FEATURES = [
    'exchange_flow_share', 'net_exchange_flow_ratio', 'whale_exchange_flow_ratio',
    'whale_exchange_asymmetry', 'vol_ratio', 'btc_ret_lag1',  
    'btc_ret_lag3', 'eth_btc_corr_30d', 'whale_volume_ratio_delta_3d',
    'exchange_volume_zscore'
]

# Entry thresholds - FROZEN
LONG_ENTRY_THRESHOLD = 0.35
SHORT_ENTRY_THRESHOLD = 0.55


class CoreTrendEngine:
    """
    Core Trend Engine - FROZEN
    Handles R1/R2 LONG and R3/R5 SHORT logic
    """
    
    def __init__(self):
        self.long_model = None
        self.short_model = None
        
    def load_models(self, model_dir='models'):
        """Load pre-trained models"""
        long_model_path = os.path.join(model_dir, 'R1_R2_LONG.pkl')
        short_model_path = os.path.join(model_dir, 'r5_short_final.pkl')
        
        try:
            if os.path.exists(long_model_path):
                self.long_model = joblib.load(long_model_path)
                print(f"✅ LONG model loaded: {len(self.long_model.feature_names_)} features")
            else:
                print(f"⚠️  LONG model not found: {long_model_path}")
            
            if os.path.exists(short_model_path):
                self.short_model = joblib.load(short_model_path)
                print(f"✅ SHORT model loaded: {len(self.short_model.feature_names_)} features")
            else:
                print(f"⚠️  SHORT model not found: {short_model_path}")
                
        except Exception as e:
            print(f"❌ Error loading models: {e}")
    
    # ========== LONG-SPECIFIC LOGIC ==========
    def long_veto(self, row):
        """LONG veto - minimal and asymmetric (FROZEN)"""
        veto = []
        
        if row.get('btc_ret_lag1', 0) < -0.02:
            veto.append("btc_drawdown")
        
        if row.get('whale_exchange_flow_ratio', 0) > 0.6:
            veto.append("distribution")
        
        if row.get('vol_ratio', 1) > 1.5:
            veto.append("vol_spike")
        
        return veto
    
    def calculate_long_support_score(self, row):
        """Calculate support score for LONG signals (clarity fix)"""
        score = 0
        
        # Price momentum support
        if row.get('eth_ret_lag1', 0) > 0:
            score += 1
        if row.get('eth_ret_lag2', 0) > 0:
            score += 1
        
        # Volatility compression support
        if row.get('vol_ratio', 1) < 1.0:
            score += 1
        
        # BTC alignment support
        if row.get('btc_ret_lag1', 0) > -0.005:
            score += 1
        
        # Whale absorption support
        if row.get('whale_volume_ratio', 0) > 0.5:
            score += 1
        
        # Flow support (not distribution)
        if row.get('whale_exchange_flow_ratio', 0) < 0.3:
            score += 1
        
        return score
    
    def adjust_confidence_long(self, prob, regime, support_score=0, row=None):
        """Confidence adjustment for LONG signals (FROZEN)"""
        base_conf = float(prob)
        
        # Apply support boost
        support_boost = np.tanh(support_score / 3) * 0.15
        adj_conf = np.clip(base_conf + support_boost, 0, 1)
        
        # Apply regime-specific caps
        if regime == "R3":
            max_conf = 0.70
        elif regime == "R5":
            max_conf = 0.85
        elif regime in ["R1", "R2"]:
            max_conf = 0.75
        else:
            max_conf = 0.95
        
        adj_conf = min(adj_conf, max_conf)
        return adj_conf
    
    # ========== SHORT-SPECIFIC LOGIC ==========
    def check_r3_short_allowed(self, row):
        """
        R3 short philosophy (early weakness only) - FROZEN
        """
        small_red = (-0.015 < row.get('eth_ret_lag1', 0) < 0)
        btc_weak = (row.get('btc_ret_lag3', 0) < 0)
        no_vol_expansion = (row.get('vol_ratio', 1) <= 1.0)
        whale_activity = (row.get('whale_volume_ratio_delta_3d', 0) > 0)
        
        return small_red and btc_weak and no_vol_expansion and whale_activity
    
    def calculate_short_veto_score(self, row):
        """Calculate veto scores for SHORT positions (FROZEN)"""
        veto = 0
        reasons = []
        
        structural_score = 0
        flow_score = 0
        context_score = 0
        
        # Flow vetoes
        if row.get('net_exchange_flow_ratio', 0) < 0 and row.get('exchange_volume_zscore', 0) > 0:
            veto += 1
            flow_score += 1
            reasons.append('net_flow_negative_with_liquidity')
        
        if row.get('whale_exchange_flow_ratio', 0) > 0.6:
            veto += 1
            flow_score += 1
            reasons.append('whale_to_exchange')
        
        # Structural vetoes
        if row.get('btc_ret_lag1', 0) < -0.02 and row.get('eth_ret_lag1', 0) < -0.01:
            veto += 2
            structural_score += 2
            reasons.append('btc_breakdown')
        
        # Vol expansion check
        if row.get('vol_ratio', 1) > 1.0:
            veto += 2
            structural_score += 2
            reasons.append('vol_expansion')
        
        # Low volatility check
        if row.get('vol_ratio', 1) < 0.7:
            veto += 1
            context_score += 1
            reasons.append('low_volatility')
        
        return veto, structural_score, flow_score, context_score, reasons
    
    def check_short_requirements(self, row, regime, structural_score, flow_score):
        """Check SHORT-specific requirements (FROZEN)"""
        reasons = []
        
        # Flow confirmation with nuance
        if flow_score == 0:
            if not (structural_score > 0 and row.get('btc_ret_lag1', 0) < 0):
                reasons.append("no_flow_confirmation")
        
        # R5 stronger flow requirement
        if regime == "R5" and flow_score > 0 and flow_score < 2:
            reasons.append("weak_distribution_flow")
        
        # Structural check
        if structural_score == 0:
            reasons.append("no_structural_break")
        
        # R3 short check
        if regime == "R3" and not self.check_r3_short_allowed(row):
            reasons.append("r3_no_early_weakness")
        
        return reasons
    
    def adjust_confidence_short(self, prob, regime, veto_score=0):
        """Confidence adjustment for SHORT signals (FROZEN)"""
        base_conf = float(prob)
        
        # Apply veto boost (same for both directions)
        veto_boost = np.tanh(veto_score / 3) * 0.15
        adj_conf = np.clip(base_conf + veto_boost, 0, 1)
        
        # Apply confidence caps based on regime
        if regime == "R3":
            max_conf = 0.70
        elif regime == "R5":
            max_conf = 0.85
        elif regime in ["R1", "R2"]:
            max_conf = 0.75
        else:
            max_conf = 0.95
        
        adj_conf = min(adj_conf, max_conf)
        return adj_conf
    
    # ========== SIGNAL GENERATION ==========
    def generate_core_signal(self, row, df):
        """
        Generate core trend signal (R1/R2/R3/R5 only)
        Returns signal dictionary or None if no valid signal
        """
        regime = row.get('regime_code', 'R0')
        date_str = str(row['block_date'].date()) if 'block_date' in row else str(row.name)
        
        # Base signal object
        signal = {
            "date": date_str,
            "regime": regime,
            "direction": None,
            "model_probability": 0.0,
            "adjusted_confidence": 0.0,
            "position_size": 0.0,
            "reasons": [],
            "action": "NO_TRADE",
            "engine": "core_trend"
        }
        
        # ===== LONG LOGIC (R1/R2) =====
        if regime in ['R1', 'R2'] and self.long_model:
            signal["direction"] = "LONG"
            
            if not hasattr(self.long_model, 'feature_names_'):
                signal["reasons"] = ["model_error: missing feature_names_"]
                return signal
            
            features = self.long_model.feature_names_
            try:
                X = row.reindex(features, fill_value=0).values.reshape(1, -1)
                prob = self.long_model.predict_proba(X)[0, 1]
                signal["model_probability"] = float(prob)
            except Exception as e:
                signal["reasons"] = [f"model_error: {str(e)[:100]}"]
                return signal
            
            # Check probability threshold
            if prob < LONG_ENTRY_THRESHOLD:
                signal["reasons"] = ["low_model_probability"]
                return signal
            
            # Calculate confirmation score
            confirm_score = 0
            if row.get('eth_ret_lag1', 0) > 0:
                confirm_score += 1
            if row.get('eth_ret_lag2', 0) > 0:
                confirm_score += 1
            if row.get('vol_ratio', 1) < 1.0:
                confirm_score += 1
            
            required_score = 1 if regime == "R1" else 2
            
            if confirm_score < required_score:
                signal["reasons"] = [f"weak_price_confirmation ({confirm_score}/{required_score})"]
                return signal
            
            # Apply LONG vetoes
            veto_reasons = self.long_veto(row)
            if veto_reasons:
                signal["reasons"] = veto_reasons
                return signal
            
            # Calculate confidence
            support_score = self.calculate_long_support_score(row)
            signal["adjusted_confidence"] = self.adjust_confidence_long(
                prob, regime, support_score, row
            )
            
            # Regime-aware confidence floor
            confidence_floor = 0.50 if regime == "R1" else 0.55
            
            if signal["adjusted_confidence"] < confidence_floor:
                signal["reasons"] = ["low_final_confidence"]
                signal["direction"] = None
                return signal
            
            # Map to position size
            signal["position_size"] = self.map_confidence_to_size(
                signal["adjusted_confidence"], regime, "LONG"
            )
            signal["reasons"] = ["ml_accumulation", "price_confirmation"]
            signal["action"] = "ENTER"
            return signal
            
        # ===== SHORT LOGIC (R3/R5) =====
        elif regime in ['R3', 'R5'] and self.short_model:
            signal["direction"] = "SHORT"
            
            if not hasattr(self.short_model, 'feature_names_'):
                signal["reasons"] = ["model_error: missing feature_names_"]
                return signal
            
            features = self.short_model.feature_names_
            try:
                X = row.reindex(features, fill_value=0).values.reshape(1, -1)
                prob = self.short_model.predict_proba(X)[0, 1]
                signal["model_probability"] = float(prob)
            except Exception as e:
                signal["reasons"] = [f"model_error: {str(e)[:100]}"]
                return signal
            
            # Check probability threshold
            if prob < SHORT_ENTRY_THRESHOLD:
                signal["reasons"] = ["low_model_probability"]
                return signal
            
            # Calculate veto scores
            veto, structural_score, flow_score, context_score, veto_reasons = \
                self.calculate_short_veto_score(row)
            
            # Check SHORT-specific requirements
            requirement_failures = self.check_short_requirements(
                row, regime, structural_score, flow_score
            )
            if requirement_failures:
                signal["reasons"] = requirement_failures
                return signal
            
            # Calculate confidence
            signal["adjusted_confidence"] = self.adjust_confidence_short(
                prob, regime, veto
            )
            
            # Final confidence check
            if signal["adjusted_confidence"] < 0.55:
                signal["reasons"] = ["low_final_confidence"]
                signal["direction"] = None
                return signal
            
            # Map to position size
            signal["position_size"] = self.map_confidence_to_size(
                signal["adjusted_confidence"], regime, "SHORT"
            )
            signal["reasons"] = veto_reasons
            signal["action"] = "ENTER"
            return signal
        
        # ===== NEUTRAL REGIME =====
        else:
            signal["reasons"] = ["neutral_regime"]
            return signal
    
    def map_confidence_to_size(self, conf, regime=None, direction=None):
        """
        Map confidence to position size (FROZEN)
        """
        # Confidence floors
        if direction == "LONG":
            if conf < 0.35:
                return 0.0
        elif direction == "SHORT":
            if conf < 0.55:
                return 0.0
        else:
            if conf < 0.55:
                return 0.0
        
        # Base sizing scale
        if conf < 0.60: 
            base_size = 0.25
        elif conf < 0.65: 
            base_size = 0.50
        elif conf < 0.70: 
            base_size = 0.75
        elif conf < 0.75: 
            base_size = 1.00
        elif conf < 0.80: 
            base_size = 1.25
        else: 
            base_size = 1.50
        
        # Regime-specific caps
        if regime in ["R3", "R5"]:
            base_size = min(base_size, 1.0)
        elif regime in ["R1", "R2"]:
            if direction == "LONG":
                base_size = min(base_size, 1.25)
            else:
                base_size = min(base_size, 1.0)
        else:
            base_size = min(base_size, 1.0)
        
        return base_size
    
    def generate_daily_signal(self, df):
        """Generate daily signal from latest row"""
        if len(df) == 0:
            return None
        
        latest_row = df.iloc[-1].copy()
        return self.generate_core_signal(latest_row, df)


# ========== HELPER FUNCTIONS ==========
def rebuild_core_models(df_pipeline, model_dir='models'):
    """
    Rebuild core models if needed
    """
    print("\n" + "="*70)
    print("REBUILDING CORE MODELS")
    print("="*70)
    
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import precision_score, recall_score
    
    # Rebuild SHORT model if needed
    short_model_path = os.path.join(model_dir, 'r5_short_final.pkl')
    if not os.path.exists(short_model_path):
        print("Building SHORT model (R5)...")
        df_r5 = df_pipeline[df_pipeline['regime_code'] == 'R5'].copy()
        
        short_features = [f for f in SHORT_FEATURES if f in df_r5.columns]
        print(f"   Using {len(short_features)} SHORT features")
        
        if len(df_r5) >= 50:
            split_idx = int(len(df_r5) * 0.8)
            X_train_short = df_r5[short_features].iloc[:split_idx].fillna(0)
            y_train_short = df_r5['y_short_t2'].iloc[:split_idx]
            
            short_model = GradientBoostingClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42
            )
            short_model.fit(X_train_short, y_train_short)
            short_model.feature_names_ = short_features
            
            joblib.dump(short_model, short_model_path)
            print(f"✅ SHORT model rebuilt and saved")
        else:
            print("⚠️  Insufficient R5 data for SHORT model")
    
    # Rebuild LONG model if needed
    long_model_path = os.path.join(model_dir, 'R1_R2_LONG.pkl')
    if not os.path.exists(long_model_path):
        print("Building LONG model (R1+R2)...")
        df_long = df_pipeline[df_pipeline['regime_code'].isin(['R1', 'R2'])].copy()
        
        # Remove obvious traps
        df_long = df_long[
            (df_long['btc_ret_lag1'] > -0.02) &
            (df_long['whale_exchange_flow_ratio'] < 0.6)
        ]
        
        long_features = [f for f in LONG_FEATURES if f in df_long.columns]
        print(f"   Using {len(long_features)} LONG features")
        
        X_long = df_long[long_features].fillna(0)
        y_long = df_long['y_long_t2']
        
        if len(X_long) >= 50:
            long_model = GradientBoostingClassifier(
                n_estimators=120,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            )
            
            long_model.fit(X_long, y_long)
            long_model.feature_names_ = long_features
            joblib.dump(long_model, long_model_path)
            
            print("✅ LONG model rebuilt and saved")
            
            # Basic validation
            probs = long_model.predict_proba(X_long)[:, 1]
            preds = (probs >= 0.60).astype(int)
            prec = precision_score(y_long, preds, zero_division=0)
            rec = recall_score(y_long, preds, zero_division=0)
            
            print(f"   Training precision: {prec:.3f}")
            print(f"   Training recall: {rec:.3f}")
        else:
            print("⚠️  Insufficient LONG data")
    
    # Load and return models
    engine = CoreTrendEngine()
    engine.load_models(model_dir)
    return engine