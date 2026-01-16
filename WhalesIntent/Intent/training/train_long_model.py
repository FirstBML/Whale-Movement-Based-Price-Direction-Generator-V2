"""
# train_long_model.py
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss

# ======================================================
# CONFIG
# ======================================================

SPLIT_DATE = "2023-01-01"
MODEL_VERSION = "1.0.0"

EXPORT_PATH = "models/r1r2_long_final_v1.0.pkl"

LONG_FEATURES = [
    "eth_ret_lag1",
    "eth_ret_lag3",
    "btc_ret_lag1",
    "vol_ratio",
    "btc_vol7",
    "btc_vol30",
    "non_exchange_tx_count",
    "deposit_withdrawal_ratio",
    "whale_volume_ratio_delta_3d",
    "eth_btc_ratio",
    "eth_btc_ratio_ma7",
    "exchange_volume_ratio"
]

# ======================================================
# MODEL WRAPPER
# ======================================================

class CalibratedModelWrapper:
    """
    Safe wrapper to preserve feature names
    and allow engine-level usage.
    """
    def __init__(self, calibrated_model, feature_names):
        self.calibrated_model = calibrated_model
        self.feature_names_ = feature_names

        for attr in ["classes_", "n_features_in_", "n_outputs_"]:
            if hasattr(calibrated_model, attr):
                setattr(self, attr, getattr(calibrated_model, attr))

    def predict_proba(self, X):
        return self.calibrated_model.predict_proba(X)

    def predict(self, X):
        return self.calibrated_model.predict(X)

# ======================================================
# TRAINING ENTRY
# ======================================================

def train_long_model(df_pipeline: pd.DataFrame):
    print("\n" + "=" * 70)
    print("TRAINING LONG MODEL (R1 / R2)")
    print("=" * 70)

    # --------------------------------------------------
    # 1. REGIME GATE
    # --------------------------------------------------

    df = df_pipeline[df_pipeline["regime_code"].isin(["R1", "R2"])].copy()

    if len(df) < 300:
        raise RuntimeError("❌ Insufficient R1/R2 data for LONG training")

    print(f"✅ Regime-gated rows: {len(df)}")

    # --------------------------------------------------
    # 2. TARGET
    # --------------------------------------------------

    y = (df["target_t2"] == 1).astype(int)
    X = df[LONG_FEATURES].fillna(0)

    pos_rate = y.mean() * 100
    print(f"📊 Positive rate: {pos_rate:.1f}%")

    if not 10 <= pos_rate <= 35:
        print("⚠️  WARNING: Target distribution outside ideal range")

    # --------------------------------------------------
    # 3. TIME-AWARE SPLIT
    # --------------------------------------------------

    train_mask = df["block_date"] < SPLIT_DATE
    val_mask   = df["block_date"] >= SPLIT_DATE

    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]

    print(f"📅 Train: {len(X_train)} | Val: {len(X_val)}")

    # --------------------------------------------------
    # 4. BASE MODEL
    # --------------------------------------------------

    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=5,
        min_samples_leaf=40,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)

    raw_probs = rf.predict_proba(X_val)[:, 1]
    raw_auc = roc_auc_score(y_val, raw_probs)
    raw_brier = brier_score_loss(y_val, raw_probs)

    print(f"📈 Raw AUC: {raw_auc:.4f}")
    print(f"📉 Raw Brier: {raw_brier:.4f}")

    # --------------------------------------------------
    # 5. CALIBRATION (LEAKAGE-SAFE)
    # --------------------------------------------------

    rf_cal = CalibratedClassifierCV(
        rf,
        method="isotonic",
        cv=5
    )

    rf_cal.fit(X_train, y_train)

    cal_probs = rf_cal.predict_proba(X_val)[:, 1]
    cal_auc = roc_auc_score(y_val, cal_probs)
    cal_brier = brier_score_loss(y_val, cal_probs)

    print(f"📈 Calibrated AUC: {cal_auc:.4f}")
    print(f"📉 Calibrated Brier: {cal_brier:.4f}")

    # --------------------------------------------------
    # 6. EXPORT
    # --------------------------------------------------

    wrapped_model = CalibratedModelWrapper(
        rf_cal,
        LONG_FEATURES
    )

    export_payload = {
        "model": wrapped_model,
        "direction": "LONG",
        "regime_gate": ["R1", "R2"],
        "r1_threshold": 0.72,   # strict accumulation
        "r2_threshold": 0.65,   # continuation
        "features": LONG_FEATURES,
        "metrics": {
            "raw_auc": raw_auc,
            "cal_auc": cal_auc,
            "raw_brier": raw_brier,
            "cal_brier": cal_brier
        },
        "version": MODEL_VERSION,
        "export_date": pd.Timestamp.now().isoformat()
    }

    os.makedirs("models", exist_ok=True)
    joblib.dump(export_payload, EXPORT_PATH)

    print("\n✅ LONG MODEL EXPORTED")
    print(f"   Path: {EXPORT_PATH}")
    print(f"   Regimes: R1 / R2")
    print(f"   Thresholds: R1=0.72 | R2=0.65")
    print(f"   Version: {MODEL_VERSION}")

    return export_payload


# ======================================================
# MAIN EXECUTION
# ======================================================

if __name__ == "__main__":
    # Load your data
    print("📂 Loading data...")
    
    try:
        # Adjust this path to your actual data file
        df_pipeline = pd.read_parquet("../../data/pipeline/pipeline_data.parquet")
        # Or if it's a CSV:
        # df_pipeline = pd.read_csv("../../data/pipeline/pipeline_data.csv")
        
        print(f"✅ Data loaded: {len(df_pipeline)} rows")
        
        # Train the model
        train_long_model(df_pipeline)
        
    except FileNotFoundError as e:
        print(f"❌ Error: Data file not found at {e.filename}")
        print("Please update the file path in train_long_model.py")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")