# training/train_short_model.py

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss

# ============================
# CONFIG
# ============================

SPLIT_DATE = "2023-01-01"
EXPORT_PATH = "models/r5_short_final_v1.1.pkl"

FINAL_FEATURES = [
    "whale_volume_ratio_delta_3d",
    "btc_ret_lag1",
    "eth_ret_lag1",
    "vol_ratio",
    "eth_btc_ratio",
    "eth_btc_ratio_ma7",
    "withdrawal_tx_count",
    "mega_whale_ratio",
    "non_exchange_tx_count",
    "deposit_withdrawal_ratio",
    "btc_vol30",
    "btc_vol7",
    "exchange_volume_ratio",
    "std_whale_tx_size_eth"
]

# ============================
# MODEL WRAPPER CLASS
# ============================

class CalibratedModelWrapper:
    def __init__(self, model, features):
        self.model = model
        self.feature_names_ = features
        self.classes_ = model.classes_

    def predict_proba(self, X):
        return self.model.predict_proba(X)

# ============================
# MAIN TRAINING FUNCTION
# ============================

def train_short_model(df):
    """
    Train SHORT model for R3/R5 regimes
    """
    print("\n" + "="*70)
    print("TRAINING SHORT MODEL (R3/R5)")
    print("="*70)
    
    # Check if df is passed, otherwise load it
    if df is None:
        from loader.pipeline import load_and_prepare_data
        from loader.targets import create_targets_two_tier
        df = load_and_prepare_data()
        df = create_targets_two_tier(df)
    
    df = df.dropna(subset=FINAL_FEATURES + ["target_t2", "regime_code"])

    # ============================
    # FILTER SHORT REGIMES
    # ============================
    df = df[df["regime_code"].isin(["R3", "R5"])].copy()
    print(f"✅ SHORT regime rows: {len(df)}")

    # ============================
    # TIME SPLIT (LEAKAGE SAFE)
    # ============================
    train_df = df[df["block_date"] < SPLIT_DATE]
    val_df   = df[df["block_date"] >= SPLIT_DATE]

    X_train = train_df[FINAL_FEATURES]
    X_val   = val_df[FINAL_FEATURES]

    y_train = (train_df["target_t2"] == -1).astype(int)
    y_val   = (val_df["target_t2"] == -1).astype(int)

    print(f"📅 Train: {len(X_train)} | Val: {len(X_val)}")
    print(f"📊 Positive rate: {y_train.mean():.1%} train, {y_val.mean():.1%} val")

    # ============================
    # TRAIN BASE MODEL
    # ============================
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=6,
        min_samples_leaf=30,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)

    # ============================
    # CALIBRATION (TRAIN ONLY)
    # ============================
    rf_cal = CalibratedClassifierCV(
        rf,
        method="isotonic",
        cv=5
    )

    rf_cal.fit(X_train, y_train)

    # ============================
    # VALIDATION METRICS
    # ============================
    val_probs = rf_cal.predict_proba(X_val)[:, 1]

    auc = roc_auc_score(y_val, val_probs)
    brier = brier_score_loss(y_val, val_probs)

    print(f"📈 SHORT RF AUC: {auc:.4f}")
    print(f"📉 SHORT RF Brier: {brier:.4f}")

    # ============================
    # EXPORT WITH METADATA
    # ============================
    export = {
        "model": CalibratedModelWrapper(rf_cal, FINAL_FEATURES),
        "features": FINAL_FEATURES,
        "regime_gate": ["R3", "R5"],
        "r3_threshold": 0.65,
        "r5_threshold": 0.55,
        "version": "1.1.0",
        "trained_at": pd.Timestamp.now().isoformat(),
        "metrics": {
            "auc": auc,
            "brier": brier
        }
    }

    joblib.dump(export, EXPORT_PATH)
    print(f"✅ Model exported → {EXPORT_PATH}")
    
    return export

# ============================
# STANDALONE EXECUTION
# ============================

if __name__ == "__main__":
    # Run as standalone script
    from loader.pipeline import load_and_prepare_data
    from loader.targets import create_targets_two_tier
    
    print("🚀 Running SHORT model training as standalone script...")
    
    df = load_and_prepare_data()
    df = create_targets_two_tier(df)
    
    result = train_short_model(df)
    
    print("\n🎯 SHORT model training complete!")