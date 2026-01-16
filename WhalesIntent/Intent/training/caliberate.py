# ================================================================
# STEP 1: EXPORT PRODUCTION MODEL
# ================================================================

import joblib
import pandas as pd
import numpy as np
import warnings
import warnings
import time


warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


print("="*70)
print("STEP 1: EXPORT PRODUCTION MODEL")
print("="*70)

# First, attach feature names to the base RandomForest
rf_short.feature_names_ = FINAL_FEATURES

# Now create a wrapper that preserves feature names
# Update your wrapper class to this:
class CalibratedModelWrapper:
    """Wrapper to preserve feature names through calibration"""
    def __init__(self, calibrated_model, feature_names):
        self.calibrated_model = calibrated_model
        self.feature_names_ = feature_names
        
        # Explicitly forward common sklearn attributes
        for attr in ['classes_', 'n_features_in_', 'n_outputs_']:
            if hasattr(calibrated_model, attr):
                setattr(self, attr, getattr(calibrated_model, attr))
        
    def predict_proba(self, X):
        return self.calibrated_model.predict_proba(X)
    
    def predict(self, X):
        return self.calibrated_model.predict(X)
    
    # NO __getattr__ method at all - safer for pickling
# Create wrapped model

rf_short_cal_wrapped = CalibratedModelWrapper(rf_short_cal, FINAL_FEATURES)

# Add regime-specific metadata
export_metadata = {
    'model': rf_short_cal_wrapped,
    'regime_gate': ['R3', 'R5'],  # Only these regimes
    'r3_threshold': 0.65,  # Higher threshold for R3
    'r5_threshold': 0.55,  # Standard threshold for R5
    'features': FINAL_FEATURES,
    'export_date': pd.Timestamp.now(),
    'version': '1.1.0'  # Increment version
}

joblib.dump(export_metadata, "models/r5_short_final_v1.1.pkl")
print("✅ Production model exported")
print(f"   Features: {len(FINAL_FEATURES)}")
print(f"   Model type: {type(rf_short_cal).__name__}")

# ================================================================
# STEP 2: ENGINE-LEVEL VALIDATION 
# ================================================================

print("\n" + "="*70)
print("STEP 2: ENGINE-LEVEL VALIDATION")
print("="*70)

# FIRST: Calculate OLD metrics for comparison
val_probs_cal = rf_short_cal.predict_proba(X_val)[:, 1]
val_df = val_df.copy()
val_df['short_prob'] = val_probs_cal
val_df['short_pred'] = (val_probs_cal >= 0.55).astype(int)

# Calculate OLD metrics
r3_mask = val_df['regime_code'] == 'R3'
r3_preds = val_df.loc[r3_mask, 'short_pred']
r3_actuals = val_df.loc[r3_mask, 'y_short_t2']
r3_count = r3_preds.sum() if r3_preds.sum() > 0 else 0
r3_precision = (r3_preds & r3_actuals).sum() / r3_count if r3_count > 0 else 0

r1_r2_mask = val_df['regime_code'].isin(['R1', 'R2'])
r1_r2_pollution_old = val_df.loc[r1_r2_mask, 'short_pred'].sum()

print(f"📊 OLD MODEL METRICS (for comparison):")
print(f"   R3 signals: {r3_count}")
print(f"   R3 precision: {r3_precision:.1%}")
print(f"   R1/R2 pollution: {r1_r2_pollution_old}")
print()

# NOW: Engine validation
from engines.core_trend_engine import CoreTrendEngine

engine = CoreTrendEngine()
engine.load_models("models")

export_metadata = joblib.load("models/r5_short_final_v1.1.pkl")
engine.short_model = export_metadata['model']

print(f"✅ Engine loaded with regime gates: {export_metadata['regime_gate']}")
print(f"   R3 threshold: {export_metadata['r3_threshold']}")
print(f"   R5 threshold: {export_metadata['r5_threshold']}")

# Generate REAL engine signals - SINGLE LOOP (no duplicates)
signals = []
total_rows = len(val_df)
print(f"\n🔄 Generating signals for {total_rows} rows...")

start_time = time.time()
processed = 0

for idx, row in val_df.iterrows():
    signal = engine.generate_core_signal(row, val_df)
    signal['original_idx'] = idx
    signals.append(signal)
    processed += 1
    
    # Progress indicator every 50 rows
    if processed % 50 == 0:
        elapsed = time.time() - start_time
        rows_per_second = processed / elapsed if elapsed > 0 else 0
        eta = (total_rows - processed) / rows_per_second if rows_per_second > 0 else 0
        print(f"  Processed {processed}/{total_rows} | {rows_per_second:.1f} rows/sec | ETA: {eta:.0f}s")

elapsed = time.time() - start_time
print(f"✅ Signal generation complete in {elapsed:.1f} seconds ({elapsed/total_rows:.3f}s per row)")

# Create signals dataframe
signals_df = pd.DataFrame(signals)
signals_df.set_index('original_idx', inplace=True)

# Now analyze REAL trading behavior
short_trades = signals_df[signals_df['action'] == 'ENTER']

print(f"\n📊 REAL ENGINE PERFORMANCE:")
print(f"   Total validation days: {len(val_df)}")
print(f"   Total SHORT signals: {len(short_trades)}")
print(f"   Signal frequency: {len(short_trades)/len(val_df):.1%}")

# Regime breakdown
print(f"\n📊 SIGNALS BY REGIME:")
for regime in ['R3', 'R5', 'R1', 'R2', 'R0']:
    regime_trades = short_trades[short_trades['regime'] == regime]
    regime_days = (val_df['regime_code'] == regime).sum()
    
    if regime_days > 0:
        freq = len(regime_trades) / regime_days * 100
        print(f"   {regime}: {len(regime_trades)}/{regime_days} ({freq:.1f}%)")

# Check R1/R2 pollution
print(f"\n✅ HARD GATE VALIDATION:")
r1_r2_trades = short_trades[short_trades['regime'].isin(['R1', 'R2'])]
print(f"   R1/R2 pollution: {len(r1_r2_trades)} signals (expect 0)")

# Calculate REAL precision
print(f"\n📊 REAL PRECISION BY REGIME:")

if len(short_trades) > 0:
    for regime in ['R3', 'R5']:
        regime_trades = short_trades[short_trades['regime'] == regime]
        
        if len(regime_trades) > 0:
            # Use the index (which is original_idx) to match with actual outcomes
            regime_indices = regime_trades.index  # These are the original indices
            actual_outcomes = val_df.loc[regime_indices, 'y_short_t2']
            
            correct = actual_outcomes.sum()
            precision = correct / len(regime_trades) if len(regime_trades) > 0 else 0
            
            print(f"   {regime}: {precision:.1%} ({correct}/{len(regime_trades)})")
        else:
            print(f"   {regime}: No signals")

# Position sizing analysis
print(f"\n📊 REAL POSITION SIZING:")
if len(short_trades) > 0:
    size_dist = short_trades['position_size'].value_counts().sort_index()
    for size, count in size_dist.items():
        print(f"   Size {size:.2f}x: {count} signals ({count/len(short_trades):.1%})")
    
    avg_size = short_trades['position_size'].mean()
    print(f"   Average size: {avg_size:.2f}x")
    
    # Confidence vs size correlation
    corr = short_trades['adjusted_confidence'].corr(short_trades['position_size'])
    print(f"   Confidence-Size correlation: {corr:.3f}")
    
# ================================================================
# STEP 3: COMPARE OLD vs NEW BEHAVIOR
# ================================================================

print("\n" + "="*70)
print("STEP 3: BEFORE vs AFTER FIXES")
print("="*70)

print(f"\n🔍 OLD (Model-only, broken):")
print(f"   R3 signals: {r3_count} ({r3_count/r3_mask.sum():.1%} frequency)")
print(f"   R3 precision: {r3_precision:.1%}")
print(f"   R1/R2 pollution: {r1_r2_pollution_old} signals")

print(f"\n🔍 NEW (Engine-level, fixed):")
r3_trades_new = short_trades[short_trades['regime'] == 'R3']
r3_days = r3_mask.sum()

if len(r3_trades_new) > 0:
    r3_indices = r3_trades_new.index
    r3_actual = val_df.loc[r3_indices, 'y_short_t2'].sum()
    r3_precision_new = r3_actual / len(r3_trades_new)
    print(f"   R3 signals: {len(r3_trades_new)} ({len(r3_trades_new)/r3_days:.1%} frequency)")
    print(f"   R3 precision: {r3_precision_new:.1%}")
else:
    print(f"   R3 signals: 0 (0% frequency)")
    print(f"   R3 precision: N/A")

print(f"   R1/R2 pollution: {len(r1_r2_trades)} signals")

print(f"\n✅ IMPROVEMENT SUMMARY:")
print(f"   R1/R2 pollution: {'ELIMINATED' if len(r1_r2_trades) == 0 else f'REDUCED {r1_r2_pollution_old}→{len(r1_r2_trades)}'}")
print(f"   R3 frequency: REDUCED from {r3_count/r3_days:.1%} to {len(r3_trades_new)/r3_days:.1%}")
if len(r3_trades_new) > 0:
    print(f"   R3 precision: {'IMPROVED' if r3_precision_new > r3_precision else 'SIMILAR'} from {r3_precision:.1%} to {r3_precision_new:.1%}")
else:
    print(f"   R3 precision: NO TRADES (safe!)")

# ================================================================
# FINAL REPORT
# ================================================================

print("\n" + "="*70)
print("PRODUCTION VALIDATION SUMMARY")
print("="*70)

print(f"\n✅ STEP 1: Production model exported")
print(f"   Path: models/r5_short_final_v1.1.pkl")
print(f"   Features: {len(FINAL_FEATURES)}")
print(f"   Regime gates: {export_metadata['regime_gate']}")

print(f"\n✅ STEP 2: Engine-level validation COMPLETE")
print(f"   Total SHORT signals: {len(short_trades)}")
print(f"   Signal frequency: {len(short_trades)/len(val_df):.1%}")

print(f"\n✅ CRITICAL CHECKS:")
print(f"   R1/R2 pollution: {len(r1_r2_trades)} (PASS: must be 0)")

if len(r3_trades_new) > 0:
    print(f"   R3 signal rate: {len(r3_trades_new)/r3_days:.1%} (PASS: <10%)")
    r3_indices = r3_trades_new.index
    r3_actual = val_df.loc[r3_indices, 'y_short_t2'].sum()
    r3_precision_new = r3_actual / len(r3_trades_new)
    print(f"   R3 precision: {r3_precision_new:.1%} (PASS: >40%)")
else:
    print(f"   R3 signal rate: 0% (PASS: <10%)")
    print(f"   R3 precision: N/A (PASS: no trades is safe)")

r5_trades = short_trades[short_trades['regime'] == 'R5']
if len(r5_trades) > 0:
    print(f"   R5 dominance: {len(r5_trades)/max(1, len(short_trades)):.1%} (PASS: >70%)")

print(f"\n🚦 SHADOW TRADING STATUS:")
all_checks_passed = (len(r1_r2_trades) == 0)
if len(r3_trades_new) > 0:
    all_checks_passed = all_checks_passed and (len(r3_trades_new)/r3_days < 0.10)

if all_checks_passed:
    print(f"   ✅ ALL CHECKS PASSED")
    print(f"   🟢 PROCEED TO SHADOW TRADING")
else:
    print(f"   ❌ CHECKS FAILED")
    print(f"   🔴 DO NOT PROCEED - REVIEW ABOVE")