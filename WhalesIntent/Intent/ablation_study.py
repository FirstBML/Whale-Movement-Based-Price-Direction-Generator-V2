"""
ETH Whale Ablation Study
Quantify incremental predictive value of on-chain data beyond price
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# FEATURE GROUP DEFINITIONS
# ============================================================================

def get_feature_groups(X):
    """Define clean, non-overlapping feature groups"""
    
    # 🅰️ PRICE-ONLY: Everything derivable from prices alone
    price_features = [
        # ETH
        'eth_log_return_lag1', 'eth_log_return_lag3', 'eth_log_return_lag7',
        'eth_vol7', 'eth_vol30', 'eth_rsi',
        'eth_price_to_ma7', 'eth_price_to_ma30',
        # BTC
        'btc_log_return_lag1', 'btc_log_return_lag3', 'btc_log_return_lag7',
        'btc_vol7', 'btc_vol30', 'btc_rsi',
        'btc_price_to_ma7', 'btc_price_to_ma30',
        # Relative
        'eth_btc_ratio', 'eth_btc_ratio_ma7', 'eth_btc_ratio_ma30',
        'eth_btc_corr_30d', 'eth_outperformance_lag1', 'eth_outperformance_ma7'
    ]
    
    # 🅱️ ON-CHAIN-ONLY: No price, no returns, no BTC
    onchain_features = [c for c in X.columns if c not in price_features and c not in 
        ['block_date', 'eth_price', 'btc_price', 'btc_ma7', 'btc_ma30', 'eth_ma7', 'eth_ma30']]
    
    # 🅲 HYBRID: Price + On-chain
    hybrid_features = price_features + onchain_features
    
    # Filter to existing columns
    price_features = [f for f in price_features if f in X.columns]
    onchain_features = [f for f in onchain_features if f in X.columns]
    hybrid_features = [f for f in hybrid_features if f in X.columns]
    
    return {
        'price': price_features,
        'onchain': onchain_features,
        'hybrid': hybrid_features
    }

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Evaluate model on test set"""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_proba),
        'precision_up': precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        'recall_up': recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        'f1_up': f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    }

def run_ablation(X, y, feature_groups, model_name='price', n_splits=5, random_state=42):
    """Run time-series cross-validation for one feature group"""
    
    features = feature_groups[model_name]
    if not features:
        print(f"⚠️ No features found for {model_name}")
        return None
    
    X_subset = X[features].fillna(0)  # Handle NaN
    
    # Time-series CV
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_subset), 1):
        X_train, X_test = X_subset.iloc[train_idx], X_subset.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Same model, same hyperparameters
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=random_state
        )
        
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
        results.append(metrics)
        
        print(f"  Fold {fold}: Acc={metrics['accuracy']:.3f}, AUC={metrics['auc']:.3f}")
    
    # Average across folds
    avg_results = {k: np.mean([r[k] for r in results]) for k in results[0].keys()}
    return avg_results

# ============================================================================
# ABLATION MATRIX
# ============================================================================

def create_ablation_matrix(X, y, n_splits=5, random_state=42):
    """Generate complete ablation matrix"""
    
    print("\n" + "="*60)
    print("FEATURE GROUP DEFINITIONS")
    print("="*60)
    
    feature_groups = get_feature_groups(X)
    
    print(f"🅰️ Price-only: {len(feature_groups['price'])} features")
    print(f"🅱️ On-chain-only: {len(feature_groups['onchain'])} features")
    print(f"🅲 Hybrid: {len(feature_groups['hybrid'])} features")
    
    print("\n" + "="*60)
    print("RUNNING ABLATION STUDY")
    print("="*60)
    
    results = {}
    
    # Price-only
    print("\n🅰️ PRICE-ONLY MODEL")
    results['price'] = run_ablation(X, y, feature_groups, 'price', n_splits, random_state)
    
    # On-chain-only
    print("\n🅱️ ON-CHAIN-ONLY MODEL")
    results['onchain'] = run_ablation(X, y, feature_groups, 'onchain', n_splits, random_state)
    
    # Hybrid
    print("\n🅲 HYBRID MODEL")
    results['hybrid'] = run_ablation(X, y, feature_groups, 'hybrid', n_splits, random_state)
    
    # Create matrix
    print("\n" + "="*60)
    print("ABLATION MATRIX")
    print("="*60)
    
    df_results = pd.DataFrame({
        'Model': ['Price-only', 'On-chain-only', 'Hybrid'],
        'Accuracy': [results['price']['accuracy'], results['onchain']['accuracy'], results['hybrid']['accuracy']],
        'AUC': [results['price']['auc'], results['onchain']['auc'], results['hybrid']['auc']],
        'Precision (Up)': [results['price']['precision_up'], results['onchain']['precision_up'], results['hybrid']['precision_up']],
        'Recall (Up)': [results['price']['recall_up'], results['onchain']['recall_up'], results['hybrid']['recall_up']],
        'F1 (Up)': [results['price']['f1_up'], results['onchain']['f1_up'], results['hybrid']['f1_up']]
    })
    
    print(df_results.to_string(index=False))
    
    # Incremental value tests
    print("\n" + "="*60)
    print("INCREMENTAL VALUE TESTS")
    print("="*60)
    
    delta1 = results['hybrid']['accuracy'] - results['price']['accuracy']
    delta2 = results['onchain']['accuracy'] - 0.50
    
    print(f"Δ₁ (Hybrid - Price): {delta1:+.4f} ({delta1*100:+.2f}%)")
    print(f"Δ₂ (On-chain - Chance): {delta2:+.4f} ({delta2*100:+.2f}%)")
    
    # Interpretation
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    
    if delta1 > 0.01 and results['hybrid']['precision_up'] > results['price']['precision_up']:
        print("✅ SUCCESS: On-chain adds real alpha")
        print("   → Move to trading logic design")
    elif 0.003 <= delta1 <= 0.01:
        print("⚠️ WEAK BUT USABLE: On-chain is regime-dependent")
        print("   → Use during high volatility or whale-dominant regimes")
    else:
        print("❌ FAILURE: On-chain is descriptive, not predictive")
        print("   → Useful only for narratives, risk filters, trade vetoes")
    
    # Save results
    df_results.to_csv('data/ablation_results.csv', index=False)
    
    # Save deltas
    pd.DataFrame({
        'Test': ['Δ₁ (Hybrid - Price)', 'Δ₂ (On-chain - Chance)'],
        'Value': [delta1, delta2],
        'Percentage': [delta1*100, delta2*100]
    }).to_csv('data/ablation_deltas.csv', index=False)
    
    print("\n💾 Saved: data/ablation_results.csv, data/ablation_deltas.csv")
    
    return df_results, feature_groups

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("ETH WHALE ABLATION STUDY")
    print("Quantify incremental value of on-chain data")
    print("="*60)
    
    # Load ML-ready dataset
    print("\n📂 Loading ML-ready dataset...")
    df = pd.read_csv('data/ml_ready_dataset.csv')
    
    X = df.drop(columns=['target', 'block_date'], errors='ignore')
    y = df['target']
    
    print(f"✅ Loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Run ablation
    results, feature_groups = create_ablation_matrix(X, y, n_splits=5, random_state=42)
    
    print("\n✅ Ablation study complete!")