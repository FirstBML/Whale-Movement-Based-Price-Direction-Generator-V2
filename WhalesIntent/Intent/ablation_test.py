import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PHASE 1: ABLATION TESTING FRAMEWORK
# ============================================================================

class AblationTester:
    """Test feature group contributions via ablation study"""
    
    def __init__(self, df, target_col='next_day_price_direction'):
        self.df = df.copy()
        self.target_col = target_col
        self.results = {}
        
    def define_feature_groups(self):
        """Define Model A-C feature sets"""
        
        # Model A: Price-only baseline
        self.groups = {
            'A_price_only': [
                'eth_daily_return', 'eth_log_return', 'eth_vol7', 'eth_vol30',
                'eth_rsi', 'eth_ret_lag1', 'eth_ret_lag3', 'eth_ret_lag7',
                'btc_daily_return', 'btc_log_return', 'btc_vol7', 'btc_vol30',
                'btc_rsi', 'btc_ret_lag1', 'btc_ret_lag3', 'btc_ret_lag7',
                'eth_btc_ratio', 'eth_btc_ratio_ma7', 'eth_btc_corr_30d',
                'eth_outperformance'
            ],
            
            # Model B: On-chain only
            'B_onchain_only': [
                'deposit_tx_count', 'withdrawal_tx_count', 'deposit_withdrawal_ratio',
                'exchange_volume_ratio', 'exchange_flow_share', 'net_exchange_flow_ratio',
                'whale_exchange_deposits_eth', 'whale_exchange_withdrawals_eth',
                'whale_net_exchange_flow_eth', 'whale_exchange_flow_ratio',
                'whale_exchange_asymmetry', 'whale_tx_count', 'whale_volume_eth',
                'whale_volume_ratio', 'whale_volume_ratio_delta_1d',
                'whale_volume_ratio_delta_3d', 'whale_tx_zscore_90d',
                'mega_whale_ratio', 'net_flow_ma7', 'tx_per_active_delta_1d',
                'tx_per_active_zscore_90d', 'block_fullness_delta_1d',
                'eth_burned_delta_1d', 'eth_burned_zscore_90d', 'median_gas_delta_1d',
                'median_gas_delta_7d', 'smart_contract_ratio_delta_1d'
            ]
        }
        
        # Model C: Hybrid (A + B, no raw prices)
        self.groups['C_hybrid'] = (
            self.groups['A_price_only'] + 
            self.groups['B_onchain_only']
        )
        
        # Verify features exist for all groups
        for group_name, features in self.groups.items():
            missing = [f for f in features if f not in self.df.columns]
            if missing:
                print(f"⚠️ {group_name} missing: {missing}")
                self.groups[group_name] = [f for f in features if f in self.df.columns]
        
        return self.groups
    
    def prepare_data(self, features):
        """Clean data for modeling"""
        df_clean = self.df[features + [self.target_col, 'block_date']].copy()
        
        # Remove rows with missing target or features
        df_clean = df_clean.dropna(subset=[self.target_col])
        df_clean = df_clean.dropna(subset=features)
        
        # Sort by date
        df_clean = df_clean.sort_values('block_date').reset_index(drop=True)
        
        return df_clean
    
    def time_series_cv(self, X, y, n_splits=5):
        """Walk-forward time series cross-validation"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        scores = {
            'accuracy': [], 'precision': [], 'recall': [], 
            'f1': [], 'roc_auc': []
        }
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Metrics
            scores['accuracy'].append(accuracy_score(y_test, y_pred))
            scores['precision'].append(precision_score(y_test, y_pred, zero_division=0))
            scores['recall'].append(recall_score(y_test, y_pred, zero_division=0))
            scores['f1'].append(f1_score(y_test, y_pred, zero_division=0))
            scores['roc_auc'].append(roc_auc_score(y_test, y_proba))
        
        return {k: np.mean(v) for k, v in scores.items()}
    
    def run_ablation(self, model_name='A_price_only', n_splits=5):
        """Execute ablation test for a feature group"""
        
        features = self.groups[model_name]
        print(f"\n{'='*70}")
        print(f"MODEL {model_name.split('_')[0]}: {model_name.replace('_', ' ').title()}")
        print(f"{'='*70}")
        print(f"Features: {len(features)}")
        print(f"Feature list: {', '.join(features[:5])}...")
        
        # Prepare data
        df_clean = self.prepare_data(features)
        print(f"Clean samples: {len(df_clean)}")
        print(f"Date range: {df_clean['block_date'].min().date()} → "
              f"{df_clean['block_date'].max().date()}")
        
        X = df_clean[features]
        y = df_clean[self.target_col]
        
        # Class distribution
        class_dist = y.value_counts(normalize=True)
        print(f"\nTarget distribution:")
        print(f"  Class 0 (down): {class_dist[0]:.1%}")
        print(f"  Class 1 (up):   {class_dist[1]:.1%}")
        
        # Run CV
        print(f"\nRunning {n_splits}-fold walk-forward CV...")
        scores = self.time_series_cv(X, y, n_splits)
        
        # Store results
        self.results[model_name] = {
            'features': features,
            'n_features': len(features),
            'n_samples': len(df_clean),
            'scores': scores
        }
        
        # Display results
        print(f"\n📊 RESULTS:")
        print(f"  Accuracy:  {scores['accuracy']:.4f}")
        print(f"  Precision: {scores['precision']:.4f}")
        print(f"  Recall:    {scores['recall']:.4f}")
        print(f"  F1 Score:  {scores['f1']:.4f}")
        print(f"  ROC AUC:   {scores['roc_auc']:.4f}")
        
        return scores
    
    def feature_importance_analysis(self, model_name='C_hybrid'):
        """Analyze which features drive predictions"""
        
        if model_name not in self.results:
            print(f"❌ Model {model_name} not tested yet")
            return
        
        features = self.groups[model_name]
        print(f"\n{'='*70}")
        print(f"FEATURE IMPORTANCE: {model_name.upper()}")
        print(f"{'='*70}")
        
        # Prepare data
        df_clean = self.prepare_data(features)
        X = df_clean[features]
        y = df_clean[self.target_col]
        
        # Train on full dataset (for importance, not prediction)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train RandomForest for better feature importance
        print("\n🌲 Training RandomForest for feature importance...")
        rf = RandomForestClassifier(n_estimators=100, max_depth=8, 
                                     random_state=42, n_jobs=-1)
        rf.fit(X_scaled, y)
        
        # Feature importance from RF
        feat_imp = pd.DataFrame({
            'feature': features,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n📊 TOP 15 FEATURES (RandomForest):")
        print(feat_imp.head(15).to_string(index=False))
        
        # Categorize features
        price_feats = [f for f in feat_imp['feature'].values[:15] 
                       if any(x in f for x in ['eth_', 'btc_', 'ratio', 'return', 
                                                'vol', 'rsi', 'lag', 'outperform'])]
        whale_feats = [f for f in feat_imp['feature'].values[:15] 
                       if any(x in f for x in ['whale', 'exchange', 'deposit', 
                                                'withdrawal', 'flow', 'tx_', 'gas', 
                                                'burned', 'mega'])]
        
        print(f"\n🔍 Feature Breakdown (Top 15):")
        print(f"  Price/Technical: {len(price_feats)}")
        print(f"  Whale/On-chain:  {len(whale_feats)}")
        
        if len(whale_feats) >= 3:
            print(f"\n✅ GOOD: Whale features present in top 15:")
            for f in whale_feats:
                imp = feat_imp[feat_imp['feature'] == f]['importance'].values[0]
                print(f"     {f}: {imp:.4f}")
        else:
            print(f"\n⚠️ WARNING: Only {len(whale_feats)} whale features in top 15")
            print("  → Model relies almost entirely on price momentum")
        
        # Permutation importance (more reliable)
        print(f"\n🔀 Computing permutation importance (slower but accurate)...")
        perm_imp = permutation_importance(rf, X_scaled, y, n_repeats=10, 
                                          random_state=42, n_jobs=-1)
        
        perm_df = pd.DataFrame({
            'feature': features,
            'importance': perm_imp.importances_mean
        }).sort_values('importance', ascending=False)
        
        print(f"\n📊 TOP 15 FEATURES (Permutation):")
        print(perm_df.head(15).to_string(index=False))
        
        return feat_imp, perm_df
    
    def compare_models(self):
        """Compare all tested models"""
        if not self.results:
            print("No models tested yet!")
            return
        
        print(f"\n{'='*70}")
        print("ABLATION COMPARISON")
        print(f"{'='*70}")
        
        comparison = pd.DataFrame({
            name: res['scores'] for name, res in self.results.items()
        }).T
        
        comparison['n_features'] = [res['n_features'] for res in self.results.values()]
        comparison = comparison[['n_features', 'accuracy', 'precision', 
                                 'recall', 'f1', 'roc_auc']]
        
        print(comparison.to_string())
        
        # Find best model
        best_acc = comparison['accuracy'].idxmax()
        best_f1 = comparison['f1'].idxmax()
        
        print(f"\n🏆 Best Accuracy: {best_acc} ({comparison.loc[best_acc, 'accuracy']:.4f})")
        print(f"🏆 Best F1 Score: {best_f1} ({comparison.loc[best_f1, 'f1']:.4f})")
        
        return comparison
        """Compare all tested models"""
        if not self.results:
            print("No models tested yet!")
            return
        
        print(f"\n{'='*70}")
        print("ABLATION COMPARISON")
        print(f"{'='*70}")
        
        comparison = pd.DataFrame({
            name: res['scores'] for name, res in self.results.items()
        }).T
        
        comparison['n_features'] = [res['n_features'] for res in self.results.values()]
        comparison = comparison[['n_features', 'accuracy', 'precision', 
                                 'recall', 'f1', 'roc_auc']]
        
        print(comparison.to_string())
        
        # Find best model
        best_acc = comparison['accuracy'].idxmax()
        best_f1 = comparison['f1'].idxmax()
        
        print(f"\n🏆 Best Accuracy: {best_acc} ({comparison.loc[best_acc, 'accuracy']:.4f})")
        print(f"🏆 Best F1 Score: {best_f1} ({comparison.loc[best_f1, 'f1']:.4f})")
        
        return comparison


# ============================================================================
# EXECUTION
# ============================================================================

def add_price_features(df, price_col, prefix):
    """Add price-based ML features"""
    df = df.sort_values('block_date').reset_index(drop=True)
    
    df[f'{prefix}_daily_return'] = df[price_col].pct_change()
    df[f'{prefix}_log_return'] = np.log(df[price_col] / df[price_col].shift(1))
    df[f'{prefix}_vol7'] = df[f'{prefix}_daily_return'].rolling(7, min_periods=1).std()
    df[f'{prefix}_vol30'] = df[f'{prefix}_daily_return'].rolling(30, min_periods=1).std()
    
    # RSI
    returns = df[f'{prefix}_daily_return']
    gains = returns.where(returns > 0, 0).rolling(14, min_periods=1).mean()
    losses = -returns.where(returns < 0, 0).rolling(14, min_periods=1).mean()
    rs = gains / (losses + 1e-10)
    df[f'{prefix}_rsi'] = 100 - (100 / (1 + rs))
    
    # Lags
    for lag in [1, 3, 7]:
        df[f'{prefix}_ret_lag{lag}'] = df[f'{prefix}_daily_return'].shift(lag)
    
    return df

def add_correlation_features(df):
    """Add ETH-BTC correlation features"""
    df['eth_btc_ratio'] = df['eth_price'] / df['btc_price']
    df['eth_btc_ratio_ma7'] = df['eth_btc_ratio'].rolling(7, min_periods=1).mean()
    df['eth_btc_corr_30d'] = df['eth_daily_return'].rolling(30, min_periods=20).corr(df['btc_daily_return'])
    df['eth_outperformance'] = df['eth_daily_return'] - df['btc_daily_return']
    return df

def create_target(df):
    """Create target: next day price direction"""
    df['next_day_return'] = df['eth_price'].pct_change().shift(-1)
    df['next_day_price_direction'] = (df['next_day_return'] > 0).astype(int)
    return df


if __name__ == "__main__":
    # Load merged dataset
    print("📂 Loading merged dataset...")
    df_merged = pd.read_csv('merged_ml_dataset.csv')
    df_merged['block_date'] = pd.to_datetime(df_merged['block_date'], utc=True)
    
    print(f"✅ Loaded {len(df_merged)} rows, {len(df_merged.columns)} columns")
    
    # Feature engineering
    print("\n⚙️ Engineering features...")
    df_merged = add_price_features(df_merged, 'eth_price', 'eth')
    df_merged = add_price_features(df_merged, 'btc_price', 'btc')
    df_merged = add_correlation_features(df_merged)
    df_merged = create_target(df_merged)
    
    print(f"✅ Features created: {len(df_merged.columns)} columns")
    print(f"Date range: {df_merged['block_date'].min().date()} → "
          f"{df_merged['block_date'].max().date()}")
    
    # Initialize tester
    tester = AblationTester(df_merged)
    
    # Define feature groups
    tester.define_feature_groups()
    
    # Run Model A: Price-only baseline
    print("\n" + "="*70)
    print("PHASE 1: BASELINE TESTING")
    print("="*70)
    tester.run_ablation('A_price_only', n_splits=5)
    
    # Run Model B: On-chain only
    print("\n" + "="*70)
    print("PHASE 2: ON-CHAIN SIGNAL TESTING")
    print("="*70)
    tester.run_ablation('B_onchain_only', n_splits=5)
    
    # Run Model C: Hybrid
    print("\n" + "="*70)
    print("PHASE 3: HYBRID MODEL (PRICE + ON-CHAIN)")
    print("="*70)
    tester.run_ablation('C_hybrid', n_splits=5)
    
    # Compare results
    comparison = tester.compare_models()
    
    # Final analysis
    print("\n" + "="*70)
    print("FINAL ABLATION ANALYSIS")
    print("="*70)
    
    acc_a = tester.results['A_price_only']['scores']['accuracy']
    acc_b = tester.results['B_onchain_only']['scores']['accuracy']
    acc_c = tester.results['C_hybrid']['scores']['accuracy']
    
    print(f"\n📊 Accuracy Comparison:")
    print(f"  Model A (price):    {acc_a:.2%}")
    print(f"  Model B (on-chain): {acc_b:.2%} ({acc_b - 0.5:.2%} above random)")
    print(f"  Model C (hybrid):   {acc_c:.2%}")
    
    print(f"\n🎯 Incremental Value:")
    c_vs_a = acc_c - acc_a
    print(f"  On-chain adds: {c_vs_a:+.2%} accuracy")
    
    if c_vs_a > 0.01:
        print(f"  ✅ PASS: On-chain signals add {c_vs_a:.2%} value")
    elif c_vs_a > 0:
        print(f"  ⚠️ MARGINAL: Only {c_vs_a:.2%} improvement")
    else:
        print(f"  ❌ FAIL: On-chain adds no value (or hurts)")
    
    # ROC AUC comparison
    auc_c = tester.results['C_hybrid']['scores']['roc_auc']
    auc_a = tester.results['A_price_only']['scores']['roc_auc']
    print(f"\n📈 ROC AUC:")
    print(f"  Model C: {auc_c:.4f} vs Model A: {auc_a:.4f} ({auc_c - auc_a:+.4f})")
    
    print("\n💡 Next Steps:")
    if c_vs_a > 0.01:
        print("  • On-chain signals validated")
        print("  • Proceed to feature importance analysis")
        print("  • Test with RandomForest/XGBoost for non-linear effects")
    else:
        print("  • On-chain signals may be regime-specific")
        print("  • Try interaction features (price_vol * whale_flow)")
        print("  • Consider threshold-based rules instead of ML")
    
    # PHASE 2: Feature Importance Analysis
    print("\n" + "="*70)
    print("PHASE 2: FEATURE DOMINANCE CHECK")
    print("="*70)
    print("Analyzing what Model C actually learned...")
    
    feat_imp_rf, feat_imp_perm = tester.feature_importance_analysis('C_hybrid')
    
    # Final verdict
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    
    whale_in_top10_rf = sum(1 for f in feat_imp_rf['feature'].values[:10] 
                            if any(x in f for x in ['whale', 'exchange', 'deposit', 
                                                     'withdrawal', 'flow', 'tx_', 
                                                     'burned', 'mega']))
    
    whale_in_top10_perm = sum(1 for f in feat_imp_perm['feature'].values[:10] 
                              if any(x in f for x in ['whale', 'exchange', 'deposit', 
                                                       'withdrawal', 'flow', 'tx_', 
                                                       'burned', 'mega']))
    
    print(f"\nWhale features in top 10:")
    print(f"  RandomForest:  {whale_in_top10_rf}/10")
    print(f"  Permutation:   {whale_in_top10_perm}/10")
    
    if whale_in_top10_perm == 0:
        print("\n❌ VERDICT: Model ignores whale data entirely")
        print("   → Whale signals redundant with price momentum")
    elif whale_in_top10_perm <= 2:
        print("\n⚠️ VERDICT: Whale data plays minor role")
        print("   → Try regime-specific modeling or interactions")
    else:
        print("\n✅ VERDICT: Whale data actively used")
        print("   → Performance issue may be feature engineering, not signal")

# ============================================================================
# STEP 1: BASIC FEATURE ENGINEERING (from your original code)
# ============================================================================

def add_price_features(df, price_col, prefix):
    """Add price-based ML features"""
    df = df.sort_values('block_date').reset_index(drop=True)
    
    df[f'{prefix}_daily_return'] = df[price_col].pct_change()
    df[f'{prefix}_log_return'] = np.log(df[price_col] / df[price_col].shift(1))
    df[f'{prefix}_vol7'] = df[f'{prefix}_daily_return'].rolling(7, min_periods=1).std()
    df[f'{prefix}_vol30'] = df[f'{prefix}_daily_return'].rolling(30, min_periods=1).std()
    
    # RSI
    returns = df[f'{prefix}_daily_return']
    gains = returns.where(returns > 0, 0).rolling(14, min_periods=1).mean()
    losses = -returns.where(returns < 0, 0).rolling(14, min_periods=1).mean()
    rs = gains / (losses + 1e-10)
    df[f'{prefix}_rsi'] = 100 - (100 / (1 + rs))
    
    # Lags
    for lag in [1, 3, 7]:
        df[f'{prefix}_ret_lag{lag}'] = df[f'{prefix}_daily_return'].shift(lag)
    
    return df


def add_correlation_features(df):
    """Add ETH-BTC correlation features"""
    df['eth_btc_ratio'] = df['eth_price'] / df['btc_price']
    df['eth_btc_ratio_ma7'] = df['eth_btc_ratio'].rolling(7, min_periods=1).mean()
    df['eth_btc_corr_30d'] = df['eth_daily_return'].rolling(30, min_periods=20).corr(df['btc_daily_return'])
    df['eth_outperformance'] = df['eth_daily_return'] - df['btc_daily_return']
    return df


# ============================================================================
# STEP 2: PHASE 3 ENHANCEMENTS - RELATIVE FEATURES & GATING
# ============================================================================

def add_phase3_features(df):
    """Add Phase 3 enhancements: relative prices, gated momentum, interactions"""
    
    print("\n🔄 Phase 3: Creating enhanced features...")
    
    # 1. RELATIVE PRICE FEATURES (replace absolute prices)
    print("  • Relative price features...")
    for col in ['eth_price', 'btc_price']:
        prefix = col.split('_')[0]
        
        # Z-scores
        mean_90d = df[col].rolling(90, min_periods=30).mean()
        std_90d = df[col].rolling(90, min_periods=30).std()
        df[f'{prefix}_price_zscore_90d'] = (df[col] - mean_90d) / (std_90d + 1e-10)
        
        # Distance from MAs
        ma_20 = df[col].rolling(20, min_periods=10).mean()
        df[f'{prefix}_pct_from_ma20'] = (df[col] - ma_20) / (ma_20 + 1e-10)
        
        ma_50 = df[col].rolling(50, min_periods=20).mean()
        df[f'{prefix}_pct_from_ma50'] = (df[col] - ma_50) / (ma_50 + 1e-10)
    
    # 2. GATED MOMENTUM (the key innovation!)
    print("  • Gated momentum features...")
    df['eth_momentum_valid'] = df['eth_ret_lag1'] * np.sign(df['whale_net_exchange_flow_eth'])
    df['btc_momentum_valid'] = df['btc_ret_lag1'] * np.sign(df['net_exchange_flow_ratio'])
    
    vol_threshold = df['eth_vol7'].quantile(0.5)
    df['eth_momentum_lowvol'] = df['eth_ret_lag1'] * (df['eth_vol7'] < vol_threshold).astype(float)
    
    df['whale_confirms_price'] = (
        np.sign(df['eth_ret_lag1']) == np.sign(df['whale_net_exchange_flow_eth'])
    ).astype(float)
    
    volume_z = (df['whale_volume_eth'] - df['whale_volume_eth'].rolling(30).mean()) / \
               (df['whale_volume_eth'].rolling(30).std() + 1e-10)
    df['momentum_volume_confirmed'] = df['eth_ret_lag1'] * (volume_z > 0.5).astype(float)
    
    # 3. REGIME FEATURES
    print("  • Regime features...")
    vol_75th = df['eth_vol30'].quantile(0.75)
    df['high_vol_regime'] = (df['eth_vol30'] > vol_75th).astype(float)
    
    df['trend_strength'] = abs(df['eth_price'].rolling(20).mean() - df['eth_price'].rolling(50).mean())
    trend_25th = df['trend_strength'].quantile(0.25)
    df['choppy_regime'] = (df['trend_strength'] < trend_25th).astype(float)
    
    corr_median = df['eth_btc_corr_30d'].median()
    df['low_corr_regime'] = (df['eth_btc_corr_30d'] < corr_median).astype(float)
    
    # 4. INTERACTIONS
    print("  • Interaction features...")
    df['vol_x_whale_flow'] = df['eth_vol7'] * df['whale_net_exchange_flow_eth']
    df['momentum_x_exchange_pressure'] = df['eth_ret_lag1'] * df['net_exchange_flow_ratio']
    df['whale_activity_x_vol'] = df['whale_tx_count'] * df['eth_vol30']
    df['gas_x_momentum'] = df['median_gas_delta_1d'] * df['eth_ret_lag1']
    
    print("  ✅ Phase 3 complete!")
    return df


# ============================================================================
# STEP 3: PHASE 4 - CONFIDENCE-WEIGHTED TARGET
# ============================================================================

def add_phase4_target(df):
    """Add Phase 4: confidence-weighted target"""
    print("\n🔄 Phase 4: Creating confidence-weighted target...")
    
    df['next_day_return'] = df['eth_price'].pct_change().shift(-1)
    df['next_day_price_direction'] = (df['next_day_return'] > 0).astype(int)
    df['signal_confidence'] = abs(df['next_day_return'])
    
    confidence_median = df['signal_confidence'].median()
    df['high_confidence_sample'] = (df['signal_confidence'] > confidence_median)
    
    print(f"  ✅ Median move: {confidence_median:.4f}")
    print(f"  ✅ High-conf samples: {df['high_confidence_sample'].sum()}")
    
    return df


# ============================================================================
# STEP 4: FEATURE GROUPS
# ============================================================================

def define_feature_groups():
    """Define enhanced feature groups"""
    
    return {
        'D_relative_price': [
            'eth_price_zscore_90d', 'eth_pct_from_ma20', 'eth_pct_from_ma50',
            'btc_price_zscore_90d', 'btc_pct_from_ma20', 'btc_pct_from_ma50',
            'eth_daily_return', 'eth_log_return', 'eth_vol7', 'eth_vol30',
            'btc_daily_return', 'btc_log_return', 'btc_vol7', 'btc_vol30',
            'eth_rsi', 'btc_rsi',
            'eth_ret_lag1', 'eth_ret_lag3', 'eth_ret_lag7',
            'btc_ret_lag1', 'btc_ret_lag3', 'btc_ret_lag7',
            'eth_btc_ratio', 'eth_btc_ratio_ma7', 'eth_btc_corr_30d', 'eth_outperformance'
        ],
        
        'E_gated_hybrid': [
            'eth_price_zscore_90d', 'eth_pct_from_ma20',
            'btc_price_zscore_90d', 'btc_pct_from_ma20',
            'eth_momentum_valid', 'btc_momentum_valid', 'eth_momentum_lowvol',
            'whale_confirms_price', 'momentum_volume_confirmed',
            'high_vol_regime', 'choppy_regime', 'low_corr_regime',
            'vol_x_whale_flow', 'momentum_x_exchange_pressure',
            'whale_activity_x_vol', 'gas_x_momentum',
            'whale_net_exchange_flow_eth', 'whale_tx_zscore_90d',
            'whale_volume_ratio_delta_3d', 'exchange_flow_share',
            'tx_per_active_delta_1d', 'eth_burned_zscore_90d', 'median_gas_delta_7d'
        ]
    }


# ============================================================================
# STEP 5: ENHANCED ABLATION TESTER
# ============================================================================

def run_confidence_weighted_cv(df, features, n_splits=5):
    """Run CV with confidence-weighted evaluation"""
    
    # Prepare data
    required = features + ['next_day_price_direction', 'signal_confidence', 'block_date']
    df_clean = df[required].dropna().sort_values('block_date').reset_index(drop=True)
    
    X = df_clean[features]
    y = df_clean['next_day_price_direction']
    conf = df_clean['signal_confidence']
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    scores_all = {'acc': [], 'prec': [], 'rec': [], 'f1': [], 'auc': []}
    scores_hc = {'acc': [], 'prec': [], 'rec': [], 'f1': [], 'auc': []}
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        conf_test = conf.iloc[test_idx]
        
        # Scale and train
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)
        
        model = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
        model.fit(X_train_sc, y_train)
        
        y_pred = model.predict(X_test_sc)
        y_proba = model.predict_proba(X_test_sc)[:, 1]
        
        # All samples
        scores_all['acc'].append(accuracy_score(y_test, y_pred))
        scores_all['prec'].append(precision_score(y_test, y_pred, zero_division=0))
        scores_all['rec'].append(recall_score(y_test, y_pred, zero_division=0))
        scores_all['f1'].append(f1_score(y_test, y_pred, zero_division=0))
        scores_all['auc'].append(roc_auc_score(y_test, y_proba))
        
        # High confidence only
        hc_mask = conf_test > conf_test.median()
        if hc_mask.sum() > 10:
            y_test_hc = y_test[hc_mask]
            y_pred_hc = y_pred[hc_mask]
            y_proba_hc = y_proba[hc_mask]
            
            scores_hc['acc'].append(accuracy_score(y_test_hc, y_pred_hc))
            scores_hc['prec'].append(precision_score(y_test_hc, y_pred_hc, zero_division=0))
            scores_hc['rec'].append(recall_score(y_test_hc, y_pred_hc, zero_division=0))
            scores_hc['f1'].append(f1_score(y_test_hc, y_pred_hc, zero_division=0))
            scores_hc['auc'].append(roc_auc_score(y_test_hc, y_proba_hc))
    
    return {
        'all': {k: np.mean(v) for k, v in scores_all.items()},
        'hc': {k: np.mean(v) for k, v in scores_hc.items()},
        'n_samples': len(df_clean)
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    print("="*70)
    print("ENHANCED ML PIPELINE - PHASE 3 & 4")
    print("="*70)
    
    # 1. Load data
    print("\n📂 Loading data...")
    df = pd.read_csv('merged_ml_dataset.csv')
    df['block_date'] = pd.to_datetime(df['block_date'], utc=True)
    print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"   Date range: {df['block_date'].min().date()} → {df['block_date'].max().date()}")
    
    # 2. Add basic features
    print("\n⚙️ Creating basic features...")
    df = add_price_features(df, 'eth_price', 'eth')
    df = add_price_features(df, 'btc_price', 'btc')
    df = add_correlation_features(df)
    
    # 3. Add Phase 3 enhancements
    df = add_phase3_features(df)
    
    # 4. Add Phase 4 target
    df = add_phase4_target(df)
    
    print(f"\n✅ Feature engineering complete: {len(df.columns)} columns")
    
    # 5. Define feature groups
    groups = define_feature_groups()
    
    # Verify features exist
    print("\n🔍 Verifying features...")
    for name, feats in groups.items():
        missing = [f for f in feats if f not in df.columns]
        if missing:
            print(f"  ⚠️ {name} missing {len(missing)} features: {missing[:3]}...")
            groups[name] = [f for f in feats if f in df.columns]
        print(f"  ✅ {name}: {len(groups[name])} features available")
    
    # 6. Run ablation tests
    results = {}
    
    print("\n" + "="*70)
    print("MODEL D: RELATIVE PRICE BASELINE")
    print("="*70)
    results['D'] = run_confidence_weighted_cv(df, groups['D_relative_price'])
    print(f"\nAll Samples:     Acc={results['D']['all']['acc']:.4f}, AUC={results['D']['all']['auc']:.4f}")
    print(f"High-Confidence: Acc={results['D']['hc']['acc']:.4f}, AUC={results['D']['hc']['auc']:.4f}")
    print(f"Lift:            {(results['D']['hc']['acc'] - results['D']['all']['acc'])*100:+.2f}%")
    
    print("\n" + "="*70)
    print("MODEL E: GATED HYBRID (THE FIX)")
    print("="*70)
    results['E'] = run_confidence_weighted_cv(df, groups['E_gated_hybrid'])
    print(f"\nAll Samples:     Acc={results['E']['all']['acc']:.4f}, AUC={results['E']['all']['auc']:.4f}")
    print(f"High-Confidence: Acc={results['E']['hc']['acc']:.4f}, AUC={results['E']['hc']['auc']:.4f}")
    print(f"Lift:            {(results['E']['hc']['acc'] - results['E']['all']['acc'])*100:+.2f}%")
    
    # 7. Final comparison
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    
    d_hc = results['D']['hc']['acc']
    e_hc = results['E']['hc']['acc']
    value = e_hc - d_hc
    
    print(f"\nHigh-Confidence Accuracy:")
    print(f"  Model D (Relative Price): {d_hc:.2%}")
    print(f"  Model E (Gated Hybrid):   {e_hc:.2%}")
    print(f"\n🎯 ON-CHAIN VALUE: {value:+.2%}")
    
    if value > 0.02:
        print("\n✅ SUCCESS! On-chain signals add significant value")
        print("   → Proceed with Model E for production")
        print("   → Use high-confidence filtering for trading")
    elif value > 0:
        print("\n⚠️ MARGINAL: Small improvement detected")
        print("   → May be regime-specific")
        print("   → Try splitting by market conditions")
    else:
        print("\n❌ FAILURE: Gating didn't solve the problem")
        print("   → On-chain may lag price")
        print("   → Try leading indicators or regime-split models")
    
    print("\n" + "="*70)
    print("COMPARISON TABLE")
    print("="*70)
    
    comparison = pd.DataFrame({
        'Model D': [results['D']['all']['acc'], results['D']['hc']['acc'], 
                    results['D']['hc']['acc'] - results['D']['all']['acc']],
        'Model E': [results['E']['all']['acc'], results['E']['hc']['acc'],
                    results['E']['hc']['acc'] - results['E']['all']['acc']]
    }, index=['All Samples', 'High-Confidence', 'Lift'])
    
    print(comparison.to_string())
    
    print("\n💾 Save enhanced dataset? Uncomment below:")
    print("# df.to_csv('enhanced_ml_dataset.csv', index=False)")