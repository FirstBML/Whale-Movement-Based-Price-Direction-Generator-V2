"""
ETH WHALE ALPHA PIPELINE - UNIFIED SYSTEM
Main Orchestration Script with Modular Engines
"""

import os
import time
import json
import warnings
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from dotenv import load_dotenv
import joblib

warnings.filterwarnings('ignore')
load_dotenv()

# ========== IMPORT MODULAR ENGINES ==========
try:
    from core_trend_engine import CoreTrendEngine, rebuild_core_models
    print("✅ Core Trend Engine imported successfully")
except ImportError as e:
    print(f"❌ Could not import Core Trend Engine: {e}")
    print("   Please create core_trend_engine.py first")
    CoreTrendEngine = None

try:
    from r6_mean_reversion import R6MeanReversionEngine
    print("✅ R6 Mean Reversion Engine imported successfully")
except ImportError:
    print("⚠️  R6 Mean Reversion Engine not available")
    R6MeanReversionEngine = None

try:
    from unified_orchestrator import UnifiedOrchestrator
    print("✅ Unified Orchestrator imported successfully")
except ImportError:
    print("⚠️  Unified Orchestrator not available")
    UnifiedOrchestrator = None

try:
    from shadow_trading import ShadowTrader
    print("✅ Modular shadow trading system imported successfully")
except ImportError as e:
    print(f"❌ Could not import from shadow_trading package: {e}")
    ShadowTrader = None

# ========== CONFIGURATION (MINIMAL - DATA LOADING ONLY) ==========
# Trading parameters
SLIPPAGE = 0.0008
FEES = 0.0004

# Create directories
for d in ['validation', 'backtest', 'models', 'logs/daily', 'validation/r6']:
    os.makedirs(d, exist_ok=True)

# ========== UTILITY FUNCTIONS ==========
def to_utc(ts):
    """Ensure timestamp is UTC"""
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")

def rolling_zscore_safe(series, window=90):
    """FIXED: Shift AFTER calculation to prevent leakage"""
    return ((series - series.rolling(window).mean()) / 
            series.rolling(window).std()).shift(1)

def rolling_feature_safe(series, window, func='mean'):
    """Safe rolling with shift"""
    if func == 'mean':
        return series.rolling(window).mean().shift(1)
    elif func == 'std':
        return series.rolling(window).std().shift(1)
    elif func == 'median':
        return series.rolling(window).median().shift(1)
    return series

# ========== PRICE POSITION HELPERS ==========
def price_not_near_lows(row, df, lookback=90, min_pct=0.25):
    """
    Require price to be above X percentile of recent range
    Only applies to LONG positions
    """
    if pd.isna(row['eth_price']):
        return False
    
    idx = row.name
    start_idx = max(0, idx - lookback)
    recent_prices = df.iloc[start_idx:idx]['eth_price'].values
    
    if len(recent_prices) < 10:
        return True
    
    price_min = np.min(recent_prices)
    price_max = np.max(recent_prices)
    
    if price_max - price_min < 1e-9:
        return True
    
    pct = (row['eth_price'] - price_min) / (price_max - price_min)
    return pct >= min_pct

def price_not_near_highs(row, df, lookback=90, max_pct=0.75):
    """
    Protection for LONG entries against buying local tops
    Only applies to LONG positions
    """
    if pd.isna(row['eth_price']):
        return True
    
    idx = row.name
    start_idx = max(0, idx - lookback)
    recent_prices = df.iloc[start_idx:idx]['eth_price'].values
    
    if len(recent_prices) < 10:
        return True
    
    price_min = np.min(recent_prices)
    price_max = np.max(recent_prices)
    
    if price_max - price_min < 1e-9:
        return True
    
    pct = (row['eth_price'] - price_min) / (price_max - price_min)
    return pct <= max_pct

# ========== DATA LOADING FROM FILES ==========
def load_data_from_files():
    """
    Load data from files saved by data_loader.py
    Now includes funding data
    """
    print("📂 Loading data from saved files...")
    
    files_to_load = {
        'whale_data': 'data/whale_ml_ready.csv',
        'market_intent': 'data/market_intent_ml_ready.csv', 
        'btc_price': 'data/price_cache/btc.csv',
        'eth_price': 'data/price_cache/eth.csv',
        'funding_data': 'data/funding_rates_ml_ready.csv'
    }
    
    loaded_data = {}
    
    for name, filepath in files_to_load.items():
        if os.path.exists(filepath):
            try:
                if 'price' in name:
                    df = pd.read_csv(filepath, parse_dates=["date"])
                    df["date"] = df["date"].apply(to_utc)
                else:
                    df = pd.read_csv(filepath, parse_dates=["block_date"])
                    df["block_date"] = df["block_date"].apply(to_utc)
                
                loaded_data[name] = df
                print(f"✅ Loaded {name}: {len(df)} rows")
                
                # Special handling for funding data
                if name == 'funding_data' and not df.empty:
                    whale_dates = loaded_data.get('whale_data', pd.DataFrame())
                    if not whale_dates.empty:
                        funding_dates = df['block_date']
                        whale_min = whale_dates['block_date'].min()
                        whale_max = whale_dates['block_date'].max()
                        
                        funding_coverage = (
                            funding_dates.min() <= whale_min and
                            funding_dates.max() >= whale_max
                        )
                        
                        if funding_coverage:
                            print(f"   ✅ Full coverage: {whale_min.date()} to {whale_max.date()}")
                        else:
                            print(f"   ⚠️ Partial coverage")
                
            except Exception as e:
                print(f"❌ Failed to load {name}: {e}")
                loaded_data[name] = pd.DataFrame()
        else:
            print(f"❌ {name} file not found: {filepath}")
            if name == 'funding_data':
                print(f"   ⚠️ Funding data missing - system will use zeros")
            loaded_data[name] = pd.DataFrame()
    
    # Check essential data
    essential_data = ['whale_data', 'market_intent', 'btc_price', 'eth_price']
    if all(len(loaded_data[d]) > 0 for d in essential_data):
        print(f"\n✅ Essential data loaded successfully")
    else:
        print(f"\n⚠️  Some essential data files are missing or empty")
        print(f"   Please run data_loader.py to fetch fresh data")
    
    return (
        loaded_data.get('whale_data', pd.DataFrame()),
        loaded_data.get('market_intent', pd.DataFrame()),
        loaded_data.get('btc_price', pd.DataFrame()),
        loaded_data.get('eth_price', pd.DataFrame()),
        loaded_data.get('funding_data', pd.DataFrame())
    )

# ========== FEATURE ENGINEERING ==========
def add_price_features(df, price_col, prefix):
    """Add technical features for a price series"""
    df = df.copy()
    
    # Log returns
    df[f'{prefix}_log_return'] = np.log(df[price_col] / df[price_col].shift(1))
    
    # Lagged returns (including lag 2 for LONG confirmation)
    for lag in [1, 2, 3, 7]:
        df[f'{prefix}_ret_lag{lag}'] = df[f'{prefix}_log_return'].shift(lag)
    
    # Volatility
    df[f'{prefix}_vol7'] = df[f'{prefix}_log_return'].rolling(7).std().shift(1)
    df[f'{prefix}_vol30'] = df[f'{prefix}_log_return'].rolling(30).std().shift(1)
    
    # RSI
    returns = df[f'{prefix}_log_return']
    gains = returns.where(returns > 0, 0).rolling(14).mean()
    losses = -returns.where(returns < 0, 0).rolling(14).mean()
    df[f'{prefix}_rsi'] = (100 - (100 / (1 + gains / (losses + 1e-10)))).shift(1)
    
    return df

def engineer_features(df_whales, df_market_intent, df_btc, df_eth, df_funding=None):
    """Engineer all features with funding data from loader"""
    print("🔧 Engineering features...")
    
    # Merge price data
    df_prices = pd.merge(df_btc, df_eth, on='date', how='outer').sort_values('date')
    
    # Merge with whale data
    df = pd.merge(
        df_whales, 
        df_prices, 
        left_on='block_date', 
        right_on='date', 
        how='left'
    ).drop(columns=['date'])
    
    # Merge with market intent data
    df = pd.merge(
        df, 
        df_market_intent, 
        on='block_date', 
        how='left', 
        suffixes=('', '_intent')
    )
    
    # ========== MERGE FUNDING DATA ==========
    if df_funding is not None and not df_funding.empty:
        df = pd.merge(
            df,
            df_funding[['block_date', 'eth_funding_rate_8h']],
            on='block_date',
            how='left'
        )
        print(f"✅ Merged funding data: {len(df_funding)} rows")
        
        funding_present = df['eth_funding_rate_8h'].notna().sum()
        funding_pct = funding_present / len(df) * 100
        print(f"   Funding coverage: {funding_present}/{len(df)} rows ({funding_pct:.1f}%)")
    else:
        print("⚠️ No funding data - funding column will not be created")
    
    df = df.sort_values('block_date').reset_index(drop=True)
    
    # Add price features
    df = add_price_features(df, 'eth_price', 'eth')
    df = add_price_features(df, 'btc_price', 'btc')
    
    # ETH/BTC ratio features
    df['eth_btc_ratio'] = df['eth_price'] / df['btc_price']
    df['eth_btc_ratio_ma7'] = df['eth_btc_ratio'].rolling(7).mean().shift(1)
    df['eth_btc_corr_30d'] = df['eth_log_return'].shift(1).rolling(30) \
        .corr(df['btc_log_return'].shift(1)).shift(1)
        
    # ========== ADD VOL_RATIO FEATURE ==========
    if 'eth_vol7' in df.columns and 'eth_vol30' in df.columns:
        df['vol_ratio'] = df['eth_vol7'] / df['eth_vol30']
        df['vol_ratio'] = df['vol_ratio'].clip(0.5, 2.0)
        print("✅ Created vol_ratio feature")
    else:
        print("⚠️  Could not create vol_ratio")
        df['vol_ratio'] = 1.0
            
    # Apply safe rolling z-scores
    zscore_pairs = [
        ('whale_tx_count', 'whale_tx_zscore_90d'),
        ('tx_per_active', 'tx_per_active_zscore_90d'),
        ('eth_burned', 'eth_burned_zscore_90d'),
        ('exchange_volume', 'exchange_volume_zscore'),
    ]
    
    for raw_col, zscore_col in zscore_pairs:
        if raw_col in df.columns:
            df[zscore_col] = rolling_zscore_safe(df[raw_col], 90)
    
    # Burn/issuance ratio
    if all(col in df.columns for col in ['eth_burned', 'total_gas_fees']):
        df['burn_issuance_ratio'] = (df['eth_burned'] / (df['total_gas_fees'] + 1e-10)).shift(1)
    
    # Whale volume deltas
    if 'whale_volume_ratio' in df.columns:
        df['whale_volume_ratio_delta_1d'] = df['whale_volume_ratio'].diff(1).shift(1)
        df['whale_volume_ratio_delta_3d'] = df['whale_volume_ratio'].diff(3).shift(1)
    
    # Clean up intermediate columns
    df = df.drop(columns=['eth_log_return', 'btc_log_return'], errors='ignore')
    
    # Save engineered features
    df.to_csv('data/features_engineered.csv', index=False)
    print(f"✅ Features engineered: {len(df.columns)} columns, {len(df)} rows")
    
    return df

# ========== TARGET CREATION ==========
def create_targets_two_tier(df, k=1.5):
    """
    Create two-tier SHORT labels (crash + breakdown)
    """
    print("🎯 Creating two-tier targets...")
    
    df = df.sort_values('block_date').reset_index(drop=True).copy()
    
    # Calculate returns
    df['eth_log_return'] = np.log(df['eth_price'] / df['eth_price'].shift(1))
    df['rolling_vol_30'] = df['eth_log_return'].rolling(30, min_periods=10).std()
    
    # T+2 returns and threshold
    df['return_t2'] = df['eth_log_return'].rolling(2).sum().shift(-2)
    
    # Dynamic threshold using 65th percentile
    df['threshold_t2'] = df['rolling_vol_30'].rolling(60, min_periods=20).quantile(0.65)
    
    # Tier 1: Crash (hard down)
    hard_down = (
        (df['return_t2'] < -df['threshold_t2']) &
        (df['eth_vol7'] > df['eth_vol30']).fillna(False)
    )
    
    # Tier 2: Breakdown (pre-crash)
    exchange_flow_median = df['exchange_flow_share'].rolling(90, min_periods=30).median()
    
    soft_down = (
        (df['eth_ret_lag1'].fillna(0) < 0) &
        (df['btc_ret_lag1'].fillna(0) < 0) &
        (df['whale_volume_ratio_delta_3d'].fillna(0) > 0) &
        (df['exchange_flow_share'] > exchange_flow_median).fillna(False)
    )
    
    # Create targets
    df['target_t2'] = 0
    df.loc[df['return_t2'] > df['threshold_t2'], 'target_t2'] = 1  # UP
    df.loc[hard_down | soft_down, 'target_t2'] = -1  # DOWN (both tiers)
    
    # Create binary targets
    df['y_long_t2'] = (df['target_t2'] == 1).astype(int)
    df['y_short_t2'] = (df['target_t2'] == -1).astype(int)
    
    # Clean up
    df = df.drop(columns=['eth_log_return'], errors='ignore')
    
    # Print distribution
    print("\n📊 Target Distribution (Two-Tier SHORT):")
    for state, label in [(-1, 'DOWN'), (0, 'FLAT'), (1, 'UP')]:
        count = (df['target_t2'] == state).sum()
        percentage = count / len(df) * 100
        print(f"  {label:5s}: {count:4d} ({percentage:5.1f}%)")
    
    hard_count = hard_down.sum()
    soft_count = soft_down.sum()
    total_down = (df['target_t2'] == -1).sum()
    
    print(f"\n  Tier 1 (crash):     {hard_count:4d}")
    print(f"  Tier 2 (breakdown): {soft_count:4d}")
    print(f"  Total DOWN:         {total_down:4d}")
    
    return df

# ========== REGIME DEFINITION ==========
def define_regimes_extended(df):
    """Define trading regimes including R5 distribution regime"""
    print("📈 Defining extended regimes...")
    
    if 'btc_ret_lag1' not in df.columns or 'eth_vol7' not in df.columns:
        df['regime_code'] = 'R0'
        return df
    
    # Standard regimes based on BTC trend and ETH volatility
    btc_trend_7d = df['btc_ret_lag1'].rolling(7).mean()
    df['btc_regime'] = pd.cut(
        btc_trend_7d, 
        bins=[-np.inf, -0.005, 0.005, np.inf], 
        labels=['DOWN', 'FLAT', 'UP']
    )
    
    vol_median = df['eth_vol7'].rolling(180, min_periods=60).median()
    df['vol_regime'] = (df['eth_vol7'] > vol_median).map({True: 'HIGH', False: 'LOW'})
    
    # Combine for standard regimes
    df['regime'] = df['btc_regime'].astype(str) + '_' + df['vol_regime'].astype(str)
    regime_map = {
        'UP_HIGH': 'R1',    # Bull high vol
        'UP_LOW': 'R2',     # Bull low vol
        'DOWN_HIGH': 'R3',  # Bear high vol
        'DOWN_LOW': 'R4',   # Bear low vol
    }
    df['regime_code'] = df['regime'].map(regime_map).fillna('R0')
    
    # R5: Whale distribution regime
    exchange_flow_median = df['exchange_flow_share'].rolling(60, min_periods=20).median()
    
    df['dist_regime'] = (
        (df['whale_volume_ratio_delta_3d'].fillna(0) > 0) &
        (df['exchange_flow_share'] > exchange_flow_median).fillna(False)
    )
    
    # Override with R5 where distribution regime is active
    df.loc[df['dist_regime'], 'regime_code'] = 'R5'
    
    # Print regime distribution
    print("\n📊 Extended Regime Distribution:")
    regime_stats = []
    for code in ['R1', 'R2', 'R3', 'R4', 'R5', 'R0']:
        count = (df['regime_code'] == code).sum()
        if len(df) > 0:
            pct = count / len(df) * 100
            icon = '🟢' if code == 'R1' else ('🔴' if code in ['R3', 'R5'] else '⚪')
            regime_stats.append(f"{icon} {code}: {count:4d} ({pct:5.1f}%)")
    
    # Print in two columns
    for i in range(0, len(regime_stats), 2):
        row = regime_stats[i:i+2]
        print("  " + " | ".join(row))
    
    return df

# ========== BUILD COMPLETE PIPELINE ==========
def build_pipeline_complete(df_features):
    """
    Create the complete pipeline dataset with features, targets, and regimes
    """
    print("\n" + "="*70)
    print("BUILDING COMPLETE PIPELINE DATASET")
    print("="*70)
    
    # Create targets
    df_with_targets = create_targets_two_tier(df_features)
    
    # Define regimes
    df_complete = define_regimes_extended(df_with_targets)
    
    # Ensure all required features exist (check with core engine if available)
    if CoreTrendEngine:
        # Get feature lists from core engine
        from core_trend_engine import LONG_FEATURES, SHORT_FEATURES
        for feature in LONG_FEATURES + SHORT_FEATURES:
            if feature not in df_complete.columns:
                df_complete[feature] = 0.0
    
    # Fill NaN values for features
    feature_cols = [col for col in df_complete.columns if col not in 
                   ['block_date', 'target_t2', 'y_long_t2', 'y_short_t2', 
                    'regime_code', 'btc_regime', 'vol_regime', 'regime', 'dist_regime']]
    
    df_complete[feature_cols] = df_complete[feature_cols].fillna(method='ffill').fillna(0)
    
    # Save complete pipeline
    df_complete.to_csv('data/pipeline_complete.csv', index=False)
    
    # Report statistics
    print(f"\n✅ Pipeline complete saved:")
    print(f"   Rows: {len(df_complete)}")
    print(f"   Columns: {len(df_complete.columns)}")
    print(f"   Date range: {df_complete['block_date'].min().date()} to {df_complete['block_date'].max().date()}")
    print(f"   File: data/pipeline_complete.csv")
    
    return df_complete

# ========== STEP 4: UPDATED FUNCTIONS ==========

def rebuild_models_if_needed(df_pipeline):
    """
    STEP 4: Rebuild models using core engine
    """
    if CoreTrendEngine is None:
        print("❌ Core Trend Engine not available")
        return None
    
    print("\n🔧 Rebuilding models using Core Engine...")
    engine = rebuild_core_models(df_pipeline)
    return engine

def generate_unified_signal(row, df, short_model=None, long_model=None):
    """
    STEP 4: Legacy wrapper - use CoreTrendEngine instead
    """
    if CoreTrendEngine is None:
        print("❌ Core Trend Engine not available")
        return {
            "date": str(row.get('block_date', 'unknown')),
            "action": "NO_TRADE",
            "reasons": ["core_engine_unavailable"]
        }
    
    # Create engine instance
    engine = CoreTrendEngine()
    if short_model:
        engine.short_model = short_model
    if long_model:
        engine.long_model = long_model
    
    # Generate signal using core engine
    return engine.generate_core_signal(row, df)

def generate_daily_signal_unified(df, short_model=None, long_model=None):
    """
    Generate unified daily signal (uses latest row)
    """
    latest_row = df.iloc[-1].copy()
    return generate_unified_signal(latest_row, df, short_model, long_model)

# ========== SHADOW TRADING FUNCTIONS ==========
def run_90_day_shadow_trading():
    """
    Run 90-day shadow trading with MAE/MFE logging
    Now using Core Trend Engine
    """
    print("\n" + "="*70)
    print("90-DAY SHADOW TRADING INITIATED")
    print("="*70)
    
    if ShadowTrader is None:
        print("❌ Shadow trading module not available")
        return []
    
    # Load pipeline data
    if not os.path.exists('data/pipeline_complete.csv'):
        print("❌ Pipeline data not found. Run unified pipeline first.")
        return []
    
    df = pd.read_csv('data/pipeline_complete.csv', parse_dates=['block_date'])
    df = df.sort_values('block_date')
    
    # Rebuild models using core engine
    engine = rebuild_models_if_needed(df)
    
    if not engine or not engine.long_model or not engine.short_model:
        print("❌ Could not load or rebuild models")
        return []
    
    # Initialize shadow trader
    trader = ShadowTrader()
    
    # Use last 90 days
    available_days = len(df)
    forward_window = 48
    
    if available_days < 90:
        print(f"⚠️  Limited data: Only {available_days} days available")
        test_days = available_days
        forward_window = min(30, available_days // 3)
    else:
        test_days = 90
    
    # Get test period
    start_idx = max(0, len(df) - test_days - forward_window)
    end_idx = len(df) - forward_window
    
    if start_idx >= end_idx:
        print(f"❌ Insufficient data for shadow trading")
        return []
    
    test_period = df.iloc[start_idx:end_idx].copy()
    
    print(f"\n📅 Running shadow trading on {len(test_period)} days:")
    print(f"   Date range: {test_period['block_date'].min().date()} to {test_period['block_date'].max().date()}")
    
    # Generate and log signals
    signals_logged = 0
    for i, row in test_period.iterrows():
        signal = engine.generate_core_signal(row, df)
        
        # Log ENTER signals only
        if signal['action'] == 'ENTER':
            trade = trader.log_trade(signal, df, trade_days=forward_window)
            if trade:
                signals_logged += 1
                direction_icon = "🟢" if signal['direction'] == 'LONG' else "🔴"
                print(f"{direction_icon} Logged {signal['date']}: {signal['direction']} @ ${row['eth_price']:.0f} "
                      f"(conf: {signal['adjusted_confidence']:.2f}, size: {signal['position_size']:.2f})")
    
    # Save and analyze
    if signals_logged > 0:
        trader.save_trades()
        
        try:
            report = trader.get_performance_report()
            if report and "error" not in report:
                print("\n📊 MODULAR PERFORMANCE REPORT:")
                print("-" * 70)
                print(f"Total trades: {report.get('total_trades', 0)}")
                print(f"Long trades: {report.get('long_trades', 0)}")
                print(f"Short trades: {report.get('short_trades', 0)}")
                print(f"Average MAE: {report.get('avg_mae', 0):.2f}%")
                print(f"Average MFE: {report.get('avg_mfe', 0):.2f}%")
        except Exception as e:
            print(f"⚠️  Error generating report: {e}")
            
    else:
        print("⚠️  No ENTER signals logged in shadow trading period")
        try:
            trader.save_trades()
        except:
            print("   Could not save empty trades file")
    
    return trader.trades if hasattr(trader, 'trades') else []

# ========== MANUAL INSPECTION FUNCTIONS ==========
def export_signals_for_manual_inspection(df, engine=None, days=60):
    """
    Export signals for manual inspection with all necessary columns
    """
    print(f"\n📋 Exporting {days} days of signals for manual inspection...")
    
    recent_data = df.iloc[-days:].copy()
    signals = []
    
    # Use core engine if available
    if engine is None and CoreTrendEngine:
        engine = CoreTrendEngine()
        engine.load_models()
    
    for idx, row in recent_data.iterrows():
        # Generate signal using appropriate engine
        if engine:
            signal = engine.generate_core_signal(row, df)
        else:
            signal = generate_unified_signal(row, df)
        
        # Get price position info
        near_lows = price_not_near_lows(row, df, lookback=90, min_pct=0.25)
        near_highs = not price_not_near_highs(row, df, lookback=90, max_pct=0.75)
        
        # Create detailed record
        record = {
            'date': signal['date'],
            'regime': signal['regime'],
            'direction': signal['direction'],
            'action': signal['action'],
            'model_probability': signal.get('model_probability', 0),
            'adjusted_confidence': signal.get('adjusted_confidence', 0),
            'position_size': signal.get('position_size', 0),
            'reasons': '|'.join(signal['reasons']) if signal['reasons'] else '',
            'eth_price': row['eth_price'],
            'btc_ret_lag1': row.get('btc_ret_lag1', 0),
            'eth_ret_lag1': row.get('eth_ret_lag1', 0),
            'vol_ratio': row.get('vol_ratio', 1),
            'price_near_lows': near_lows,
            'price_near_highs': near_highs,
            'mae_survivable_check': 'NEEDS_CHECK',
            'signal_quality': 'NEEDS_REVIEW',
            'comments': ''
        }
        
        signals.append(record)
    
    # Create DataFrame
    df_signals = pd.DataFrame(signals)
    
    # Save to CSV
    output_file = 'validation/manual_inspection_signals.csv'
    df_signals.to_csv(output_file, index=False)
    
    print(f"✅ Exported {len(df_signals)} signals to {output_file}")
    return df_signals

def create_manual_review_template():
    """Create a template for manual review"""
    template_content = """LONG SIGNAL MANUAL REVIEW TEMPLATE

Date: __________
Reviewer: __________

FOR EACH LONG SIGNAL:

1. REASONS MAKE SENSE? (Y/N/Partial)
   - Comments: ________________________________

2. IS IT RARE? (Signal frequency < 15% of days)
   - Signal count: ___
   - Days in period: ___
   - Frequency: ___%
   - Rating: Rare/Moderate/Frequent

3. NEAR CONTINUATION, NOT TOPS?
   - Price near highs: Yes/No
   - Rating: Good continuation/Potential top/Mixed

4. MAE SURVIVABLE?
   - Approx MAE: ___%
   - 3x leverage liquidates at -33.33%
   - Rating: Safe/Caution/Dangerous

OVERALL ASSESSMENT:
- Signal quality: Good/Needs improvement/Bad
- Action needed: 
  □ No action needed
  □ Review features
  □ Consider R6 for mean reversion
- Comments: ________________________________
"""

    with open('validation/manual_review_template.txt', 'w') as f:
        f.write(template_content)
    
    print("✅ Created manual review template: validation/manual_review_template.txt")

# ========== TREND LONG ENGINE FUNCTIONS ==========
def run_trend_long_pipeline():
    """
    Execute complete Trend LONG pipeline using Core Engine
    """
    print("\n" + "="*70)
    print("CORE TREND LONG ENGINE - DAILY SIGNAL GENERATION")
    print("="*70)
    
    # Load data
    df_whales, df_market, df_btc, df_eth, df_funding = load_data_from_files()
    
    if any(d.empty for d in [df_whales, df_market, df_btc, df_eth]):
        print("❌ Missing essential data")
        return None, None
    
    # Engineer features
    df_features = engineer_features(df_whales, df_market, df_btc, df_eth, df_funding)
    df_pipeline = build_pipeline_complete(df_features)
    
    # Load or rebuild engine
    engine = rebuild_models_if_needed(df_pipeline)
    
    if not engine or not engine.long_model:
        print("❌ Could not load LONG model")
        return None, None
    
    # Generate signals for last 90 days
    recent_data = df_pipeline.iloc[-90:].copy()
    trend_signals = []
    
    print(f"\n📊 Generating Trend LONG signals for {len(recent_data)} days...")
    
    for idx, row in recent_data.iterrows():
        signal = engine.generate_core_signal(row, df_pipeline)
        trend_signals.append(signal)
    
    # Analyze results
    df_signals = pd.DataFrame(trend_signals)
    valid_signals = df_signals[df_signals['action'] == 'ENTER']
    
    print(f"\n📈 TREND LONG SIGNAL ANALYSIS:")
    print(f"   Total days analyzed: {len(df_signals)}")
    print(f"   Valid LONG signals: {len(valid_signals)}")
    print(f"   Signal frequency: {len(valid_signals)/len(df_signals)*100:.1f}%")
    
    # Get latest signal
    latest_signal = trend_signals[-1] if trend_signals else None
    
    # Save to validation
    df_signals.to_csv('validation/trend_long_signals.csv', index=False)
    
    if latest_signal:
        with open('data/latest_trend_long_signal.json', 'w') as f:
            json.dump(latest_signal, f, indent=2)
        
        print(f"\n🎯 LATEST TREND LONG SIGNAL:")
        print(json.dumps(latest_signal, indent=2))
    
    print(f"\n✅ Trend LONG pipeline complete")
    print(f"   Signals saved to: validation/trend_long_signals.csv")
    
    return df_signals, latest_signal

def inspect_trend_long_signals(days=60):
    """
    Manual inspection of Trend LONG signals
    """
    print("\n" + "="*70)
    print(f"TREND LONG SIGNAL INSPECTION ({days} days)")
    print("="*70)
    
    # Load pipeline
    df_whales, df_market, df_btc, df_eth, df_funding = load_data_from_files()
    df_features = engineer_features(df_whales, df_market, df_btc, df_eth, df_funding)
    df_pipeline = build_pipeline_complete(df_features)
    
    # Load engine
    engine = rebuild_models_if_needed(df_pipeline)
    
    if not engine:
        print("❌ Could not load engine")
        return None
    
    # Get recent data
    recent_data = df_pipeline.iloc[-days:].copy()
    signals = []
    
    for idx, row in recent_data.iterrows():
        signal = engine.generate_core_signal(row, df_pipeline)
        signals.append(signal)
    
    df_signals = pd.DataFrame(signals)
    valid_signals = df_signals[df_signals['action'] == 'ENTER']
    
    print(f"\n📊 Summary:")
    print(f"   Days analyzed: {len(df_signals)}")
    print(f"   Valid LONG signals: {len(valid_signals)}")
    print(f"   Frequency: {len(valid_signals)/len(df_signals)*100:.1f}%")
    
    return df_signals

# ========== R6 FUNCTIONS ==========
def test_r6_engine():
    """Test R6 Mean Reversion Engine"""
    print("\n" + "="*70)
    print("TEST R6 MEAN REVERSION ENGINE")
    print("="*70)
    
    if R6MeanReversionEngine is None:
        print("❌ R6 Engine not available")
        print("   Please create r6_mean_reversion.py")
        return
    
    # Load data
    df_whales, df_market, df_btc, df_eth, df_funding = load_data_from_files()
    df_features = engineer_features(df_whales, df_market, df_btc, df_eth, df_funding)
    df_pipeline = build_pipeline_complete(df_features)
    
    # Initialize R6 engine
    r6_engine = R6MeanReversionEngine()
    
    # Test on recent data
    recent = df_pipeline.iloc[-30:].copy()
    print(f"\n🔍 Testing R6 detection on {len(recent)} recent days...")
    
    r6_days = 0
    for idx, row in recent.iterrows():
        regime = row.get('regime_code', 'R0')
        r6_active, reason = r6_engine.detect_r6_regime(row, df_pipeline, regime)
        
        if r6_active:
            r6_days += 1
            print(f"✅ {row['block_date'].date()}: R6 ACTIVE ({reason})")
    
    print(f"\n📊 R6 Regime Statistics:")
    print(f"   R6 active days: {r6_days}/{len(recent)} ({r6_days/len(recent)*100:.1f}%)")

def run_r6_shadow_trading(days=60):
    """Run R6 shadow trading"""
    print("\n" + "="*70)
    print("R6 SHADOW TRADING")
    print("="*70)
    
    if R6MeanReversionEngine is None:
        print("❌ R6 Engine not available")
        return
    
    # Load data
    df_whales, df_market, df_btc, df_eth, df_funding = load_data_from_files()
    df_features = engineer_features(df_whales, df_market, df_btc, df_eth, df_funding)
    df_pipeline = build_pipeline_complete(df_features)
    
    # Initialize R6 engine
    r6_engine = R6MeanReversionEngine()
    
    # Run shadow trading
    signals = r6_engine.run_shadow_trading(df_pipeline, days=days)
    
    print(f"\n✅ R6 shadow trading complete")

# ========== UNIFIED ORCHESTRATOR FUNCTIONS ==========
def run_unified_execution():
    """Run unified daily execution"""
    print("\n" + "="*70)
    print("UNIFIED DAILY EXECUTION")
    print("="*70)
    
    if UnifiedOrchestrator is None:
        print("❌ Unified Orchestrator not available")
        print("   Please create unified_orchestrator.py")
        return
    
    # Load pipeline data
    if not os.path.exists('data/pipeline_complete.csv'):
        print("❌ Pipeline data not found")
        return
    
    df = pd.read_csv('data/pipeline_complete.csv', parse_dates=['block_date'])
    
    # Load models
    engine = rebuild_models_if_needed(df)
    
    if not engine or not engine.long_model or not engine.short_model:
        print("❌ Could not load models")
        return
    
    # Initialize orchestrator
    orchestrator = UnifiedOrchestrator()
    
    # Get latest trend signal
    latest_row = df.iloc[-1].copy()
    trend_signal = engine.generate_core_signal(latest_row, df)
    
    # Get unified signal
    unified_signal = orchestrator.get_daily_signal(latest_row, df, trend_signal)
    
    print(f"\n📋 UNIFIED SIGNAL:")
    print(json.dumps(unified_signal, indent=2))
    
    # Save to file
    with open('data/latest_unified_signal.json', 'w') as f:
        json.dump(unified_signal, f, indent=2)
    
    print(f"\n✅ Unified execution complete")

# ========== MAIN PIPELINE ==========
def run_unified_pipeline():
    """
    Execute complete pipeline with funding data from loader
    """
    print("\n" + "="*70)
    print("ETH WHALE ALPHA PIPELINE - UNIFIED SYSTEM")
    print("="*70)
    
    # Step 1: Load all data including funding
    df_whales, df_market, df_btc, df_eth, df_funding = load_data_from_files()
    
    # Check essential data
    if any(d.empty for d in [df_whales, df_market, df_btc, df_eth]):
        print("\n❌ Missing essential data. Run data_loader.py first.")
        return None, None, None
    
    # Step 2: Engineer features with funding data
    df_features = engineer_features(df_whales, df_market, df_btc, df_eth, df_funding)
    
    # Step 3: Build pipeline
    df_pipeline = build_pipeline_complete(df_features)
    
    # Step 4: Rebuild models using core engine
    engine = rebuild_models_if_needed(df_pipeline)
    
    # Step 5: Generate live signal
    if engine and engine.long_model and engine.short_model:
        print("\n" + "="*70)
        print("GENERATING LIVE SIGNAL")
        print("="*70)
        
        signal = engine.generate_daily_signal(df_pipeline)
        print(json.dumps(signal, indent=2))
        
        with open('data/latest_signal.json', 'w') as f:
            json.dump(signal, f, indent=2)
    
    return df_pipeline, engine

# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    print("\n" + "="*70)
    print("ETH WHALE ALPHA PIPELINE - UNIFIED SYSTEM")
    print("="*70)
    
    # Check file system
    print("\n📁 Checking file system...")
    for d in ['shadow_trading', 'validation', 'models', 'data']:
        if os.path.exists(d):
            print(f"✅ {d}: Found")
        else:
            print(f"❌ {d}: Not found")
    
    # Check required files
    required_files = [
        'data/whale_ml_ready.csv',
        'data/market_intent_ml_ready.csv', 
        'data/price_cache/btc.csv',
        'data/price_cache/eth.csv'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"\n⚠️  Missing data files:")
        for f in missing_files:
            print(f"   - {f}")
        print(f"\nPlease run data_loader.py first to fetch data")
        exit(1)
    
    # Main menu
    print("\n" + "="*70)
    print("MAIN MENU")
    print("="*70)
    print("📋 Available Options:")
    print("   1. Run unified pipeline (train models + generate signal)")
    print("   2. 60-day paper trade test")
    print("   3. Manual signal review (30 days)")
    print("   4. Inspect signals (60 days)")
    print("   5. Load and check data only")
    print("   6. Extended LONG inspection (2020-2021 bull cycle)")
    print("   7. 90-day shadow trading with MAE/MFE")
    print("   8. [DEBUG] Test shadow trading system")
    print("   9. [MANUAL INSPECTION] Export signals for manual review")
    print("   10. Create manual review template")
    print("   11. [TREND] Generate Trend LONG signals (Core Engine)")
    print("   12. [TREND] Inspect Trend LONG signals (Manual Review)")
    print("   13. [R6] Test R6 Mean Reversion Engine")
    print("   14. [R6] Run R6 shadow trading (60 days)")
    print("   15. [UNIFIED] Run unified daily execution")
    print("   16. Check system status")
    
    choice = input("\nSelect option (1-16): ").strip()
    
    if choice == '1':
        df_pipeline, engine = run_unified_pipeline()
        
        if df_pipeline is not None:
            print("\n✅ Unified pipeline complete")
            print("\n📋 Next steps:")
            print("   1. Review data/latest_signal.json")
            print("   2. Run option 2 for paper trade test")
            print("   3. Run option 11 for Trend LONG signals")
    
    elif choice == '2':
        signals = run_90_day_shadow_trading()
        print("\n✅ Paper trade test complete")
    
    elif choice == '3':
        num_days = input("How many days to review? (default: 30): ").strip()
        try:
            num_days = int(num_days) if num_days else 30
        except:
            num_days = 30
        
        print(f"\n📊 Manual review for {num_days} days")
        print("="*50)
        
        # Load data
        df_whales, df_market, df_btc, df_eth, df_funding = load_data_from_files()
        df_features = engineer_features(df_whales, df_market, df_btc, df_eth, df_funding)
        df_pipeline = build_pipeline_complete(df_features)
        
        # Load engine
        engine = rebuild_models_if_needed(df_pipeline)
        
        if engine:
            recent = df_pipeline.iloc[-num_days:].copy()
            for idx, row in recent.iterrows():
                signal = engine.generate_core_signal(row, df_pipeline)
                if signal['action'] == 'ENTER':
                    print(f"\n📅 {signal['date']} - {signal['direction']} in {signal['regime']}")
                    print(f"   Confidence: {signal['adjusted_confidence']:.2f}")
                    print(f"   Size: {signal['position_size']:.2f}")
                    print(f"   Reasons: {', '.join(signal['reasons'])}")
    
    elif choice == '4':
        # Load pipeline data
        if not os.path.exists('data/pipeline_complete.csv'):
            print("❌ Pipeline data not found. Run option 1 first.")
        else:
            df = pd.read_csv('data/pipeline_complete.csv', parse_dates=['block_date'])
            engine = rebuild_models_if_needed(df)
            
            if engine:
                recent = df.iloc[-60:].copy()
                print(f"\n📊 Signal inspection (60 days):")
                print("="*50)
                
                long_count = 0
                short_count = 0
                for idx, row in recent.iterrows():
                    signal = engine.generate_core_signal(row, df)
                    if signal['action'] == 'ENTER':
                        icon = "🟢" if signal['direction'] == 'LONG' else "🔴"
                        print(f"{icon} {signal['date']}: {signal['direction']} @ ${row['eth_price']:.0f} "
                              f"(conf: {signal['adjusted_confidence']:.2f}, size: {signal['position_size']:.2f})")
                        
                        if signal['direction'] == 'LONG':
                            long_count += 1
                        else:
                            short_count += 1
                
                print(f"\n📊 Summary: LONG: {long_count}, SHORT: {short_count}, Total: {long_count + short_count}")
    
    elif choice == '5':
        print("\n📂 Loading and checking data...")
        df_whales, df_market, df_btc, df_eth, df_funding = load_data_from_files()
        
        print(f"\n✅ Data loaded successfully:")
        print(f"   Whale data: {len(df_whales)} rows")
        print(f"   Market data: {len(df_market)} rows")
        print(f"   BTC price: {len(df_btc)} rows")
        print(f"   ETH price: {len(df_eth)} rows")
        print(f"   Funding data: {len(df_funding)} rows")
    
    elif choice == '6':
        print("\n" + "="*70)
        print("EXTENDED LONG INSPECTION (2020-2021 BULL CYCLE)")
        print("="*70)
        
        # Load pipeline data
        if not os.path.exists('data/pipeline_complete.csv'):
            print("❌ Pipeline data not found. Run option 1 first.")
        else:
            df = pd.read_csv('data/pipeline_complete.csv', parse_dates=['block_date'])
            engine = rebuild_models_if_needed(df)
            
            if engine:
                # Filter to bull cycle period
                mask = (df['block_date'] >= '2020-01-01') & (df['block_date'] <= '2022-01-01')
                bull_data = df[mask].copy()
                
                print(f"\n📊 Bull cycle analysis:")
                print(f"   Period: {bull_data['block_date'].min().date()} to {bull_data['block_date'].max().date()}")
                print(f"   Total days: {len(bull_data)}")
                
                # Get R1/R2 days
                bull_regimes = bull_data[bull_data['regime_code'].isin(['R1', 'R2'])].copy()
                print(f"   R1/R2 days: {len(bull_regimes)}")
    
    elif choice == '7':
        trades = run_90_day_shadow_trading()
        
        export_inspection = input("\n📋 Also export for manual inspection? (y/n): ").strip().lower()
        if export_inspection == 'y' or export_inspection == 'yes':
            if os.path.exists('data/pipeline_complete.csv'):
                df = pd.read_csv('data/pipeline_complete.csv', parse_dates=['block_date'])
                engine = rebuild_models_if_needed(df)
                
                if engine:
                    print("\n" + "="*70)
                    print("EXPORTING FOR MANUAL INSPECTION")
                    print("="*70)
                    export_signals_for_manual_inspection(df, engine, days=90)
        
        print("\n📋 Shadow trading complete!")
        print("\n📊 Output files:")
        print("   shadow_trading/shadow_trades_90d.csv - Trade log")
        print("   shadow_trading/shadow_analysis.md - Detailed analysis")
    
    elif choice == '8':
        print("\n🧪 Testing Shadow Trading System...")
        
        # Create test data
        dates = pd.date_range(start='2025-01-01', end='2025-12-31', freq='D')
        test_df = pd.DataFrame({
            'block_date': dates,
            'eth_price': np.random.uniform(2000, 4000, len(dates))
        })
        
        # Test signal
        test_signal = {
            'date': '2025-06-15',
            'action': 'ENTER',
            'direction': 'SHORT',
            'regime': 'R5',
            'position_size': 1.0,
            'adjusted_confidence': 0.75,
            'model_probability': 0.80,
            'reasons': ['test_reason']
        }
        
        if ShadowTrader:
            trader = ShadowTrader()
            trade = trader.log_trade(test_signal, test_df, trade_days=10)
            
            if trade:
                print(f"✅ Basic test passed!")
            else:
                print(f"❌ Basic test failed")
        else:
            print("❌ ShadowTrader not available")
    
    elif choice == '9':
        print("\n📋 MANUAL SIGNAL INSPECTION EXPORT")
        print("="*50)
        
        # Load pipeline data
        if not os.path.exists('data/pipeline_complete.csv'):
            print("❌ Pipeline data not found. Run unified pipeline first.")
        else:
            df = pd.read_csv('data/pipeline_complete.csv', parse_dates=['block_date'])
            engine = rebuild_models_if_needed(df)
            
            if engine:
                days = input("How many days to export? (default: 60): ").strip()
                try:
                    days = int(days) if days else 60
                except:
                    days = 60
                
                signals = export_signals_for_manual_inspection(df, engine, days)
                
                print(f"\n✅ Files created for manual inspection:")
                print(f"   1. validation/manual_inspection_signals.csv")
                
                print(f"\n📝 MANUAL INSPECTION GUIDE:")
                print(f"   1. Open manual_inspection_signals.csv in Excel/Sheets")
                print(f"   2. Filter for action = 'ENTER'")
                print(f"   3. For each signal, answer the review questions")
    
    elif choice == '10':
        print("\n📋 CREATING MANUAL REVIEW TEMPLATE")
        print("="*50)
        create_manual_review_template()
    
    elif choice == '11':
        print("\n" + "="*70)
        print("STEP 2: GENERATE TREND LONG SIGNALS")
        print("="*70)
        
        df_signals, latest_signal = run_trend_long_pipeline()
        
        print(f"\n📝 NEXT STEP:")
        print("   Review signals in validation/trend_long_signals.csv")
        print("   Check frequency and quality match expectations")
    
    elif choice == '12':
        days = input("How many days to inspect? (default: 60): ").strip()
        try:
            days = int(days) if days else 60
        except:
            days = 60
        
        print("\n" + "="*70)
        print("STEP 2: MANUAL INSPECTION OF TREND LONG SIGNALS")
        print("="*70)
        
        df_signals = inspect_trend_long_signals(days)
        
        print(f"\n📋 MANUAL ASSESSMENT GUIDELINES:")
        print("If LONGs look stupid → fix features, not thresholds")
        print("If LONGs look good but rare → tune thresholds")
        print("If LONGs cluster late → add R3/R6 later (separate module)")
    
    elif choice == '13':
        test_r6_engine()
    
    elif choice == '14':
        days = input("How many days? (default: 60): ").strip()
        try:
            days = int(days) if days else 60
        except:
            days = 60
        
        run_r6_shadow_trading(days)
    
    elif choice == '15':
        run_unified_execution()
    
    elif choice == '16':
        print("\n" + "="*70)
        print("SYSTEM STATUS CHECK")
        print("="*70)
        print(f"Core Trend Engine: {'✅ Available' if CoreTrendEngine else '❌ Missing'}")
        print(f"R6 Mean Reversion: {'✅ Available' if R6MeanReversionEngine else '❌ Missing'}")
        print(f"Unified Orchestrator: {'✅ Available' if UnifiedOrchestrator else '❌ Missing'}")
        print(f"Shadow Trading: {'✅ Available' if ShadowTrader else '❌ Missing'}")
        
        # Check data files
        print(f"\n📁 Data files:")
        for f in required_files:
            status = "✅ Found" if os.path.exists(f) else "❌ Missing"
            print(f"   {f}: {status}")
    
    else:
        print("❌ Invalid option")
    
    print("\n" + "="*70)
    print("EXECUTION COMPLETE")
    print("="*70)