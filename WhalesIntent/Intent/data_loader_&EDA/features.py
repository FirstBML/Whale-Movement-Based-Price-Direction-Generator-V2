# data/features.py
import pandas as pd
import numpy as np

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


def add_price_features(df, price_col, prefix):
    """Add technical features for a price series"""
    df = df.copy()
    
    # Log returns
    df[f'{prefix}_log_return'] = np.log(df[price_col] / df[price_col].shift(1))
    
    # Lagged returns
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
    
    # Merge funding data
    if df_funding is not None and not df_funding.empty:
        df = pd.merge(
            df,
            df_funding[['block_date', 'eth_funding_rate_8h']],
            on='block_date',
            how='left'
        )
        print(f"✅ Merged funding data: {len(df_funding)} rows")
    
    df = df.sort_values('block_date').reset_index(drop=True)
    
    # Add price features
    df = add_price_features(df, 'eth_price', 'eth')
    df = add_price_features(df, 'btc_price', 'btc')
    
    # ETH/BTC ratio features
    df['eth_btc_ratio'] = df['eth_price'] / df['btc_price']
    df['eth_btc_ratio_ma7'] = df['eth_btc_ratio'].rolling(7).mean().shift(1)
    df['eth_btc_corr_30d'] = df['eth_log_return'].shift(1).rolling(30) \
        .corr(df['btc_log_return'].shift(1)).shift(1)
        
    # Add VOL_RATIO feature
    if 'eth_vol7' in df.columns and 'eth_vol30' in df.columns:
        df['vol_ratio'] = df['eth_vol7'] / df['eth_vol30']
        df['vol_ratio'] = df['vol_ratio'].clip(0.5, 2.0)
        print("✅ Created vol_ratio feature")
    else:
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
    
    # Fill NaN values
    feature_cols = [col for col in df.columns if col not in ['block_date']]
    df[feature_cols] = df[feature_cols].fillna(method='ffill').fillna(0)
    
    # Save engineered features
    df.to_csv('data/features_engineered.csv', index=False)
    print(f"✅ Features engineered: {len(df.columns)} columns, {len(df)} rows")
    
    return df