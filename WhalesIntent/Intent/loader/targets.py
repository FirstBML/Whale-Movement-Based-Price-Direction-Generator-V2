# data/targets.py
import pandas as pd
import numpy as np

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
