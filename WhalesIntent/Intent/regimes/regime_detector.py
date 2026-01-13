# regimes/regime_detector.py
import pandas as pd
import numpy as np

def define_all_regimes(df):
    """
    Define all regimes (R0-R5) in one function
    Returns DataFrame with 'regime_code' column
    """
    print("📈 Defining all regimes (R0-R5)...")
    
    # Start with R0
    df['regime_code'] = 'R0'
    
    # ===== R1-R4: Trend Regimes =====
    if 'btc_ret_lag1' not in df.columns or 'eth_vol7' not in df.columns:
        print("⚠️  Missing required columns for R1-R4")
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
    
    # ===== R5: Distribution Regime =====
    if 'whale_volume_ratio_delta_3d' in df.columns and 'exchange_flow_share' in df.columns:
        exchange_flow_median = df['exchange_flow_share'].rolling(60, min_periods=20).median()
        
        r5_mask = (
            (df['whale_volume_ratio_delta_3d'].fillna(0) > 0) &
            (df['exchange_flow_share'] > exchange_flow_median).fillna(False)
        )
        
        df.loc[r5_mask, 'regime_code'] = 'R5'
        r5_count = r5_mask.sum()
        print(f"   R5 days detected: {r5_count} ({r5_count/len(df)*100:.1f}%)")
    
    # Print distribution
    print("\n📊 Regime Distribution:")
    regime_stats = []
    for code in ['R1', 'R2', 'R3', 'R4', 'R5', 'R0']:
        count = (df['regime_code'] == code).sum()
        if len(df) > 0:
            pct = count / len(df) * 100
            icon = '🟢' if code == 'R1' else ('🔴' if code in ['R3', 'R5'] else '⚪')
            regime_stats.append(f"{icon} {code}: {count:4d} ({pct:5.1f}%)")
    
    for i in range(0, len(regime_stats), 2):
        row = regime_stats[i:i+2]
        print("  " + " | ".join(row))
    
    return df