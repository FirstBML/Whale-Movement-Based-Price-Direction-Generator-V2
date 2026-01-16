import numpy as np
import pandas as pd

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
