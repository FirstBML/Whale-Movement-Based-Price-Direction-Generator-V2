import pandas as pd
import numpy as np
import os

def price_not_near_lows(row, df, lookback=90, min_pct=0.25):
    """Require price above X percentile of recent range (LONG only)"""
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
    """Protection against buying local tops (LONG only)"""
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

def export_signals_for_manual_inspection(df, engine=None, days=60):
    """Export signals for manual inspection"""
    print(f"\n📋 Exporting {days} days of signals for manual inspection...")
    
    recent_data = df.iloc[-days:].copy()
    signals = []
    
    for idx, row in recent_data.iterrows():
        if engine:
            signal = engine.generate_core_signal(row, df)
        else:
            signal = {
                "date": str(row['block_date'].date()),
                "regime": row.get('regime_code', 'R0'),
                "direction": None,
                "action": "NO_TRADE",
                "reasons": ["no_engine"]
            }
        
        near_lows = price_not_near_lows(row, df, lookback=90, min_pct=0.25)
        near_highs = not price_not_near_highs(row, df, lookback=90, max_pct=0.75)
        
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
    
    df_signals = pd.DataFrame(signals)
    
    os.makedirs('validation', exist_ok=True)
    output_file = 'validation/manual_inspection_signals.csv'
    df_signals.to_csv(output_file, index=False)
    
    print(f"✅ Exported {len(df_signals)} signals to {output_file}")
    return df_signals

def create_manual_review_template():
    """Create template for manual review"""
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
  [ ] No action needed
  [ ] Review features
  [ ] Consider R6 for mean reversion
- Comments: ________________________________
"""

    os.makedirs('validation', exist_ok=True)
    with open('validation/manual_review_template.txt', 'w', encoding='utf-8') as f:
        f.write(template_content)
    
    print("✅ Created manual review template: validation/manual_review_template.txt")