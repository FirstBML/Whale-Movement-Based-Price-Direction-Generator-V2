"""
PIPELINE BUILDER
Composes full dataset: features + targets + regimes
"""
import pandas as pd
from loader.features import engineer_features
from loader.targets import create_targets_two_tier
from regimes.trend_regimes import define_all_regimes

def build_pipeline_complete(
    df_whales,
    df_market,
    df_btc,
    df_eth,
    df_funding
) -> pd.DataFrame:
    """
    Build full pipeline dataframe for:
    - EDA, training, backtests, live inference
    
    Returns:
        pd.DataFrame (chronologically ordered)
    """
    # Feature engineering
    df_features = engineer_features(
        df_whales, 
        df_market, 
        df_btc, 
        df_eth, 
        df_funding
    )
    
    # Target creation
    df_targets = create_targets_two_tier(df_features)
    
    # Regime detection
    df_complete = define_all_regimes(df_targets)
    
    # Final cleanup
    df_complete = df_complete.sort_values("block_date").reset_index(drop=True)
    
    return df_complete