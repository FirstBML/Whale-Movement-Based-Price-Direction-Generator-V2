"""
Lightweight data loader for analysis and notebooks.
Wraps core pipeline logic without target creation.
"""

from loader.data_loader import load_cached_data
from loader.features import engineer_features
from regimes.trend_regimes import define_all_regimes


def load_and_prepare_data():
    """
    Returns feature + regime dataframe (NO TARGETS).
    Safe for EDA and analysis.
    """

    df_whales, df_market, df_btc, df_eth, df_funding = load_cached_data()

    df_features = engineer_features(
        df_whales,
        df_market,
        df_btc,
        df_eth,
        df_funding
    )

    df_final = define_all_regimes(df_features)

    return df_final.sort_values("block_date").reset_index(drop=True)
