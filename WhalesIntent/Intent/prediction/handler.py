import json
from engines.core_trend_engine import CoreTrendEngine
from loader.pipeline import load_and_prepare_data

engine = CoreTrendEngine.load_production()

def lambda_handler(event, context):
    """
    AWS Lambda entry point.
    JSON only. No prints.
    """

    df = load_and_prepare_data()
    signal = engine.generate_daily_signal(df)

    return {
        "statusCode": 200,
        "body": json.dumps(signal)
    }
