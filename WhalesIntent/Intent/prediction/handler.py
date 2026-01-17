import pandas as pd
import json
from engines.core_trend_engine import CoreTrendEngine
from loader.pipeline import load_and_prepare_data

# Initialize engine once (Lambda container reuse)
engine = CoreTrendEngine.load_production()

def lambda_handler(event, context):
    """
    AWS Lambda entry point.
    Works with both API Gateway and Function URL.
    """
    try:
        # Load and prepare data
        df = load_and_prepare_data()
        
        # Generate signal
        signal = engine.generate_daily_signal(df)
        
        # Return properly formatted response
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps(signal)  # ✅ Already correct
        }
    
    except Exception as e:
        # Error handling
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({
                "error": str(e),
                "type": type(e).__name__
            })
        }