"""
prediction/handler.py
Complete AWS Lambda handler with S3 data loading
"""

import json
import os
import pandas as pd
import boto3
from io import StringIO
from engines.core_trend_engine import CoreTrendEngine

# ================================================================
# CONFIGURATION
# ================================================================

# S3 bucket configuration (set via Lambda environment variables)
S3_BUCKET = os.environ.get('DATA_BUCKET', 'eth-whale-alpha-data')
S3_KEY = os.environ.get('DATA_KEY', 'pipeline_complete.csv')

# Initialize AWS clients
s3_client = boto3.client('s3')

# Global engine instance (reused across Lambda invocations)
engine = None


# ================================================================
# S3 DATA LOADING
# ================================================================

def load_data_from_s3():
    """
    Load pipeline data from S3
    Uses streaming to avoid loading entire file into memory
    """
    try:
        print(f"Loading data from s3://{S3_BUCKET}/{S3_KEY}")
        
        # Stream data from S3
        obj = s3_client.get_object(
            Bucket=S3_BUCKET,
            Key=S3_KEY
        )
        
        # Read directly into pandas
        df = pd.read_csv(
            StringIO(obj['Body'].read().decode('utf-8')),
            parse_dates=['block_date']
        )
        
        print(f"✅ Loaded {len(df)} rows from S3")
        return df
        
    except s3_client.exceptions.NoSuchKey:
        raise FileNotFoundError(
            f"Data file not found: s3://{S3_BUCKET}/{S3_KEY}"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load data from S3: {str(e)}")


# ================================================================
# ENGINE INITIALIZATION
# ================================================================

def initialize_engine():
    """
    Initialize engine on cold start
    Reused across warm invocations
    """
    global engine
    
    if engine is None:
        print("Initializing CoreTrendEngine...")
        engine = CoreTrendEngine()
        engine.load_models()
        print("✅ Engine initialized")
    
    return engine


# ================================================================
# PREDICTION LOGIC
# ================================================================

def generate_prediction(df):
    """
    Generate prediction using CoreTrendEngine
    """
    # Get engine instance
    eng = initialize_engine()
    
    # Generate signal
    signal = eng.generate_daily_signal(df)
    
    # Add metadata
    latest_row = df.iloc[-1]
    
    return {
        "date": str(latest_row["block_date"].date()),
        "regime": str(latest_row.get("regime_code", "UNKNOWN")),
        "eth_price": float(latest_row.get("eth_price", 0)),
        "signal": {
            "action": signal.get("action", "NO_TRADE"),
            "direction": signal.get("direction"),
            "confidence": float(signal.get("adjusted_confidence", 0)),
            "position_size": float(signal.get("position_size", 0)),
            "reasons": signal.get("reasons", []),
            "engine": signal.get("engine", "core_trend")
        }
    }


# ================================================================
# LAMBDA HANDLER
# ================================================================

def lambda_handler(event, context):
    """
    AWS Lambda entry point
    Works with API Gateway, Function URL, and direct invocation
    """
    try:
        print("Lambda invocation started")
        
        # Load data from S3
        df = load_data_from_s3()
        
        # Generate prediction
        prediction = generate_prediction(df)
        
        print("✅ Prediction generated successfully")
        
        # Return response
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Cache-Control": "no-cache"
            },
            "body": json.dumps(prediction, indent=2)
        }
    
    except FileNotFoundError as e:
        print(f"❌ Data file not found: {e}")
        return {
            "statusCode": 404,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({
                "error": "Data file not found in S3",
                "message": str(e),
                "type": "FileNotFoundError"
            })
        }
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({
                "error": "Internal server error",
                "message": str(e),
                "type": type(e).__name__
            })
        }


# ================================================================
# LOCAL TESTING
# ================================================================

if __name__ == "__main__":
    """
    Test locally before deploying
    """
    print("\n🧪 LOCAL TESTING MODE")
    print("="*70)
    
    # Mock event and context
    test_event = {}
    test_context = None
    
    # Override S3 bucket for local testing
    os.environ['DATA_BUCKET'] = 'eth-whale-alpha-data'
    os.environ['DATA_KEY'] = 'pipeline_complete.csv'
    
    # Run handler
    response = lambda_handler(test_event, test_context)
    
    # Print result
    print("\n📊 Response:")
    print(json.dumps(response, indent=2))
    
    # Validate
    if response['statusCode'] == 200:
        print("\n✅ Test passed!")
    else:
        print("\n❌ Test failed!")

# ================================================================
# MONITORING
# ================================================================

def log_metrics(prediction):
    """
    Optional: Send custom metrics to CloudWatch
    """
    try:
        cloudwatch = boto3.client('cloudwatch')
        
        cloudwatch.put_metric_data(
            Namespace='ETHWhaleAlpha',
            MetricData=[
                {
                    'MetricName': 'PredictionGenerated',
                    'Value': 1.0,
                    'Unit': 'Count',
                    'Dimensions': [
                        {
                            'Name': 'Regime',
                            'Value': prediction.get('regime', 'UNKNOWN')
                        },
                        {
                            'Name': 'Action',
                            'Value': prediction['signal'].get('action', 'NO_TRADE')
                        }
                    ]
                }
            ]
        )
    except Exception as e:
        print(f"⚠️  Failed to log metrics: {e}")
        # Don't fail the request if metrics fail
        

# import pandas as pd
# import json
# from engines.core_trend_engine import CoreTrendEngine
# from loader.pipeline import load_and_prepare_data

# # Initialize engine once (Lambda container reuse)
# engine = CoreTrendEngine.load_production()

# def lambda_handler(event, context):
#     """
#     AWS Lambda entry point.
#     Works with both API Gateway and Function URL.
#     """
#     try:
#         # Load and prepare data
#         df = load_and_prepare_data()
        
#         # Generate signal
#         signal = engine.generate_daily_signal(df)
        
#         # Return properly formatted response
#         return {
#             "statusCode": 200,
#             "headers": {
#                 "Content-Type": "application/json",
#                 "Access-Control-Allow-Origin": "*"
#             },
#             "body": json.dumps(signal)  # ✅ Already correct
#         }
    
#     except Exception as e:
#         # Error handling
#         return {
#             "statusCode": 500,
#             "headers": {
#                 "Content-Type": "application/json",
#                 "Access-Control-Allow-Origin": "*"
#             },
#             "body": json.dumps({
#                 "error": str(e),
#                 "type": type(e).__name__
#             })
#         }