"""
prediction/handler.py
Hybrid handler - works both locally and in AWS Lambda
"""

import json
import os
import pandas as pd
from engines.core_trend_engine import CoreTrendEngine

# Try importing boto3 (only available in Lambda or if installed locally)
try:
    import boto3
    from io import StringIO
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    print("⚠️  boto3 not available - will use local data source")

# ================================================================
# CONFIGURATION
# ================================================================

# S3 bucket configuration (for Lambda)
S3_BUCKET = os.environ.get('DATA_BUCKET', 'eth-whale-alpha-data')
S3_KEY = os.environ.get('DATA_KEY', 'pipeline_complete.csv')

# Global engine instance (reused across Lambda invocations)
engine = None


# ================================================================
# ENVIRONMENT DETECTION
# ================================================================

def is_lambda_environment():
    """Check if running in real AWS Lambda (not Docker)"""
    # Real Lambda has this specific environment variable pattern
    aws_exec_env = os.environ.get('AWS_EXECUTION_ENV', '')
    return aws_exec_env.startswith('AWS_Lambda_')

# ================================================================
# DATA LOADING - HYBRID APPROACH
# ================================================================

def load_data():
    """
    Smart data loading:
    - Try S3 if in Lambda environment and boto3 available
    - Fall back to local file otherwise
    """
    # If in Lambda and boto3 is available, use S3
    if is_lambda_environment() and BOTO3_AVAILABLE:
        try:
            return load_data_from_s3()
        except Exception as e:
            print(f"⚠️  S3 load failed: {e}")
            print("Falling back to local data...")
            return load_data_from_local()
    
    # Otherwise, use local file
    return load_data_from_local()


def load_data_from_s3():
    """
    Load pipeline data from S3 (Lambda environment)
    """
    try:
        print(f"☁️  Loading data from s3://{S3_BUCKET}/{S3_KEY}")
        
        # Initialize S3 client
        s3_client = boto3.client('s3')
        
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
        
    except Exception as e:
        raise RuntimeError(f"Failed to load data from S3: {str(e)}")


def load_data_from_local():
    """
    Load pipeline data from local file (local environment)
    """
    try:
        print("📂 Loading data from local file...")
        
        # Try multiple possible paths
        possible_paths = [
            "data/pipeline_complete.csv",
            "../data/pipeline_complete.csv",
            "../../data/pipeline_complete.csv",
            os.path.join(os.path.dirname(__file__), "../data/pipeline_complete.csv")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                df = pd.read_csv(path, parse_dates=["block_date"])
                print(f"✅ Loaded {len(df)} rows from {path}")
                return df
        
        raise FileNotFoundError(
            "Could not find pipeline_complete.csv in any of these locations:\n" +
            "\n".join(f"  - {p}" for p in possible_paths)
        )
        
    except Exception as e:
        raise RuntimeError(f"Failed to load local data: {str(e)}")


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
        print("🔧 Initializing CoreTrendEngine...")
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
        "btc_price": float(latest_row.get("btc_price", 0)),
        "signal": {
            "action": signal.get("action", "NO_TRADE"),
            "direction": signal.get("direction"),
            "confidence": float(signal.get("adjusted_confidence", 0)),
            "position_size": float(signal.get("position_size", 0)),
            "model_probability": float(signal.get("model_probability", 0)),
            "reasons": signal.get("reasons", []),
            "engine": signal.get("engine", "core_trend")
        },
        "metadata": {
            "environment": "lambda" if is_lambda_environment() else "local",
            "data_source": "s3" if (is_lambda_environment() and BOTO3_AVAILABLE) else "local_file"
        }
    }


# ================================================================
# LAMBDA HANDLER
# ================================================================

def lambda_handler(event, context):
    """
    AWS Lambda entry point
    Works with API Gateway, Function URL, and direct invocation
    Also works for local testing
    """
    try:
        print("\n" + "="*70)
        print("ETH WHALE ALPHA - PREDICTION")
        print("="*70)
        print(f"Environment: {'AWS Lambda' if is_lambda_environment() else 'Local Development'}")
        print(f"boto3 available: {BOTO3_AVAILABLE}")
        print("="*70)
        
        # Load data (automatically chooses source)
        df = load_data()
        
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
                "error": "Data file not found",
                "message": str(e),
                "type": "FileNotFoundError"
            })
        }
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
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
    
    # Mock event and context
    test_event = {}
    test_context = None
    
    # Run handler
    response = lambda_handler(test_event, test_context)
    
    # Print result
    print("\n" + "="*70)
    print("RESPONSE")
    print("="*70)
    print(f"Status Code: {response['statusCode']}")
    
    if response['statusCode'] == 200:
        body = json.loads(response['body'])
        print("\nPrediction:")
        print(json.dumps(body, indent=2))
        print("\n✅ Test PASSED!")
    else:
        print("\nError:")
        print(response['body'])
        print("\n❌ Test FAILED!")


# ================================================================
# MONITORING (Lambda only)
# ================================================================

def log_metrics(prediction):
    """
    Optional: Send custom metrics to CloudWatch
    Only works in Lambda environment
    """
    if not is_lambda_environment() or not BOTO3_AVAILABLE:
        return
    
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
