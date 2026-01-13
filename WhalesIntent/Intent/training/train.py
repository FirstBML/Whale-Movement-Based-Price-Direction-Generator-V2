# training/train_models.py - ALL IN ONE FILE
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
import json
import os

# ========== MODEL DEFINITIONS ==========
def get_logistic():
    """Baseline logistic regression"""
    return LogisticRegression(
        C=0.1,
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    )

def get_random_forest():
    """Random forest classifier"""
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight='balanced'
    )

def get_xgboost():
    """XGBoost classifier"""
    return XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        scale_pos_weight=1,
        eval_metric='auc'
    )

# ========== MODEL REGISTRY ==========
def save_model_with_metadata(model, metadata, path):
    """Save model with metadata"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save model
    joblib.dump(model, path)
    
    # Save metadata
    metadata_path = path.replace('.pkl', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"💾 Model saved: {path}")
    print(f"💾 Metadata saved: {metadata_path}")

def load_model_with_metadata(path):
    """Load model and metadata"""
    model = joblib.load(path)
    
    metadata_path = path.replace('.pkl', '_metadata.json')
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    return model, metadata

# ========== TRAINING FUNCTION ==========
def train_multi_models(df_pipeline, direction='LONG'):
    """
    Train multiple models and select the best one
    """
    print(f"\n🤖 Training {direction} models...")
    
    # Get feature lists
    from core_trend_engine import LONG_FEATURES, SHORT_FEATURES
    
    if direction == 'LONG':
        df_train = df_pipeline[df_pipeline['regime_code'].isin(['R1', 'R2'])].copy()
        y = (df_train['target_t2'] == 1).astype(int)
        features = [f for f in LONG_FEATURES if f in df_train.columns]
    elif direction == 'SHORT':
        df_train = df_pipeline[df_pipeline['regime_code'].isin(['R3', 'R5'])].copy()
        y = (df_train['target_t2'] == -1).astype(int)
        features = [f for f in SHORT_FEATURES if f in df_train.columns]
    else:
        raise ValueError(f"Unknown direction: {direction}")
    
    X = df_train[features].fillna(0)
    
    # Check if we have enough data
    if len(X) < 50:
        print(f"⚠️  Insufficient data for {direction}: only {len(X)} samples")
        return pd.DataFrame(), None
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize models
    models = {
        'logistic': get_logistic(),
        'random_forest': get_random_forest(),
        'xgboost': get_xgboost()
    }
    
    # Train and evaluate
    results = []
    best_score = 0
    best_model = None
    
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred_proba)
            
            results.append({
                'model': name,
                'auc': auc,
                'precision': precision_score(y_val, (y_pred_proba > 0.5).astype(int), zero_division=0),
                'recall': recall_score(y_val, (y_pred_proba > 0.5).astype(int), zero_division=0),
                'n_samples': len(X_train)
            })
            
            print(f"   {name:15s} AUC: {auc:.4f} (n={len(X_train)})")
            
            if auc > best_score:
                best_score = auc
                best_model = model
                best_model_name = name
                
        except Exception as e:
            print(f"   {name:15s} Error: {str(e)[:50]}")
    
    # Save best model
    if best_model is not None:
        # Add feature names for consistency
        best_model.feature_names_ = features
        
        model_path = f'models/best_{direction.lower()}_model.pkl'
        metadata = {
            'direction': direction,
            'auc': best_score,
            'model_type': best_model_name,
            'features': features,
            'train_date': pd.Timestamp.now().isoformat(),
            'n_samples': len(X_train),
            'results': results
        }
        
        save_model_with_metadata(best_model, metadata, model_path)
        print(f"\n✅ Best {direction} model: {best_model_name} (AUC: {best_score:.4f})")
        print(f"   Saved to: {model_path}")
    
    return pd.DataFrame(results), best_model