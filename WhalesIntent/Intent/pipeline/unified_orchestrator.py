"""
UNIFIED ORCHESTRATOR
Coordinates execution priority:
1. R5 shorts (highest priority)
2. R1/R2 trend trades  
3. R6 mean reversion (only if no active trend signal)
"""

import pandas as pd
import json
import os
from datetime import datetime

# Import engines
try:
    from r6_mean_reversion import R6MeanReversionEngine
    print("✅ R6 Mean Reversion Engine imported")
except ImportError:
    print("⚠️  R6 module not found - create r6_mean_reversion.py first")
    R6MeanReversionEngine = None

class UnifiedOrchestrator:
    """
    Coordinates between different trading engines with proper priority
    """
    
    def __init__(self):
        self.r6_engine = R6MeanReversionEngine() if R6MeanReversionEngine else None
        self.execution_log = []
        
    def get_daily_signal(self, row, df, trend_signal):
        """
        Get unified daily signal with proper priority
        """
        current_regime = row.get('regime_code', 'R0')
        date_str = row['block_date'].strftime('%Y-%m-%d') if hasattr(row['block_date'], 'strftime') else str(row['block_date'])
        
        # Initialize unified signal
        unified_signal = {
            "date": date_str,
            "regime": current_regime,
            "primary_signal": None,
            "primary_engine": None,
            "secondary_signal": None,
            "secondary_engine": None,
            "final_action": "NO_TRADE",
            "final_direction": None,
            "final_confidence": 0.0,
            "final_size": 0.0,
            "execution_priority": [],
            "reasons": []
        }
        
        # ===== EXECUTION PRIORITY =====
        
        # 1. R5 SHORTS (highest priority)
        if current_regime == "R5" and trend_signal.get('direction') == "SHORT":
            unified_signal["primary_signal"] = trend_signal
            unified_signal["primary_engine"] = "trend_short"
            unified_signal["final_action"] = "ENTER"
            unified_signal["final_direction"] = "SHORT"
            unified_signal["final_confidence"] = trend_signal.get('adjusted_confidence', 0)
            unified_signal["final_size"] = trend_signal.get('position_size', 0)
            unified_signal["execution_priority"] = ["R5_short", "R1/R2_long", "R6_long"]
            unified_signal["reasons"] = ["priority_1_r5_short"]
            return unified_signal
        
        # 2. R1/R2 TREND LONGS
        elif current_regime in ["R1", "R2"] and trend_signal.get('direction') == "LONG":
            unified_signal["primary_signal"] = trend_signal
            unified_signal["primary_engine"] = "trend_long"
            unified_signal["final_action"] = "ENTER"
            unified_signal["final_direction"] = "LONG"
            unified_signal["final_confidence"] = trend_signal.get('adjusted_confidence', 0)
            unified_signal["final_size"] = trend_signal.get('position_size', 0)
            unified_signal["execution_priority"] = ["R1/R2_long", "R6_long", "R5_short"]
            unified_signal["reasons"] = ["priority_2_trend_long"]
            return unified_signal
        
        # 3. R6 MEAN REVERSION (only if trend engine says "I don't know")
        elif self.r6_engine and trend_signal.get('action') == "NO_TRADE":
            # Check if R6 regime is active
            r6_active, _ = self.r6_engine.detect_r6_regime(row, df, current_regime)
            
            if r6_active:
                r6_signal = self.r6_engine.generate_r6_signal(row, df, current_regime)
                
                if r6_signal["entry_triggered"]:
                    unified_signal["secondary_signal"] = r6_signal
                    unified_signal["secondary_engine"] = "r6_mean_reversion"
                    unified_signal["final_action"] = "ENTER"
                    unified_signal["final_direction"] = "LONG"
                    unified_signal["final_confidence"] = r6_signal.get('confidence', 0)
                    unified_signal["final_size"] = r6_signal.get('position_size', 0)
                    unified_signal["execution_priority"] = ["R6_long", "R1/R2_long", "R5_short"]
                    unified_signal["reasons"] = r6_signal.get('reasons', [])
                    return unified_signal
        
        # No trade signal
        unified_signal["final_action"] = "NO_TRADE"
        unified_signal["reasons"] = trend_signal.get('reasons', []) + ["no_priority_signal"]
        return unified_signal
    
    def run_daily_execution(self, df, trend_short_model, trend_long_model):
        """
        Run complete daily execution workflow
        """
        print("\n" + "="*70)
        print("UNIFIED DAILY EXECUTION")
        print("="*70)
        
        # Get latest row
        latest_row = df.iloc[-1].copy()
        
        # Get trend signal (from existing system)
        from WhalesIntent.Intent.former_main import generate_unified_signal
        
        trend_signal = generate_unified_signal(latest_row, df, trend_short_model, trend_long_model)
        
        # Get unified signal with priority
        unified_signal = self.get_daily_signal(latest_row, df, trend_signal)
        
        # Log execution
        self.execution_log.append({
            "timestamp": datetime.now().isoformat(),
            "date": unified_signal["date"],
            "signal": unified_signal
        })
        
        # Save to file
        os.makedirs('logs/unified', exist_ok=True)
        log_file = f"logs/unified/{datetime.now().strftime('%Y-%m-%d')}.json"
        with open(log_file, 'w') as f:
            json.dump(unified_signal, f, indent=2)
        
        print(f"\n📋 UNIFIED SIGNAL:")
        print(json.dumps(unified_signal, indent=2))
        print(f"\n📝 Log saved to: {log_file}")
        
        return unified_signal