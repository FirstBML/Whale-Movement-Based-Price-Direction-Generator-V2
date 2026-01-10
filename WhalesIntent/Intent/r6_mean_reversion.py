"""
R6 MEAN REVERSION MODULE - FIXED: INVERTED VOL_RATIO LOGIC
Clean definition: Statistical reversion where price is stretched from equilibrium 
WITHOUT strong trend. Edge comes from snap-back during low-conviction periods.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

# ========== R6 CONFIGURATION ==========
R6_CONFIG = {
    # Regime Detection (FIXED: vol_ratio is now VETO only, not requirement)
    'vol_ratio_extreme_veto': 1.6,  # FIXED: Extreme volatility veto threshold
    'max_abs_return': 0.025,  # 2.5% single day move limit
    'trend_strength_threshold': None,  # FIXED: Dynamic bottom 40% percentile
    
    # Entry Logic
    'zscore_price_threshold': -2.0,
    'funding_neutral_threshold': 0.0001,  # 0.01%
    
    # Confirmation (intentionally light)
    'required_confirmation_score': 1,
    
    # Veto Rules
    'btc_drawdown_threshold': -0.03,  # -3%
    'eth_3d_drawdown_threshold': -0.08,  # -8%
    'r5_veto': True,  # Never trade in R5
}


class R6MeanReversionEngine:
    """
    R6 Mean Reversion Engine - COMPLETELY SEPARATE from trend logic
    """
    
    def __init__(self, config=None):
        self.config = config or R6_CONFIG.copy()
        self.signals = []
        
    def detect_r6_regime(self, row, df, current_regime):
        """
        FIXED: R6 is now active when price is stretched, trend is weak, 
        and volatility is NOT extreme.
        
        New R6 regime definition:
        1. trend_strength in bottom 40% (weak trend)
        2. |eth_ret_1d| < 2.5% (no strong single-day move)
        3. regime ∈ {R0, R3, R4} (not in trend regimes)
        4. vol_ratio < 1.6 (NOT in extreme volatility)
        """
        # Diagnostic print for vol_ratio distribution
        if row.name == 0:  # Only print once
            print(f"\n🔍 R6 DIAGNOSTIC - vol_ratio distribution:")
            if 'vol_ratio' in df.columns:
                try:
                    quantiles = df['vol_ratio'].quantile([0.5, 0.7, 0.8, 0.9, 0.95, 0.99]).round(3)
                    print(f"   50th percentile: {quantiles[0.5]:.3f}")
                    print(f"   70th percentile: {quantiles[0.7]:.3f}")
                    print(f"   80th percentile: {quantiles[0.8]:.3f}")
                    print(f"   90th percentile: {quantiles[0.9]:.3f}")
                    print(f"   95th percentile: {quantiles[0.95]:.3f}")
                    print(f"   99th percentile: {quantiles[0.99]:.3f}")
                    
                    # Check veto threshold coverage
                    pct_extreme = (df['vol_ratio'] > self.config['vol_ratio_extreme_veto']).mean() * 100
                    print(f"   Days above extreme veto ({self.config['vol_ratio_extreme_veto']:.1f}): {pct_extreme:.1f}%")
                    
                    # Also check z-score distribution
                    print(f"\n🔍 R6 DIAGNOSTIC - z-score opportunities:")
                    z_scores = []
                    for idx, row in df.iterrows():
                        z = self.calculate_price_zscore(row, df)
                        z_scores.append(z)
                    
                    if z_scores:
                        z_series = pd.Series(z_scores)
                        pct_stretched = (z_series <= -2.0).mean() * 100
                        print(f"   Days with z-score ≤ -2.0: {pct_stretched:.1f}%")
                        print(f"   Min z-score: {z_series.min():.2f}")
                        print(f"   Max z-score: {z_series.max():.2f}")
                        print(f"   Mean z-score: {z_series.mean():.2f}")
                except Exception as e:
                    print(f"   Diagnostic error: {e}")
        
        # --------------------------------------------------
        # FIXED: Check 1: Trend strength must be weak
        # --------------------------------------------------
        trend_strength = self._calculate_trend_strength(row, df)
        
        # Calculate dynamic threshold if not set
        if self.config['trend_strength_threshold'] is None:
            self._calculate_trend_strength_threshold(df)
        
        # FIXED: Use dynamic threshold (bottom 40% of trend strength)
        if trend_strength >= self.config['trend_strength_threshold']:
            return False, f"trend_strength_too_high_{trend_strength:.4f}_vs_{self.config['trend_strength_threshold']:.4f}"
        
        # --------------------------------------------------
        # Check 2: No strong single-day move
        # --------------------------------------------------
        eth_ret_1d = row.get('eth_ret_lag1', 0)
        if abs(eth_ret_1d) >= self.config['max_abs_return']:
            return False, f"single_day_trend_too_strong_{eth_ret_1d:.3f}"
        
        # --------------------------------------------------
        # FIXED: Check 3: Volatility NOT extreme (veto only)
        # --------------------------------------------------
        vol_ratio = row.get('vol_ratio', 1.0)
        if vol_ratio > self.config['vol_ratio_extreme_veto']:
            return False, f"vol_ratio_extreme_veto_{vol_ratio:.2f}"
        
        # --------------------------------------------------
        # Check 4: Regime eligibility
        # --------------------------------------------------
        # Explicitly blocked regimes
        if current_regime in ["R1", "R2", "R5"]:
            return False, f"in_trend_or_distribution_{current_regime}"
        
        # Explicitly preferred regime
        if current_regime == "R4":
            return True, f"r4_mean_reversion_vol_{vol_ratio:.2f}"
        
        # Neutral / weak regimes allowed
        if current_regime in ["R0", "R3"]:
            return True, f"r6_active_{current_regime}_vol_{vol_ratio:.2f}"
        
        return False, f"unsupported_regime_{current_regime}"
    
    def _calculate_trend_strength_threshold(self, df, percentile=40):
        """
        Calculate dynamic trend strength threshold
        Use bottom 40% of historical trend strength
        """
        try:
            # Calculate trend strength for entire dataframe
            trend_strengths = []
            for idx, row in df.iterrows():
                strength = self._calculate_trend_strength(row, df)
                trend_strengths.append(strength)
            
            if trend_strengths:
                trend_strengths_series = pd.Series(trend_strengths)
                # Remove zeros and extremes
                trend_strengths_series = trend_strengths_series[trend_strengths_series > 0]
                if len(trend_strengths_series) > 10:
                    threshold = np.percentile(trend_strengths_series, percentile)
                    self.config['trend_strength_threshold'] = threshold
                    print(f"🔍 R6 TREND STRENGTH THRESHOLD: {threshold:.6f} (bottom {percentile}% percentile)")
                    print(f"   Mean trend strength: {np.mean(trend_strengths_series):.6f}")
                    print(f"   Max trend strength: {np.max(trend_strengths_series):.6f}")
                    return threshold
            
            # Fallback to reasonable default
            self.config['trend_strength_threshold'] = 0.001  # Much more reasonable default
            return self.config['trend_strength_threshold']
            
        except Exception as e:
            print(f"⚠️  Error calculating trend strength threshold: {e}")
            self.config['trend_strength_threshold'] = 0.001  # Reasonable fallback
            return self.config['trend_strength_threshold']
    
    def _calculate_trend_strength(self, row, df, lookback=30):
        """
        Calculate trend strength based on price structure
        """
        try:
            current_idx = row.name
            start_idx = max(0, current_idx - lookback)
            
            # Get recent prices
            recent_prices = df.iloc[start_idx:current_idx]['eth_price'].values
            
            if len(recent_prices) < 10:
                return 0.0
            
            # Simple linear regression slope as trend strength
            x = np.arange(len(recent_prices))
            slope, _ = np.polyfit(x, recent_prices, 1)
            
            # Normalize by average price
            avg_price = np.mean(recent_prices)
            if avg_price > 0:
                trend_strength = abs(slope / avg_price)
            else:
                trend_strength = 0.0
                
            return trend_strength
            
        except:
            return 0.0
    
    def calculate_price_zscore(self, row, df, lookback=90):
        """
        Calculate price z-score from equilibrium
        z-score <= -2.0 indicates stretched to the downside
        """
        try:
            current_idx = row.name
            start_idx = max(0, current_idx - lookback)
            
            recent_prices = df.iloc[start_idx:current_idx]['eth_price'].values
            
            if len(recent_prices) < 20:
                return 0.0
            
            mean_price = np.mean(recent_prices)
            std_price = np.std(recent_prices)
            
            if std_price > 0:
                zscore = (row['eth_price'] - mean_price) / std_price
            else:
                zscore = 0.0
                
            return zscore
            
        except:
            return 0.0
    
    def check_hard_vetoes(self, row, df, current_regime):
        """
        Hard Veto Rules
        Mean reversion is not bravery. It is controlled opportunism.
        """
        veto_reasons = []
        
        # 1. BTC strong down trend veto
        btc_trend_7d = row.get('btc_ret_lag1', 0)
        if btc_trend_7d < self.config['btc_drawdown_threshold']:
            veto_reasons.append(f"btc_strong_down_{btc_trend_7d:.3f}")
        
        # 2. Never trade in R5 (distribution)
        if self.config['r5_veto'] and current_regime == "R5":
            veto_reasons.append("r5_distribution_regime")
        
        # 3. Falling knife veto
        eth_ret_3d = self._calculate_3d_return(row, df)
        if eth_ret_3d < self.config['eth_3d_drawdown_threshold']:
            veto_reasons.append(f"falling_knife_{eth_ret_3d:.2%}")
        
        # FIXED: 4. Extreme volatility veto (already checked in regime detection, but keep here for clarity)
        vol_ratio = row.get('vol_ratio', 1.0)
        if vol_ratio > self.config['vol_ratio_extreme_veto']:
            veto_reasons.append(f"extreme_volatility_{vol_ratio:.2f}")
        
        return veto_reasons
    
    def _calculate_3d_return(self, row, df):
        """Calculate 3-day return"""
        try:
            current_idx = row.name
            if current_idx >= 3:
                current_price = row['eth_price']
                price_3d_ago = df.iloc[current_idx - 3]['eth_price']
                return (current_price / price_3d_ago - 1)
        except:
            return 0.0
        return 0.0
    
    def calculate_confirmation_score(self, row):
        """
        Confirmation (intentionally light)
        Mean reversion dies if you over-confirm.
        """
        confirm_score = 0
        
        # Funding rate confirmation
        funding_rate = row.get('eth_funding_rate_8h', 0)
        if funding_rate <= self.config['funding_neutral_threshold']:
            confirm_score += 1
        
        # FIXED: Volatility confirmation removed - vol_ratio is now veto only
        # We don't require volatility spike for mean reversion
        
        return confirm_score
    
    def generate_r6_signal(self, row, df, current_regime):
        """
        Generate R6 mean reversion signal
        """
        signal = {
            "date": row['block_date'].strftime('%Y-%m-%d') if hasattr(row['block_date'], 'strftime') else str(row['block_date']),
            "direction": "LONG",
            "model_type": "r6_mean_reversion",
            "regime": current_regime,
            "r6_regime_active": False,
            "entry_triggered": False,
            "confidence": 0.0,
            "reasons": [],
            "veto_reasons": [],
            "position_size": 0.0,
            "quality_metrics": {}
        }
        
        # Step 1: Check if R6 regime is active
        r6_active, r6_reason = self.detect_r6_regime(row, df, current_regime)
        signal["r6_regime_active"] = r6_active
        signal["reasons"].append(r6_reason)
        
        if not r6_active:
            return signal
        
        # Step 2: Calculate price z-score
        price_zscore = self.calculate_price_zscore(row, df)
        signal["quality_metrics"]["price_zscore"] = price_zscore
        
        # Step 3: Check hard vetoes
        veto_reasons = self.check_hard_vetoes(row, df, current_regime)
        signal["veto_reasons"] = veto_reasons
        
        if veto_reasons:
            signal["reasons"].extend([f"veto_{r}" for r in veto_reasons])
            return signal
        
        # Step 4: Check entry conditions
        entry_conditions = []
        
        # FIXED: Only z-score is required for entry
        if price_zscore <= self.config['zscore_price_threshold']:
            entry_conditions.append("price_stretched")
            signal["reasons"].append(f"zscore_{price_zscore:.2f}")
        else:
            signal["reasons"].append(f"zscore_not_stretched_{price_zscore:.2f}")
            return signal
        
        # Funding rate neutral or negative
        funding_rate = row.get('eth_funding_rate_8h', 0)
        if funding_rate <= self.config['funding_neutral_threshold']:
            entry_conditions.append("funding_neutral")
            signal["reasons"].append(f"funding_{funding_rate:.4%}")
        
        if not entry_conditions:
            signal["reasons"].append("no_entry_conditions")
            return signal
        
        # Step 5: Calculate confirmation score
        confirm_score = self.calculate_confirmation_score(row)
        signal["quality_metrics"]["confirmation_score"] = confirm_score
        
        if confirm_score < self.config['required_confirmation_score']:
            signal["reasons"].append(f"insufficient_confirmation_{confirm_score}")
            return signal
        
        # ✅ ALL CHECKS PASSED - R6 ENTRY TRIGGERED
        signal["entry_triggered"] = True
        
        # Calculate confidence based on z-score depth
        base_conf = min(0.7, abs(price_zscore) / 3.0)  # Cap at 0.7
        signal["confidence"] = min(0.8, base_conf)  # Cap at 0.8
        
        # Map confidence to size (conservative sizing for R6)
        if signal["confidence"] < 0.4:
            signal["position_size"] = 0.25
        elif signal["confidence"] < 0.6:
            signal["position_size"] = 0.5
        else:
            signal["position_size"] = 0.75
        
        signal["reasons"].append("r6_mean_reversion_entry")
        
        return signal
    
    def run_shadow_trading(self, df, days=60, save_results=True):
        """
        Run shadow trading for R6 signals only
        """
        print(f"\n🔍 R6 SHADOW TRADING ({days} days)")
        print("="*50)
        
        # Calculate dynamic trend strength threshold before starting
        self._calculate_trend_strength_threshold(df)
        
        recent_data = df.iloc[-days:].copy()
        signals = []
        entry_count = 0
        r6_active_count = 0
        
        for idx, row in recent_data.iterrows():
            current_regime = row.get('regime_code', 'R0')
            signal = self.generate_r6_signal(row, df, current_regime)
            
            signals.append(signal)
            
            if signal["r6_regime_active"]:
                r6_active_count += 1
                
                if signal["entry_triggered"]:
                    entry_count += 1
                    print(f"✅ R6 LONG: {signal['date']} | Z-score: {signal['quality_metrics'].get('price_zscore', 0):.2f} | "
                          f"Conf: {signal['confidence']:.2f} | Size: {signal['position_size']:.2f}x | "
                          f"Regime: {signal['regime']}")
                else:
                    # Show why active but no entry
                    if "zscore_not_stretched" in str(signal['reasons']):
                        print(f"🔵 R6 ACTIVE (no entry): {signal['date']} | "
                              f"Z-score: {signal['quality_metrics'].get('price_zscore', 0):.2f} | "
                              f"Regime: {signal['regime']} | Reason: {signal['reasons'][-1]}")
        
        # Save results
        if save_results and signals:
            df_signals = pd.DataFrame(signals)
            os.makedirs('validation/r6', exist_ok=True)
            output_file = f'validation/r6/r6_shadow_trading_{days}d.csv'
            df_signals.to_csv(output_file, index=False)
            
            print(f"\n📊 R6 Signal Analysis:")
            print(f"   Total days: {len(signals)}")
            print(f"   R6 regime active: {r6_active_count}")
            print(f"   R6 entries triggered: {entry_count}")
            
            if len(signals) > 0:
                active_pct = r6_active_count/len(signals)*100
                entry_pct = entry_count/len(signals)*100 if len(signals) > 0 else 0
                print(f"   R6 active frequency: {active_pct:.1f}%")
                print(f"   Entry frequency: {entry_pct:.1f}%")
            
            if entry_count > 0:
                avg_confidence = df_signals[df_signals['entry_triggered']]['confidence'].mean()
                avg_zscore = df_signals[df_signals['entry_triggered']]['quality_metrics'].apply(
                    lambda x: x.get('price_zscore', 0) if isinstance(x, dict) else 0
                ).mean()
                
                print(f"   Avg confidence: {avg_confidence:.2f}")
                print(f"   Avg z-score: {avg_zscore:.2f}")
            
            # Show active days breakdown
            active_signals = [s for s in signals if s['r6_regime_active']]
            if active_signals:
                print(f"\n📋 R6 ACTIVE DAYS BREAKDOWN:")
                regime_counts = {}
                for sig in active_signals:
                    regime = sig['regime']
                    regime_counts[regime] = regime_counts.get(regime, 0) + 1
                
                for regime, count in sorted(regime_counts.items()):
                    pct = count / len(active_signals) * 100
                    print(f"   {regime}: {count} days ({pct:.1f}%)")
            
            # FIXED: Updated expected behavior
            print(f"\n🎯 EXPECTED BEHAVIOR (after vol_ratio inversion):")
            print(f"   R6 active days: 6-15 (Got: {r6_active_count})")
            print(f"   Entries: 2-6 (Got: {entry_count})")
            
            if r6_active_count == 0:
                print(f"\n⚠️  DIAGNOSTIC: Zero R6 active days - checking...")
                # Check what's blocking
                reasons = []
                for sig in signals:
                    if not sig['r6_regime_active'] and 'reasons' in sig:
                        reasons.extend(sig['reasons'])
                
                from collections import Counter
                reason_counts = Counter(reasons)
                print(f"   Top blockers:")
                for reason, count in reason_counts.most_common(5):
                    pct = count / len(signals) * 100
                    print(f"     {reason}: {count} days ({pct:.1f}%)")
            
            print(f"\n✅ Results saved to: {output_file}")
        
        return signals