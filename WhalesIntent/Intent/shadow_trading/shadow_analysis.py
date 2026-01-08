"""
Shadow Trading Analysis Module
Handles MAE/MFE analysis and performance reporting for shadow trades
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

class ShadowAnalysis:
    """Shadow trading analysis and reporting"""
    
    def __init__(self, output_dir='shadow_trading'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def calculate_trade_metrics(self, trade_date, df, entry_price, direction, trade_days=48):
        """
        Calculate MAE, MFE, and final returns for a trade
        """
        try:
            # Find trade index
            date_matches = df[df['block_date'] == pd.Timestamp(trade_date)]
            if date_matches.empty:
                return None
            
            trade_idx = date_matches.index[0]
            
            # Get entry price
            if pd.isna(entry_price):
                entry_price = df.iloc[trade_idx]['eth_price']
            
            # Calculate T+48h window
            end_idx = min(trade_idx + trade_days, len(df) - 1)
            
            if end_idx <= trade_idx:
                return None
            
            # Get price window
            price_window = df.iloc[trade_idx:end_idx+1]['eth_price'].values
            
            if len(price_window) == 0:
                return None
            
            # Calculate returns in window
            entry_log = np.log(entry_price)
            window_log = np.log(price_window)
            window_returns = window_log - entry_log
            
            # MAE (Maximum Adverse Excursion) - worst loss
            mae_return = np.min(window_returns)
            mae_price = entry_price * np.exp(mae_return)
            
            # MFE (Maximum Favorable Excursion) - best gain
            mfe_return = np.max(window_returns)
            mfe_price = entry_price * np.exp(mfe_return)
            
            # Final T+48h return
            final_return = window_returns[-1]
            final_price = entry_price * np.exp(final_return)
            
            return {
                'mae_pct': mae_return * 100,
                'mae_price': mae_price,
                'mfe_pct': mfe_return * 100,
                'mfe_price': mfe_price,
                'final_pct': final_return * 100,
                'final_price': final_price,
                'trade_days': len(price_window) - 1
            }
            
        except Exception as e:
            print(f"⚠️  Error calculating metrics: {str(e)[:100]}")
            return None
    
    def print_summary(self, df_trades):
        """Print shadow trading summary"""
        print("\n" + "="*70)
        print("SHADOW TRADING SUMMARY")
        print("="*70)
        
        if len(df_trades) == 0:
            print("No trades recorded")
            return
        
        # Handle column name variations
        final_col = None
        for col in ['final_pct', 'final_return_pct', 'final_return']:
            if col in df_trades.columns:
                final_col = col
                break
        
        if final_col is None:
            print("⚠️  Could not find final return column")
            return
        
        # Rename for consistency
        df_trades = df_trades.copy()
        df_trades['final_pct'] = df_trades[final_col]
        
               
        # Basic stats
        print(f"Total trades: {len(df_trades)}")
        print(f"LONG trades: {(df_trades['direction'] == 'LONG').sum()}")
        print(f"SHORT trades: {(df_trades['direction'] == 'SHORT').sum()}")
        
        # Regime distribution
        print("\nRegime Distribution:")
        regime_counts = df_trades['regime'].value_counts()
        for regime, count in regime_counts.items():
            print(f"  {regime}: {count} trades")
        
        # MAE Analysis
        print("\nMAE (Maximum Adverse Excursion):")
        print(f"  Average MAE: {df_trades['mae_pct'].mean():.2f}%")
        print(f"  Max MAE: {df_trades['mae_pct'].min():.2f}%")
        print(f"  Min MAE: {df_trades['mae_pct'].max():.2f}%")
        
        # MFE Analysis
        print("\nMFE (Maximum Favorable Excursion):")
        print(f"  Average MFE: {df_trades['mfe_pct'].mean():.2f}%")
        print(f"  Max MFE: {df_trades['mfe_pct'].max():.2f}%")
        print(f"  Min MFE: {df_trades['mfe_pct'].min():.2f}%")
        
        # Final returns
        print("\nFinal T+48h Returns:")
        print(f"  Average return: {df_trades['final_pct'].mean():.2f}%")
        print(f"  Positive returns: {(df_trades['final_pct'] > 0).sum()}/{len(df_trades)}")
        
        # Confidence vs Performance
        print("\nConfidence vs Performance:")
        if len(df_trades) >= 5:
            # Top 25% confidence trades
            high_conf = df_trades.nlargest(int(len(df_trades) * 0.25), 'confidence')
            low_conf = df_trades.nsmallest(int(len(df_trades) * 0.25), 'confidence')
            
            print(f"  High confidence (top 25%): {high_conf['final_pct'].mean():.2f}% avg return")
            print(f"  Low confidence (bottom 25%): {low_conf['final_pct'].mean():.2f}% avg return")
        
        # Risk metrics
        print("\nRisk Assessment:")
        worst_trade = df_trades.loc[df_trades['mae_pct'].idxmin()]
        print(f"  Worst MAE: {worst_trade['mae_pct']:.2f}% on {worst_trade['entry_date']}")
        
        # Check for liquidation risk (assuming 3x leverage)
        liquidation_risk = df_trades[df_trades['mae_pct'] < -33.33]
        if len(liquidation_risk) > 0:
            print(f"  WARNING: {len(liquidation_risk)} trades would liquidate at 3x leverage")
    
    def save_detailed_analysis(self, df_trades):
        """Save detailed analysis to file"""
        analysis_file = os.path.join(self.output_dir, 'shadow_analysis.md')
        
        try:
            # Use UTF-8 encoding to handle special characters
            with open(analysis_file, 'w', encoding='utf-8') as f:
                f.write("# SHADOW TRADING ANALYSIS\n\n")
                f.write(f"**Period:** {df_trades['entry_date'].min()} to {df_trades['entry_date'].max()}\n")
                f.write(f"**Total trades:** {len(df_trades)}\n\n")
                
                f.write("## Performance Questions\n\n")
                f.write("1. **Are SHORT signals rare but violent?**\n")
                shorts = df_trades[df_trades['direction'] == 'SHORT']
                if len(shorts) > 0:
                    avg_short_mfe = shorts['mfe_pct'].mean()
                    f.write(f"   - SHORT count: {len(shorts)} ({len(shorts)/len(df_trades)*100:.1f}%)\n")
                    f.write(f"   - Average SHORT MFE: {avg_short_mfe:.2f}%\n")
                    # Use plain text indicators instead of emojis
                    result = "YES" if avg_short_mfe > 5 else "Needs review"
                    f.write(f"   - Result: {result}\n\n")
                
                f.write("2. **Do SHORTs cluster near distribution / breakdowns?**\n")
                f.write("   - Manual review needed: Check R5 regime trades\n\n")
                
                f.write("3. **Are rejected signals obviously bad in hindsight?**\n")
                f.write("   - Manual review needed: Compare NO_TRADE days with subsequent price action\n\n")
                
                f.write("4. **Does higher confidence → better MFE?**\n")
                if len(df_trades) >= 10:
                    corr = df_trades['confidence'].corr(df_trades['mfe_pct'])
                    result = "YES" if corr > 0.1 else "No clear relationship"
                    f.write(f"   - Correlation (confidence vs MFE): {corr:.3f}\n")
                    f.write(f"   - Result: {result}\n\n")
                
                f.write("## Trade Details\n\n")
                for _, trade in df_trades.iterrows():
                    f.write(f"### {trade['entry_date']} - {trade['direction']} in {trade['regime']}\n")
                    f.write(f"- Entry: ${trade['entry_price']:.0f}, Size: {trade['position_size']:.2f}\n")
                    f.write(f"- Confidence: {trade['confidence']:.3f}, Model prob: {trade['model_prob']:.3f}\n")
                    f.write(f"- MAE: {trade['mae_pct']:.2f}% (${trade['mae_price']:.0f})\n")
                    f.write(f"- MFE: {trade['mfe_pct']:.2f}% (${trade['mfe_price']:.0f})\n")
                    f.write(f"- Final T+48h: {trade['final_pct']:.2f}% (${trade['final_price']:.0f})\n")
                    f.write(f"- Reasons: {trade['reasons']}\n\n")
            
            print(f"✅ Detailed analysis saved to {analysis_file}")
            
        except Exception as e:
            print(f"❌ Failed to save detailed analysis: {e}")
            # Try a simpler version without special characters
            try:
                with open(analysis_file, 'w') as f:
                    f.write(f"SHADOW TRADING ANALYSIS\n")
                    f.write(f"Period: {df_trades['entry_date'].min()} to {df_trades['entry_date'].max()}\n")
                    f.write(f"Total trades: {len(df_trades)}\n")
                print(f"✅ Saved basic analysis (no formatting)")
            except:
                print(f"❌ Could not save any analysis file")
    
    def generate_performance_report(self, df_trades):
        """
        Generate comprehensive performance report
        Returns dictionary with key metrics
        """
        if len(df_trades) == 0:
            return {"error": "No trades to analyze"}
        
        report = {
            "total_trades": len(df_trades),
            "long_trades": (df_trades['direction'] == 'LONG').sum(),
            "short_trades": (df_trades['direction'] == 'SHORT').sum(),
            "avg_mae": float(df_trades['mae_pct'].mean()),
            "avg_mfe": float(df_trades['mfe_pct'].mean()),
            "avg_final_return": float(df_trades['final_pct'].mean()),
            "win_rate": float((df_trades['final_pct'] > 0).sum() / len(df_trades)),
            "regime_distribution": df_trades['regime'].value_counts().to_dict(),
            "liquidation_risk_3x": (df_trades['mae_pct'] < -33.33).sum()
        }
        
        # Additional metrics
        if report['total_trades'] >= 5:
            # Risk-adjusted returns
            sharpe_ratio = report['avg_final_return'] / (df_trades['final_pct'].std() + 1e-10)
            report['sharpe_ratio'] = float(sharpe_ratio)
            
            # Best and worst trades
            report['best_trade'] = {
                'date': df_trades.loc[df_trades['final_pct'].idxmax(), 'entry_date'],
                'return': float(df_trades['final_pct'].max()),
                'direction': df_trades.loc[df_trades['final_pct'].idxmax(), 'direction']
            }
            
            report['worst_trade'] = {
                'date': df_trades.loc[df_trades['final_pct'].idxmin(), 'entry_date'],
                'return': float(df_trades['final_pct'].min()),
                'direction': df_trades.loc[df_trades['final_pct'].idxmin(), 'direction']
            }
        
        return report