"""
Shadow Trader Module
Logs and analyzes shadow trades with MAE/MFE tracking
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

class ShadowTrader:
    """Shadow trading system for paper trading with MAE/MFE analysis"""
    
    def __init__(self, output_dir='shadow_trading'):
        self.output_dir = output_dir
        self.trades_file = os.path.join(output_dir, 'shadow_trades_90d.csv')
        self.trades = []
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"✅ ShadowTrader initialized")
        print(f"   Output directory: {output_dir}")
    
        def log_trade(self, signal, df, trade_days=48):
            """
            Log a trade signal with T+48h MAE/MFE tracking
            """
            if signal['action'] != 'ENTER':
                return None
            
            try:
                trade_date = pd.Timestamp(signal['date'])
                
                # Check if date exists in dataframe
                date_matches = df[df['block_date'] == trade_date]
                if date_matches.empty:
                    print(f"⚠️  Warning: Trade date {signal['date']} not found in data")
                    return None
                
                trade_idx = date_matches.index[0]
                
                # Get entry price
                entry_price = df.iloc[trade_idx]['eth_price']
                
                # Calculate metrics using analysis module
                metrics = self.analysis.calculate_trade_metrics(
                    signal['date'], df, entry_price, signal['direction'], trade_days
                )
                
                if metrics is None:
                    return None
                
                # Create trade record - FIXED: Use consistent column names
                trade = {
                    'entry_date': signal['date'],
                    'direction': signal['direction'],
                    'regime': signal['regime'],
                    'entry_price': round(entry_price, 2),
                    'position_size': signal['position_size'],
                    'confidence': round(signal['adjusted_confidence'], 3),
                    'model_prob': round(signal['model_probability'], 3),
                    'reasons': '|'.join(signal['reasons']),
                    'mae_pct': round(metrics['mae_pct'], 2),
                    'mae_price': round(metrics['mae_price'], 2),
                    'mfe_pct': round(metrics['mfe_pct'], 2),
                    'mfe_price': round(metrics['mfe_price'], 2),
                    'final_pct': round(metrics['final_pct'], 2),  # FIXED: Use final_pct consistently
                    'final_price': round(metrics['final_price'], 2),
                    'funding_rate': signal.get('funding_rate', None),
                    'funding_available': signal.get('funding_available', False),
                    'log_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'trade_days': metrics.get('trade_days', trade_days)
                }
                
                self.trades.append(trade)
                return trade
                
            except Exception as e:
                print(f"⚠️  Error logging trade for {signal['date']}: {str(e)[:100]}")
                return None
    
    def save_trades(self):
        """Save all trades to CSV - ALWAYS creates file"""
        # Always create the directory
        os.makedirs('shadow_trading', exist_ok=True)
        
        if not self.trades:
            print("⚠️ No trades to save")
            # Create empty CSV file
            empty_df = pd.DataFrame()
            empty_df.to_csv(self.log_file, index=False)
            print(f"✅ Created empty file: {self.log_file}")
            return
        
        # Create DataFrame
        df_trades = pd.DataFrame(self.trades)
        
        # Ensure all required columns exist
        required_columns = ['entry_date', 'direction', 'regime', 'entry_price', 
                          'position_size', 'confidence', 'model_prob', 'mae_pct', 
                          'mae_price', 'mfe_pct', 'mfe_price', 'final_pct', 'final_price',
                          'reasons', 'funding_rate', 'funding_available', 'log_time', 'trade_days']
        
        # Add missing columns
        for col in required_columns:
            if col not in df_trades.columns:
                df_trades[col] = None if col != 'trade_days' else 48
        
        # Reorder columns for consistency
        df_trades = df_trades[required_columns]
        
        # Save to CSV
        df_trades.to_csv(self.log_file, index=False)
        print(f"✅ Saved {len(self.trades)} shadow trades to {self.log_file}")
        
        # Print summary
        self.analysis.print_summary(df_trades)
        
        # Save detailed analysis
        try:
            self.analysis.save_detailed_analysis(df_trades)
        except Exception as e:
            print(f"⚠️  Could not save detailed analysis: {e}")
    
    def get_performance_report(self):
        """Generate performance report using ShadowAnalysis class"""
        if not self.trades:
            return {"error": "no_trades"}
        
        # Import here to avoid circular dependency
        from .shadow_analysis import ShadowAnalysis
        
        df_trades = pd.DataFrame(self.trades)
        analyzer = ShadowAnalysis(self.output_dir)
        
        # Generate report
        report = analyzer.generate_performance_report(df_trades)
        analyzer.save_detailed_analysis(df_trades)
        
        return report