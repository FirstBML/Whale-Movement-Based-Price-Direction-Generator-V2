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
        Log a trade and calculate MAE/MFE over forward window
        
        Args:
            signal: Signal dict from generate_unified_signal()
            df: Full dataframe with price data
            trade_days: Forward window for MAE/MFE calculation
        
        Returns:
            dict: Trade with metrics, or None if failed
        """
        # Extract signal info
        entry_date_str = signal['date']
        direction = signal['direction']
        regime = signal['regime']
        confidence = signal['adjusted_confidence']
        position_size = signal['position_size']
        model_prob = signal['model_probability']
        reasons = ','.join(signal['reasons'])
        
        # FIX: Parse date string properly
        try:
            entry_date = pd.Timestamp(entry_date_str)
        except:
            print(f"⚠️  Could not parse date: {entry_date_str}")
            return None
        
        # FIX: Find matching row with flexible date matching
        # Try exact match first
        date_matches = df[df['block_date'] == entry_date]
        
        if date_matches.empty:
            # Try date-only match (ignore time component)
            df_dates = pd.to_datetime(df['block_date']).dt.date
            entry_date_only = entry_date.date()
            date_matches = df[df_dates == entry_date_only]
        
        if date_matches.empty:
            print(f"⚠️  Entry date {entry_date_str} not found in dataframe")
            print(f"   Dataframe date range: {df['block_date'].min()} to {df['block_date'].max()}")
            return None
        
        # Get trade entry index and price
        trade_idx = date_matches.index[0]
        entry_price = date_matches.iloc[0]['eth_price']
        
        if pd.isna(entry_price):
            print(f"⚠️  Entry price is NaN for {entry_date_str}")
            return None
        
        # Calculate forward window
        end_idx = min(trade_idx + trade_days, len(df) - 1)
        
        if end_idx <= trade_idx:
            print(f"⚠️  Insufficient forward data for {entry_date_str}")
            return None
        
        # Get price window
        price_window = df.iloc[trade_idx:end_idx+1]['eth_price'].values
        
        if len(price_window) < 2:
            print(f"⚠️  Price window too short for {entry_date_str}")
            return None
        
        # Calculate returns (log returns for accuracy)
        try:
            entry_log = np.log(entry_price)
            window_log = np.log(price_window)
            window_returns = window_log - entry_log
            
            # Flip returns for SHORT positions
            if direction == 'SHORT':
                window_returns = -window_returns
            
            # MAE (Maximum Adverse Excursion) - worst drawdown
            mae_return = np.min(window_returns)
            mae_price = entry_price * np.exp(mae_return if direction == 'LONG' else -mae_return)
            
            # MFE (Maximum Favorable Excursion) - best gain
            mfe_return = np.max(window_returns)
            mfe_price = entry_price * np.exp(mfe_return if direction == 'LONG' else -mfe_return)
            
            # Final T+trade_days return
            final_return = window_returns[-1]
            final_price = entry_price * np.exp(final_return if direction == 'LONG' else -final_return)
            
            # Create trade record
            trade = {
                'entry_date': entry_date_str,
                'direction': direction,
                'regime': regime,
                'entry_price': entry_price,
                'position_size': position_size,
                'confidence': confidence,
                'model_prob': model_prob,
                'mae_pct': mae_return * 100,
                'mae_price': mae_price,
                'mfe_pct': mfe_return * 100,
                'mfe_price': mfe_price,
                'final_return_pct': final_return * 100,
                'final_price': final_price,
                'trade_days': len(price_window) - 1,
                'reasons': reasons
            }
            
            self.trades.append(trade)
            return trade
            
        except Exception as e:
            print(f"⚠️  Error calculating metrics for {entry_date_str}: {str(e)[:100]}")
            return None
    
    def save_trades(self):
        """Save trades to CSV"""
        if not self.trades:
            # Create empty file with headers
            empty_df = pd.DataFrame(columns=[
                'entry_date', 'direction', 'regime', 'entry_price', 'position_size',
                'confidence', 'model_prob', 'mae_pct', 'mae_price', 'mfe_pct',
                'mfe_price', 'final_return_pct', 'final_price', 'trade_days', 'reasons'
            ])
            empty_df.to_csv(self.trades_file, index=False)
            print(f"⚠️  No trades to save - created empty file: {self.trades_file}")
            return
        
        df_trades = pd.DataFrame(self.trades)
        df_trades.to_csv(self.trades_file, index=False)
        print(f"✅ Saved {len(self.trades)} trades to {self.trades_file}")
    
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