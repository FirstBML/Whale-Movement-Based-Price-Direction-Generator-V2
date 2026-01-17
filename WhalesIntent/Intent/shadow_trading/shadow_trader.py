"""
shadow_trader.py - Modular Shadow Trading System
Fixed version with log_trade method
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
from typing import Dict, List, Optional

class ShadowTrader:
    def __init__(self):
        self.trades = []
        self.trade_id = 1
        
    def log_trade(self, signal: Dict, price_data: pd.DataFrame, trade_days: int = 48) -> Optional[Dict]:
        """
        Log a trade and calculate MAE/MFE over trade_days horizon
        """
        try:
            # Check if signal is valid
            if signal.get('action') != 'ENTER':
                return None
            
            # Get entry date - ensure it's timezone-naive
            entry_date = pd.Timestamp(signal['date'])
            if entry_date.tz is not None:
                entry_date = entry_date.tz_localize(None)
            
            # Find matching date in price_data
            if 'block_date' in price_data.columns:
                date_col = 'block_date'
            elif 'date' in price_data.columns:
                date_col = 'date'
            else:
                print(f"❌ No date column found in price_data")
                return None
            
            # Convert date columns to datetime and remove timezone if present
            if not pd.api.types.is_datetime64_any_dtype(price_data[date_col]):
                price_data[date_col] = pd.to_datetime(price_data[date_col])
            
            # Remove timezone from price_data dates for comparison
            price_data[date_col] = price_data[date_col].dt.tz_localize(None)
            
            # Find entry index
            entry_mask = price_data[date_col] == entry_date
            if not entry_mask.any():
                # Try to find nearest date
                entry_mask = price_data[date_col] <= entry_date
                if not entry_mask.any():
                    print(f"❌ No price data found for {entry_date}")
                    return None
                entry_idx = price_data[entry_mask].index[-1]
            else:
                entry_idx = price_data[entry_mask].index[0]
            
            # Get entry price
            if 'eth_price' not in price_data.columns:
                print(f"❌ No eth_price column in price_data")
                return None
            
            entry_price = price_data.iloc[entry_idx]['eth_price']
        
            end_idx = min(entry_idx + trade_days, len(price_data) - 1)
            
            if end_idx <= entry_idx:
                print(f"❌ Insufficient data for forward window")
                return None
            
            # Extract forward prices
            forward_data = price_data.iloc[entry_idx:end_idx + 1]
            forward_prices = forward_data['eth_price'].values
            
            if len(forward_prices) < 2:
                print(f"❌ Insufficient forward prices")
                return None
            
            # Calculate returns
            returns = (forward_prices / entry_price - 1) * 100
            
            # For LONG: positive returns are good
            # For SHORT: negative returns are good
            if signal['direction'] == 'LONG':
                mae = np.min(returns)  # Maximum Adverse Excursion (worst drawdown)
                mfe = np.max(returns)  # Maximum Favorable Excursion (best gain)
                final_return = returns[-1]  # Final return at end of window
            else:  # SHORT
                # Invert returns for SHORT positions
                returns = -returns
                mae = np.min(returns)
                mfe = np.max(returns)
                final_return = returns[-1]
            
            # Create trade record
            trade = {
                'trade_id': self.trade_id,
                'entry_date': signal['date'],
                'direction': signal['direction'],
                'regime': signal.get('regime', 'UNKNOWN'),
                'entry_price': entry_price,
                'confidence': signal.get('adjusted_confidence', 0),
                'position_size': signal.get('position_size', 0),
                'reasons': '|'.join(signal.get('reasons', [])),
                'mae_pct': float(mae),
                'mfe_pct': float(mfe),
                'final_pct': float(final_return),
                'trade_days': trade_days,
                'window_days': len(forward_prices) - 1,
                'max_price': float(np.max(forward_prices)),
                'min_price': float(np.min(forward_prices)),
                'signal_details': json.dumps(signal)
            }
            
            # Add funding data if available
            if signal.get('funding_available'):
                trade['funding_rate'] = signal.get('funding_rate')
            
            self.trades.append(trade)
            self.trade_id += 1
            
            return trade
            
        except Exception as e:
            print(f"❌ Error logging trade: {e}")
            return None
    
    def save_trades(self, filename: str = 'shadow_trading/shadow_trades_90d.csv'):
        """Save trades to CSV file"""
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            if self.trades:
                df = pd.DataFrame(self.trades)
                df.to_csv(filename, index=False)
                print(f"✅ Saved {len(df)} trades to {filename}")
                
                # Also save as JSON for detailed analysis
                json_file = filename.replace('.csv', '.json')
                with open(json_file, 'w') as f:
                    json.dump(self.trades, f, indent=2)
                print(f"✅ Saved detailed trades to {json_file}")
            else:
                print(f"⚠️  No trades to save")
        except Exception as e:
            print(f"❌ Error saving trades: {e}")
    
    def get_performance_report(self):
        """Generate performance report - JSON-safe version"""
        if not self.trades:
            return {"error": "No trades logged"}
        
        try:
            df = pd.DataFrame(self.trades)
            
            report = {
                'total_trades': int(len(df)),
                'long_trades': int((df['direction'] == 'LONG').sum()),
                'short_trades': int((df['direction'] == 'SHORT').sum()),
                'avg_mae': float(df['mae_pct'].mean()),
                'avg_mfe': float(df['mfe_pct'].mean()),
                'avg_final_return': float(df['final_pct'].mean()),
                'win_rate': float((df['final_pct'] > 0).sum()) / float(len(df)) if len(df) > 0 else 0.0,
                'sharpe_ratio': float(df['final_pct'].mean()) / float(df['final_pct'].std()) if float(df['final_pct'].std()) > 0 else 0.0,
                'liquidation_risk_3x': int((df['mae_pct'] < -33.33).sum())
            }
            
            # Convert regime distribution safely
            regime_dist = df['regime'].value_counts().to_dict()
            report['regime_distribution'] = {str(k): int(v) for k, v in regime_dist.items()}
            
            # Best and worst trades
            if len(df) > 0:
                best_idx = df['final_pct'].idxmax()
                worst_idx = df['final_pct'].idxmin()
                
                report['best_trade'] = {
                    'date': str(df.loc[best_idx, 'entry_date']),
                    'direction': str(df.loc[best_idx, 'direction']),
                    'return': float(df.loc[best_idx, 'final_pct'])
                }
                
                report['worst_trade'] = {
                    'date': str(df.loc[worst_idx, 'entry_date']),
                    'direction': str(df.loc[worst_idx, 'direction']),
                    'return': float(df.loc[worst_idx, 'final_pct'])
                }
            
            return report
            
        except Exception as e:
            return {"error": f"Error generating report: {str(e)}"}
            
    def clear_trades(self):
        """Clear all logged trades"""
        self.trades = []
        self.trade_id = 1
        print("✅ Cleared all trades")