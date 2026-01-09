"""
Shadow Trading Package
Modular system for shadow trading with MAE/MFE analysis
"""

from .shadow_trader import ShadowTrader
from .shadow_analysis import ShadowAnalysis

# Create convenience function that matches the old import pattern
def generate_analysis_report(trades_df, output_path='shadow_trading/shadow_analysis.md'):
    """
    Convenience wrapper around ShadowAnalysis class
    Maintains backward compatibility
    """
    analyzer = ShadowAnalysis()
    
    if trades_df.empty:
        return {"error": "no_trades"}
    
    # Generate report
    report = analyzer.generate_performance_report(trades_df)
    analyzer.save_detailed_analysis(trades_df)
    
    return report

__all__ = ['ShadowTrader', 'ShadowAnalysis', 'generate_analysis_report']
