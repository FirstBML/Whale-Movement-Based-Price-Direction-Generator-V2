def inspect_trend_long_signals(days=60):
    """
    Manual inspection of Trend LONG signals
    """
    print("\n" + "="*70)
    print(f"TREND LONG SIGNAL INSPECTION ({days} days)")
    print("="*70)
    
    # Load pipeline
    df_whales, df_market, df_btc, df_eth, df_funding = load_data_from_files()
    df_features = engineer_features(df_whales, df_market, df_btc, df_eth, df_funding)
    df_pipeline = build_pipeline_complete(df_features)
    
    # Load engine
    engine = rebuild_models_if_needed(df_pipeline)
    
    if not engine:
        print("❌ Could not load engine")
        return None
    
    # Get recent data
    recent_data = df_pipeline.iloc[-days:].copy()
    signals = []
    
    for idx, row in recent_data.iterrows():
        signal = engine.generate_core_signal(row, df_pipeline)
        signals.append(signal)
    
    df_signals = pd.DataFrame(signals)
    valid_signals = df_signals[df_signals['action'] == 'ENTER']
    
    print(f"\n📊 Summary:")
    print(f"   Days analyzed: {len(df_signals)}")
    print(f"   Valid LONG signals: {len(valid_signals)}")
    print(f"   Frequency: {len(valid_signals)/len(df_signals)*100:.1f}%")
    
    return df_signals
