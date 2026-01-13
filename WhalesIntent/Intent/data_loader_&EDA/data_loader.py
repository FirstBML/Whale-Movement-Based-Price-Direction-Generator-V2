### DATA LOADING MODULE

import os
import time
import json
import warnings
import requests
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
load_dotenv()

# CONFIGURATION 
DUNE_API_KEYS = [
    os.getenv("DUNE_LASEVEN7"),
    os.getenv("DUNE_FIRSTBML"),
    os.getenv("DUNE_LASEVEN71"),
    os.getenv("DUNE_LASEVEN7_TEAM"),
    os.getenv("DUNE_FIRSTBML_TEAM")
]

DUNE_API_KEYS = [key for key in DUNE_API_KEYS if key and str(key).strip()]
print("🔍 Loaded API Keys:")
env_names = [
    "DUNE_LASEVEN7", 
    "DUNE_FIRSTBML",
    "DUNE_LASEVEN71",
    "DUNE_LASEVEN7_TEAM",
    "DUNE_FIRSTBML_TEAM"
]
for i, key in enumerate(DUNE_API_KEYS):
    env_name = env_names[i] if i < len(env_names) else "UNKNOWN"
    print(f"   ✅ {i+1}. {env_name}: {key[:8]}...")

if not DUNE_API_KEYS:
    raise ValueError("No Dune API keys found!")

COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY")

for d in ['data', 'data/price_cache', 'logs']:
    os.makedirs(d, exist_ok=True)

QUERIES = {
    "whales": ("6395391", "data/dune_whales_cache.json", "data/whale_ml_ready.csv"),
    "market_intent": ("6385600", "data/dune_intent_cache.json", "data/market_intent_ml_ready.csv")
}

DUNE_START = pd.Timestamp('2017-10-16', tz='UTC')
COINGECKO_BTC_START = '01-01-2013'
COINGECKO_ETH_START = '01-08-2015'

# KEY ROTATION

class DuneKeyRotator:
    def __init__(self, api_keys):
        self.api_keys = api_keys
        self.key_index = 0
        self.key_usage = {key: {"count": 0, "last_used": None, "errors": 0, "exhausted": False} for key in api_keys}
        self.total_requests = 0
    
    def get_next_key(self):
        if not self.api_keys:
            raise ValueError("No API keys available")
        
        available_keys = []
        for key in self.api_keys:
            usage = self.key_usage[key]
            
            if usage["exhausted"]:  # Skip exhausted keys
                continue
                
            if usage["errors"] >= 3:
                continue
                
            if usage["last_used"]:
                time_since_use = (datetime.now() - usage["last_used"]).total_seconds()
                if time_since_use < 60:
                    continue
                    
            available_keys.append((key, usage["count"], usage["errors"]))
        
        if not available_keys:
            # Check if ALL keys are exhausted
            if all(usage["exhausted"] for usage in self.key_usage.values()):
                raise Exception("ALL_API_KEYS_EXHAUSTED")
            
            available_keys = [(key, self.key_usage[key]["count"], self.key_usage[key]["errors"]) 
                            for key in self.api_keys if not self.key_usage[key]["exhausted"]]
        
        if not available_keys:
            raise Exception("NO_KEYS_AVAILABLE")
        
        available_keys.sort(key=lambda x: (x[2], x[1]))
        selected_key = available_keys[0][0]
        
        self.key_usage[selected_key]["count"] += 1
        self.key_usage[selected_key]["last_used"] = datetime.now()
        self.total_requests += 1
        
        return selected_key
    
    def mark_exhausted(self, key):
        """Mark a key as exhausted (reached monthly limit)"""
        if key in self.key_usage:
            self.key_usage[key]["exhausted"] = True
            print(f"⛔ KEY {key[:8]}... MARKED AS EXHAUSTED (monthly limit reached)")
    
    def are_all_keys_exhausted(self):
        """Check if all keys are exhausted"""
        return all(usage["exhausted"] for usage in self.key_usage.values())
        
    def mark_error(self, key):
        if key in self.key_usage:
            self.key_usage[key]["errors"] += 1
    
    def mark_success(self, key):
        if key in self.key_usage and self.key_usage[key]["errors"] > 0:
            self.key_usage[key]["errors"] = max(0, self.key_usage[key]["errors"] - 1)
    
    def get_stats(self):
        return {
            "total_requests": self.total_requests,
            "active_keys": sum(1 for key, stats in self.key_usage.items() if stats["errors"] < 5)
        }

key_rotator = DuneKeyRotator(DUNE_API_KEYS)
# API CLIENT

class DuneAPIClient:
    def __init__(self, key_rotator):
        self.key_rotator = key_rotator
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
    
    def make_request(self, method, url, **kwargs):
        max_retries = len(self.key_rotator.api_keys)  # Try all keys
        
        for attempt in range(max_retries):
            try:
                api_key = self.key_rotator.get_next_key()
                headers = kwargs.get('headers', {}).copy()
                headers["x-dune-api-key"] = api_key
                kwargs['headers'] = headers
                
                response = self.session.request(method, url, **kwargs)
                
                if response.status_code == 402:
                    # This key is exhausted for the month
                    self.key_rotator.mark_exhausted(api_key)
                    self.key_rotator.mark_error(api_key)
                    
                    # Check if all keys are now exhausted
                    if self.key_rotator.are_all_keys_exhausted():
                        return response  # Return 402 to signal complete exhaustion
                    
                    continue  # Try another key
                    
                elif response.status_code == 200:
                    self.key_rotator.mark_success(api_key)
                    return response
                    
                else:
                    self.key_rotator.mark_error(api_key)
                    
                    if attempt == max_retries - 1:
                        return response
                        
                    time.sleep(2)
                    
            except Exception as e:
                self.key_rotator.mark_error(api_key)
                
                if attempt == max_retries - 1:
                    raise
        
        raise Exception(f"Failed after {max_retries} attempts")
    
    def get(self, url, **kwargs):
        return self.make_request("GET", url, **kwargs)
    
    def post(self, url, **kwargs):
        return self.make_request("POST", url, **kwargs)

dune_client = DuneAPIClient(key_rotator)


# UTILITY FUNCTIONS
 # Dune

def to_utc(ts):
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def fetch_dune_incremental(qid, cache_path, query_name="whale_data", force_fetch=False):
    """
    Stops on rate limit instead of skipping dates
    """
    today = pd.Timestamp.now(tz='UTC').normalize()
    yesterday = today - pd.Timedelta(days=1)
    
    print(f"\n📊 Fetching {query_name}...")
    print(f"   Using {len(DUNE_API_KEYS)} API keys")
    
    df_cached = pd.DataFrame()
    last_date = None
    
    if os.path.exists(cache_path) and not force_fetch:
        try:
            with open(cache_path) as f:
                cached = json.load(f)
            
            if 'data' in cached and cached['data']:
                df_cached = pd.DataFrame(cached["data"])
                
                if 'block_date' in df_cached.columns:
                    df_cached["block_date"] = pd.to_datetime(df_cached["block_date"], utc=True)
                    
                    if 'is_estimate' in df_cached.columns:
                        df_cached = df_cached.drop('is_estimate', axis=1)
                    
                    last_date = df_cached["block_date"].max()
                    
                    if last_date >= yesterday:
                        print(f"✅ {query_name} cache current ({last_date.date()})")
                        return df_cached
                    
                    print(f"📅 Cache: {last_date.date()}, fetching new data...")
                else:
                    last_date = DUNE_START
            else:
                last_date = DUNE_START
                
        except Exception as e:
            print(f"⚠️  Cache error: {e}")
            last_date = DUNE_START
    else:
        print(f"📝 Fetching from {DUNE_START.date()}...")
        last_date = DUNE_START
    
    fetch_start = last_date + pd.Timedelta(days=1) if last_date else DUNE_START
    fetch_end = yesterday
    
    if fetch_start > fetch_end:
        return df_cached
    
    print(f"🔍 Fetching: {fetch_start.date()} to {fetch_end.date()}")
    
    # FETCH ONE DAY AT A TIME
    all_new_rows = []
    current_date = fetch_start
    
    while current_date <= fetch_end:
        query_params = {
            "start_date": current_date.strftime("%Y-%m-%d"),
            "end_date": current_date.strftime("%Y-%m-%d")
        }
        
        try:
            execute_url = f"https://api.dune.com/api/v1/query/{qid}/execute"
            execute_payload = {"query_parameters": query_params}
            
            print(f"   🔸 {current_date.date()}...", end="")
            
            resp = dune_client.post(execute_url, json=execute_payload, timeout=60)
            
            # ✅ Stop on 402, don't skip
            if resp.status_code != 200:
                if resp.status_code == 402:
                    print(f" ❌ 402 RATE LIMIT")
                    print(f"\n⚠️  STOPPED at {current_date.date()} - rate limit hit")
                    print(f"   Cache saved up to: {last_date.date() if last_date else 'N/A'}")
                    print(f"   Resume tomorrow when limits reset")
                    break  # STOP completely, don't skip
                
                print(f" ❌ {resp.status_code}")
                current_date += pd.Timedelta(days=1)
                time.sleep(2)
                continue
            
            resp_json = resp.json()
            
            if 'execution_id' not in resp_json:
                print(f" ❌ No execution_id")
                current_date += pd.Timedelta(days=1)
                time.sleep(2)
                continue
            
            eid = resp_json["execution_id"]
            
            # Wait for completion
            for attempt in range(60):
                status_url = f"https://api.dune.com/api/v1/execution/{eid}/status"
                
                try:
                    status_resp = dune_client.get(status_url, timeout=30)
                    
                    if status_resp.status_code == 200:
                        status_data = status_resp.json()
                        state = status_data.get("state", "UNKNOWN")
                        
                        if state == "QUERY_STATE_COMPLETED":
                            break
                        elif state in ["QUERY_STATE_FAILED", "QUERY_STATE_CANCELLED"]:
                            print(f" ❌ {state}")
                            break
                    
                except Exception as e:
                    print(f" ⚠️ {e}")
                    break
                
                time.sleep(5)
            else:
                print(f" ⏱️ Timeout")
                current_date += pd.Timedelta(days=1)
                time.sleep(2)
                continue
            
            # Get results
            results_url = f"https://api.dune.com/api/v1/execution/{eid}/results"
            results_resp = dune_client.get(results_url, timeout=30)
            
            if results_resp.status_code == 200:
                results_data = results_resp.json()
                
                if 'result' in results_data and 'rows' in results_data['result']:
                    rows = results_data["result"]["rows"]
                    if rows:
                        all_new_rows.extend(rows)
                        print(f" ✅ {len(rows)} rows")
                    else:
                        print(f" ⚠️ No data")
                else:
                    print(f" ❌ No results")
            else:
                print(f" ❌ {results_resp.status_code}")
            
        except Exception as e:
            print(f" ❌ {str(e)[:50]}")
        
        current_date += pd.Timedelta(days=1)
        time.sleep(2)
    
    if not all_new_rows:
        print(f"⚠️  No new data fetched")
        return df_cached
    
    df_new = pd.DataFrame(all_new_rows)
    print(f"📥 Total: {len(df_new)} new rows")
    
    if 'block_date' not in df_new.columns:
        print(f"❌ Missing block_date!")
        return df_cached
    
    df_new["block_date"] = pd.to_datetime(df_new["block_date"], utc=True)
    
    # Merge with cache
    if not df_cached.empty:
        common_cols = list(set(df_cached.columns) & set(df_new.columns))
        df_cached = df_cached[common_cols]
        df_new = df_new[common_cols]
        
        df_combined = pd.concat([df_cached, df_new], ignore_index=True)
        df_combined = df_combined.drop_duplicates(
            subset=['block_date'],
            keep='last'
        ).sort_values('block_date').reset_index(drop=True)
        
        print(f"📊 Combined: {len(df_combined)} rows")
    else:
        df_combined = df_new
        print(f"📊 New dataset: {len(df_combined)} rows")
    
    # Update cache
    update_cache_file(cache_path, df_combined)
    
    return df_combined
def update_cache_file(cache_path, df):
    try:
        if 'block_date' in df.columns:
            df_dates = df['block_date'].copy()
            
            df_serializable = df.copy()
            df_serializable['block_date'] = df_serializable['block_date'].dt.strftime('%Y-%m-%d')
            
            with open(cache_path, 'w') as f:
                json.dump({
                    "last_block_date": df_dates.max().strftime("%Y-%m-%d"),
                    "data": json.loads(df_serializable.to_json(orient="records", date_format='iso'))
                }, f, indent=2)
            
            print(f"💾 Cache updated to {df_dates.max().date()}")
            
    except Exception as e:
        print(f"❌ Cache update failed: {e}")

# Data Fetching for Bitcoin and Etherum from Coingecko

def fetch_cg_chunked(cg_id, start_date_str, end_date, key, days=30):
    url = "https://pro-api.coingecko.com/api/v3"
    headers = {"x-cg-pro-api-key": key}
    
    if isinstance(start_date_str, str):
        try:
            start_dt = pd.to_datetime(start_date_str, format='%d-%m-%Y', utc=True)
        except:
            start_dt = pd.to_datetime(start_date_str, utc=True)
    else:
        start_dt = to_utc(start_date_str)
    
    end_dt = to_utc(end_date) + pd.Timedelta(days=1)
    
    all_prices, curr = [], start_dt
    
    while curr < end_dt:
        next_dt = min(curr + pd.Timedelta(days=days), end_dt)
        params = {
            "vs_currency": "usd", 
            "from": int(curr.timestamp()), 
            "to": int(next_dt.timestamp())
        }
        
        try:
            r = requests.get(
                f"{url}/coins/{cg_id}/market_chart/range", 
                params=params, 
                headers=headers, 
                timeout=30
            )
            
            if r.status_code == 200:
                prices = r.json().get("prices", [])
                if prices:
                    all_prices.extend(prices)
            
        except Exception:
            pass
        
        time.sleep(0.5)
        curr = next_dt
    
    if not all_prices:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_prices, columns=["timestamp", "price"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.floor("D")
    return df.groupby("date")["price"].mean().reset_index()
def get_price_incremental(symbol, cg_id, start_date_str, end, key, force_fetch=False):
    cache_path = f"data/price_cache/{symbol}.csv"
    today_utc = pd.Timestamp.utcnow().floor("D")
    yesterday = today_utc - pd.Timedelta(days=1)
    end_dt = min(to_utc(end), yesterday)
    
    print(f"\n💰 Fetching {symbol.upper()} prices...")
    
    if force_fetch and os.path.exists(cache_path):
        os.remove(cache_path)
    
    df_cached = pd.DataFrame()
    if os.path.exists(cache_path) and not force_fetch:
        try:
            df_cached = pd.read_csv(cache_path, parse_dates=["date"])
            df_cached["date"] = df_cached["date"].apply(to_utc)
            
            if not df_cached.empty:
                last_date = df_cached["date"].max()
                first_date = df_cached["date"].min()
                expected_start = to_utc(start_date_str)
                
                needs_historical = first_date > expected_start
                needs_updates = last_date < end_dt
                
                if not needs_historical and not needs_updates:
                    print(f"✅ {symbol.upper()} current ({first_date.date()} to {last_date.date()})")
                    return df_cached
                
                fetch_ranges = []
                
                if needs_historical:
                    fetch_ranges.append((expected_start, first_date - pd.Timedelta(days=1)))
                
                if needs_updates:
                    fetch_ranges.append((last_date + pd.Timedelta(days=1), end_dt))
                
                all_new_data = []
                for fetch_start, fetch_end in fetch_ranges:
                    if fetch_start <= fetch_end:
                        new_data = fetch_cg_chunked(cg_id, fetch_start, fetch_end, key)
                        if not new_data.empty:
                            all_new_data.append(new_data)
                
                if not all_new_data:
                    return df_cached
                
                df_new = pd.concat(all_new_data, ignore_index=True)
                df_new = df_new.rename(columns={"price": f"{symbol}_price"})
                
                df_combined = pd.concat([df_cached, df_new], ignore_index=True)
                df_combined = df_combined.drop_duplicates("date").sort_values("date").reset_index(drop=True)
                
                print(f"📊 {symbol.upper()}: {len(df_combined)} total")
                
        except Exception as e:
            print(f"⚠️  Cache error: {e}")
            df_cached = pd.DataFrame()
            fetch_start = to_utc(start_date_str)
    else:
        fetch_start = to_utc(start_date_str)
    
    if df_cached.empty:
        new_data = fetch_cg_chunked(cg_id, fetch_start, end_dt, key)
        
        if new_data.empty:
            return pd.DataFrame()
        
        df_combined = new_data.rename(columns={"price": f"{symbol}_price"})
    
    df_combined.to_csv(cache_path, index=False)
    print(f"💾 {symbol.upper()} saved: {df_combined['date'].min().date()} to {df_combined['date'].max().date()}")
    
    return df_combined

# BINANCE FUNDING RATE CONFIGURATION
CROWDED_LONG = 0.03    # +3% per 8h (very aggressive longs)
CROWDED_SHORT = -0.02  # -2% per 8h (short squeeze risk)

def fetch_binance_funding_incremental(symbol="ETHUSDT", start_date=None, end_date=None, force_fetch=False):
    """
    Fetch Binance funding rate data incrementally with caching
    """
    cache_path = "data/funding_rates_cache.json"
    cache_file = "data/funding_rates.csv"
    
    print(f"\n💰 Fetching Binance {symbol} funding rates...")
    
    # Load cached data if exists
    df_cached = pd.DataFrame()
    last_date = None
    
    if os.path.exists(cache_file) and not force_fetch:
        try:
            df_cached = pd.read_csv(cache_file, parse_dates=["funding_time"])
            df_cached["funding_time"] = df_cached["funding_time"].apply(to_utc)
            
            if not df_cached.empty:
                last_date = df_cached["funding_time"].max()
                print(f"📅 Cache: {len(df_cached)} records up to {last_date.date()}")
        except Exception as e:
            print(f"⚠️ Funding cache error: {e}")
            df_cached = pd.DataFrame()
    
    # Determine date range to fetch
    if start_date is None:
        if last_date:
            fetch_start = last_date + pd.Timedelta(hours=8)  # Next 8h slot
        else:
            fetch_start = pd.Timestamp("2020-01-01", tz="UTC")
    else:
        fetch_start = to_utc(start_date)
    
    if end_date is None:
        end_date = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=8)  # Avoid partial data
    else:
        end_date = to_utc(end_date)
    
    if fetch_start >= end_date:
        print(f"✅ Funding data current")
        return df_cached
    
    print(f"🔍 Fetching: {fetch_start.date()} to {end_date.date()}")
    
    # Binance API returns up to 1000 records per call
    # Each record is 8h, so 1000 records ≈ 333 days
    all_new_data = []
    
    try:
        url = "https://fapi.binance.com/fapi/v1/fundingRate"
        params = {
            "symbol": symbol,
            "limit": 1000
        }
        
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            if data:
                df_new = pd.DataFrame(data)
                df_new['funding_time'] = pd.to_datetime(df_new['fundingTime'], unit='ms', utc=True)
                df_new['funding_rate_8h'] = df_new['fundingRate'].astype(float)
                df_new = df_new[['funding_time', 'funding_rate_8h']]
                
                # Filter to requested date range
                df_new = df_new[(df_new['funding_time'] >= fetch_start) & 
                               (df_new['funding_time'] <= end_date)]
                
                if not df_new.empty:
                    all_new_data.append(df_new)
                    print(f"📥 Fetched {len(df_new)} funding records")
                else:
                    print(f"⚠️ No new funding data in range")
            else:
                print(f"⚠️ Empty response from Binance")
        else:
            print(f"⚠️ Binance API error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Funding fetch error: {e}")
    
    if not all_new_data:
        print(f"⚠️ No new funding data fetched")
        return df_cached
    
    # Combine new data with cache
    df_new_combined = pd.concat(all_new_data, ignore_index=True)
    
    if not df_cached.empty:
        # Remove overlapping dates from cache
        df_cached = df_cached[df_cached['funding_time'] < df_new_combined['funding_time'].min()]
        df_combined = pd.concat([df_cached, df_new_combined], ignore_index=True)
    else:
        df_combined = df_new_combined
    
    # Sort and deduplicate
    df_combined = df_combined.sort_values('funding_time').drop_duplicates('funding_time')
    
    # Save to cache
    df_combined.to_csv(cache_file, index=False)
    print(f"💾 Funding data saved: {len(df_combined)} records")
    print(f"   Date range: {df_combined['funding_time'].min().date()} to {df_combined['funding_time'].max().date()}")
    
    return df_combined

def process_funding_for_ml(df_funding, df_whales):
    """
    Convert 8h funding rates to daily and align with whale data dates
    """
    if df_funding.empty:
        print("⚠️ No funding data available, using zeros")
        return pd.DataFrame(columns=['block_date', 'eth_funding_rate_8h'])
    
    # Convert to daily (mean of 3 funding periods per day)
    df_funding['date'] = df_funding['funding_time'].dt.date
    daily_funding = (
        df_funding
        .groupby('date')['funding_rate_8h']
        .mean()
        .reset_index()
        .rename(columns={'funding_rate_8h': 'eth_funding_rate_8h'})
    )
    
    # Get whale data date range
    whale_dates = pd.to_datetime(df_whales['block_date']).dt.date
    min_date = whale_dates.min()
    max_date = whale_dates.max()
    
    # Create full date range
    all_dates = pd.date_range(start=min_date, end=max_date, freq='D')
    all_dates_df = pd.DataFrame({'date': all_dates.date})
    
    # Merge funding data
    df_merged = pd.merge(
        all_dates_df,
        daily_funding,
        on='date',
        how='left'
    )
    
    # Fill missing dates (forward fill, then backward fill, then 0)
    df_merged['eth_funding_rate_8h'] = (
        df_merged['eth_funding_rate_8h']
        .ffill()
        .bfill()
        .fillna(0)
    )
    
    # Add block_date for merging
    df_merged['block_date'] = pd.to_datetime(df_merged['date'])
    
    return df_merged[['block_date', 'eth_funding_rate_8h']]

def load_all_data_incremental(force_fetch=False):
    """
    Load all data including funding rates
    """
    print("📊 Loading data...")
    
    # Load Dune and CoinGecko data first
    df_whales = fetch_dune_incremental(
        QUERIES["whales"][0], 
        QUERIES["whales"][1],
        query_name="whale_data",
        force_fetch=force_fetch
    )
    df_whales.to_csv(QUERIES["whales"][2], index=False)
    
    time.sleep(2)
    
    df_market = fetch_dune_incremental(
        QUERIES["market_intent"][0], 
        QUERIES["market_intent"][1],
        query_name="market_intent",
        force_fetch=force_fetch
    )
    df_market.to_csv(QUERIES["market_intent"][2], index=False)
    
    # Get max date for price fetching
    max_date = max(
        df_whales["block_date"].max() if not df_whales.empty else DUNE_START,
        df_market["block_date"].max() if not df_market.empty else DUNE_START
    )
    
    # Fetch prices
    df_btc = get_price_incremental("btc", "bitcoin", COINGECKO_BTC_START, max_date, COINGECKO_API_KEY, force_fetch)
    df_eth = get_price_incremental("eth", "ethereum", COINGECKO_ETH_START, max_date, COINGECKO_API_KEY, force_fetch)
    
    # ========== FETCH FUNDING DATA ==========
    # Get date range from whale data for funding fetch
    if not df_whales.empty:
        funding_start = df_whales['block_date'].min() - pd.Timedelta(days=7)  # Buffer
        funding_end = df_whales['block_date'].max()
        
        df_funding = fetch_binance_funding_incremental(
            symbol="ETHUSDT",
            start_date=funding_start,
            end_date=funding_end,
            force_fetch=force_fetch
        )
        
        # Process funding for ML
        df_funding_processed = process_funding_for_ml(df_funding, df_whales)
        
        # Save funding data
        funding_file = "data/funding_rates_ml_ready.csv"
        df_funding_processed.to_csv(funding_file, index=False)
        print(f"💾 Funding data saved: {funding_file}")
        
        # Print funding statistics
        if not df_funding_processed.empty:
            funding_stats = df_funding_processed['eth_funding_rate_8h']
            print(f"📊 Funding statistics:")
            print(f"   Mean: {funding_stats.mean():.6f}")
            print(f"   Min: {funding_stats.min():.6f}")
            print(f"   Max: {funding_stats.max():.6f}")
            print(f"   > {CROWDED_LONG}: {(funding_stats > CROWDED_LONG).sum()} days")
            print(f"   < {CROWDED_SHORT}: {(funding_stats < CROWDED_SHORT).sum()} days")
    
    print(f"\n📈 Summary:")
    print(f"   Whale: {len(df_whales)} rows")
    print(f"   Market: {len(df_market)} rows")
    print(f"   BTC: {len(df_btc)} rows")
    print(f"   ETH: {len(df_eth)} rows")
    print(f"   Funding: {len(df_funding_processed) if 'df_funding_processed' in locals() else 0} rows")
    
    return df_whales, df_market, df_btc, df_eth, df_funding_processed

def load_cached_data():
    """
    Load all cached data including funding rates
    """
    print("📂 Loading cached data...")
    
    files = {
        'whale': 'data/whale_ml_ready.csv',
        'market': 'data/market_intent_ml_ready.csv',
        'btc': 'data/price_cache/btc.csv',
        'eth': 'data/price_cache/eth.csv',
        'funding': 'data/funding_rates_ml_ready.csv'
    }
    
    loaded = {}
    
    for name, path in files.items():
        if os.path.exists(path):
            try:
                if name in ['btc', 'eth']:
                    df = pd.read_csv(path, parse_dates=["date"])
                    df["date"] = df["date"].apply(to_utc)
                elif name == 'funding':
                    df = pd.read_csv(path, parse_dates=["block_date"])
                    df["block_date"] = df["block_date"].apply(to_utc)
                else:
                    df = pd.read_csv(path, parse_dates=["block_date"])
                    df["block_date"] = df["block_date"].apply(to_utc)
                
                loaded[name] = df
                print(f"✅ {name}: {len(df)} rows")
            except Exception as e:
                print(f"❌ {name}: {e}")
                loaded[name] = pd.DataFrame()
        else:
            print(f"⚠️  {name} not found")
            loaded[name] = pd.DataFrame()
    
    return (loaded.get('whale', pd.DataFrame()),
            loaded.get('market', pd.DataFrame()),
            loaded.get('btc', pd.DataFrame()),
            loaded.get('eth', pd.DataFrame()),
            loaded.get('funding', pd.DataFrame()))

# Main Loading Execution

if __name__ == "__main__":
    print("\n" + "="*70)
    print("📊 DATA LOADER - FIXED VERSION")
    print("="*70)
    
    print("\n📋 OPTIONS:")
    print("1️⃣  Fetch fresh (incremental)")
    print("2️⃣  Force re-fetch all")
    print("3️⃣  Load cached only")
    print("="*70)

    choice = input("\nSelect (1-3): ").strip()

    if choice == '1':
        print("\n🚀 Fetching fresh data...")
        load_all_data_incremental(force_fetch=False)
        
    elif choice == '2':
        confirm = input("Delete cache and fetch all? (y/n): ").lower()
        if confirm == 'y':
            load_all_data_incremental(force_fetch=True)
        
    elif choice == '3':
        load_cached_data()
        
    else:
        print("❌ Invalid option")
    
    print(f"\n{'='*70}")
    print("✅ Complete!")
    print(f"{'='*70}")
