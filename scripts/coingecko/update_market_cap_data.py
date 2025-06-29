import os
import pandas as pd
import requests
import time
from datetime import datetime

from src import config

# --- Configuration ---
DATA_DIR = config.GECKO_MCAP_DATA_DIR
TOP_N_COINS = 1200
# For a weekly script, fetching the last 15 days is safe to cover any gaps.
UPDATE_LOOKBACK_DAYS = 100
# It's polite and safe to wait between API calls to avoid being rate-limited.
API_SLEEP_INTERVAL_SECONDS = 15
# A longer sleep interval for the retry pass
RETRY_SLEEP_SECONDS = 60
# File to store the list of top coins, as you did previously.
COIN_LIST_FILE = config.GECKO_TICKERS_FILE


# --- CoinGecko API Functions (from your original code, made more robust) ---

def get_top_coins(limit=TOP_N_COINS):
    """
    Fetches the list of top N cryptocurrencies by market capitalization from CoinGecko.
    """
    print(f"Fetching the current top {limit} coins from CoinGecko...")
    coins = []
    # CoinGecko's max per_page is 250
    per_page = 250
    # Calculate the number of pages needed
    num_pages = (limit + per_page - 1) // per_page

    for page in range(1, num_pages + 1):
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            'vs_currency': 'usd',
            'order': 'market_cap_desc',
            'per_page': per_page,
            'page': page,
            'sparkline': False
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
            coins.extend(response.json())
            print(f"Successfully fetched page {page}/{num_pages}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to get page {page}: {e}")
            return []  # Return empty list on failure
        time.sleep(5)  # Brief pause between paged requests

    # Return only the coin IDs, up to the specified limit
    return [coin['id'] for coin in coins[:limit]]


def get_market_cap_history(coin_id, days='365'):
    """
    Fetches historical market cap for a single coin from CoinGecko.
    The 'days' parameter can be an integer or 'max'.
    """
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {
        'vs_currency': 'usd',
        'days': days,
        'interval': 'daily'
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        market_caps = data.get('market_caps', [])

        if market_caps:
            # The API returns daily data, but the timestamp can be slightly off.
            # Normalizing to the start of the day (midnight UTC) ensures consistency.
            df = pd.DataFrame({
                'date': pd.to_datetime([item[0] for item in market_caps], unit='ms').normalize(),
                'market_cap': [item[1] for item in market_caps]
            })
            df['coin_id'] = coin_id
            # The API can sometimes return the current day's incomplete data. We remove the last row if it matches today's date.
            if not df.empty and df['date'].iloc[-1].date() == datetime.utcnow().date():
                df = df.iloc[:-1]
            return df

    except requests.exceptions.RequestException as e:
        print(f"❌ Could not fetch history for {coin_id}: {e}")
    return None


# --- Helper Function for Processing a Single Coin ---

def process_single_coin(coin_id):
    """
    Processes one coin: updates if its file exists, creates it if not.
    Returns True on success, False on failure.
    """
    file_path = os.path.join(DATA_DIR, f"{coin_id}.csv")
    try:
        if os.path.exists(file_path):
            # --- UPDATE LOGIC ---
            update_df = get_market_cap_history(coin_id, days=UPDATE_LOOKBACK_DAYS)

            if update_df is None:  # Explicitly check for API/network failure
                return False  # The error is already printed by the function above

            if not update_df.empty:
                existing_df = pd.read_csv(file_path, parse_dates=['date'])
                combined_df = pd.concat([existing_df, update_df], ignore_index=True)
                combined_df.sort_values(by='date', inplace=True)
                combined_df.drop_duplicates(subset='date', keep='last', inplace=True)
                combined_df.to_csv(file_path, index=False, date_format='%Y-%m-%d %H:%M:%S')
                new_rows = len(combined_df) - len(existing_df)
                print(f"  ✅ Updated. Added {new_rows} new records. Total: {len(combined_df)}")
            else:
                print(f"  - No new daily records to add. File unchanged.")
        else:
            # --- CREATE LOGIC ---
            print(f"  📂 File not found. Performing initial 365-day download...")
            df = get_market_cap_history(coin_id, days='365')

            if df is None:  # Explicitly check for API/network failure
                return False

            if not df.empty:
                df.to_csv(file_path, index=False, date_format='%Y-%m-%d %H:%M:%S')
                print(f"  ✅ Saved initial history with {len(df)} records.")
            else:
                print(f"  - No historical data found for this new coin.")

    except Exception as e:
        # Catches other errors like file read/write issues, pandas errors, etc.
        print(f"  ❌ A local processing error occurred for {coin_id}: {e}")
        return False

    return True  # Return True only if all steps complete without errors


# --- Main Update Logic with Retry ---

def run_weekly_update():
    """
    Main function to orchestrate the weekly update, now with a retry loop.
    """
    print("🚀 Starting weekly data update from a fixed list...")

    # Setup: Ensure data directory exists and read the master coin list
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    try:
        master_list_df = pd.read_csv(COIN_LIST_FILE)
        coins_to_process = master_list_df['coin_id'].dropna().unique().tolist()
        if not coins_to_process:
            print(f"🚨 Master list '{COIN_LIST_FILE}' is empty. Aborting.")
            return
        print(f"📄 Found {len(coins_to_process)} unique coins to process in '{COIN_LIST_FILE}'.")
    except (FileNotFoundError, KeyError) as e:
        print(f"🚨 CRITICAL ERROR reading master list: {e}. Aborting.")
        return

    # --- Pass 1: Main Processing Loop ---
    print("\n--- Pass 1: Processing all coins from the list ---")
    failed_updates = []
    for i, coin_id in enumerate(coins_to_process, 1):
        print(f"[{i}/{len(coins_to_process)}] Processing: {coin_id}")
        success = process_single_coin(coin_id)
        if not success:
            failed_updates.append(coin_id)
        time.sleep(API_SLEEP_INTERVAL_SECONDS)

    # --- Pass 2: Retry Loop for Failed Updates ---
    if failed_updates:
        print(f"\n--- Pass 2: Retrying {len(failed_updates)} failed update(s) ---")
        final_failures = []
        for i, coin_id in enumerate(failed_updates, 1):
            print(f"[{i}/{len(failed_updates)}] Retrying: {coin_id}")
            success = process_single_coin(coin_id)
            if not success:
                final_failures.append(coin_id)
            time.sleep(RETRY_SLEEP_SECONDS)  # Use a longer wait time for retries

        # --- Final Report on Failures ---
        if final_failures:
            print("\n🚨 The following coins failed to update after two attempts:")
            for coin_id in final_failures:
                print(f"  - {coin_id}")
            print("Please check their status or your network connection manually.")
        else:
            print("\n✅ All failed updates were successful on the second attempt!")

    print("\n\n🎉 Data update process has completed! 🎉")


# --- Script Execution ---
if __name__ == "__main__":
    run_weekly_update()
    # top_crypto_list = get_top_coins()
    # print(top_crypto_list)
    #
    # test_df = get_market_cap_history("bitcoin")
    # print(test_df)