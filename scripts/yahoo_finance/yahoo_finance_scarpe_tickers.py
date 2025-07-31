import os
import json
import requests
from lxml import html

# --- Configuration ---
ASSET_LISTS_DIR = os.path.join('../..', 'asset_lists')
TICKERS_FILE = os.path.join(ASSET_LISTS_DIR, 'crypto_tickers.json')

BASE_URL = "https://finance.yahoo.com/markets/crypto/all"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}


def scrape_and_update_tickers(top_num=1000):
    """
    Scrapes top crypto tickers (up to top_num) from Yahoo Finance and updates JSON file.
    """
    scraped_tickers = []

    for start in range(0, top_num, 100):
        url = f"{BASE_URL}/?start={start}&count=100"
        print(f"Fetching: {url}")

        try:
            response = requests.get(url, headers=HEADERS)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            continue

        tree = html.fromstring(response.content)

        # Update this XPath if Yahoo changes layout
        tickers = tree.xpath('//table/tbody/tr/td[1]//a//text()')
        tickers = [ticker.strip() for ticker in tickers if ticker.strip()]
        tickers = [t for t in tickers if '-' in t and t.endswith('USD')]
        print(tickers)
        print(f" - Found {len(tickers)} tickers on this page.")

        scraped_tickers.extend(tickers)

    if not scraped_tickers:
        print("No tickers found.")
        return

    print(f"\nTotal scraped tickers: {len(scraped_tickers)}")

    # # Load existing tickers from file (if any)
    # existing_tickers = []
    # if os.path.exists(TICKERS_FILE):
    #     try:
    #         with open(TICKERS_FILE, 'r') as f:
    #             existing_tickers = json.load(f)
    #         print(f"Loaded {len(existing_tickers)} existing tickers.")
    #     except (json.JSONDecodeError, FileNotFoundError):
    #         print("Warning: Couldn't read existing tickers. Starting fresh.")
    #
    # # Merge and remove duplicates
    # combined_set = set(existing_tickers).union(scraped_tickers)
    # updated_tickers = sorted(combined_set)

    # Save back to file
    os.makedirs(ASSET_LISTS_DIR, exist_ok=True)
    with open(TICKERS_FILE, 'w') as f:
        json.dump(scraped_tickers, f, indent=4)

    print(f"Saved {len(scraped_tickers)} tickers in market cap order to {TICKERS_FILE}.")
    # with open(TICKERS_FILE, 'w') as f:
    #     json.dump(updated_tickers, f, indent=4)
    #
    # print(f"\nUpdated {TICKERS_FILE}. Added {len(updated_tickers) - len(existing_tickers)} new tickers.")
    # print(f"Total tickers saved: {len(updated_tickers)}")


# --- Main Execution ---
if __name__ == '__main__':
    scrape_and_update_tickers(top_num=1000)
