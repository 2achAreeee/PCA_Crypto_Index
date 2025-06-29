import os
from pathlib import Path

# --- Base Directory ---
# This defines the absolute path to your project's root folder (PCA_Index/)
# It assumes this config.py file is located at: PCA_Index/src/config.py
BASE_DIR = Path(__file__).resolve().parent.parent

# --- Data Directories ---
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

# Specific raw data sources
GECKO_MCAP_DATA_DIR = RAW_DATA_DIR / 'coingecko_market_cap'
GECKO_DAILY_DATA_DIR = RAW_DATA_DIR / 'coingecko_daily'
YAHOO_DATA_DIR = RAW_DATA_DIR / 'yahoo_finance_daily' # This is the variable you requested

# --- Asset List Directories ---
ASSET_LISTS_DIR = BASE_DIR / 'asset_lists' # This is the variable you requested

# Specific ticker files
GECKO_TICKERS_FILE = ASSET_LISTS_DIR / 'cryptocurrency_list_1200.csv'
YAHOO_TICKERS_FILE = ASSET_LISTS_DIR / 'crypto_tickers.json' # This is the variable you requested

# --- Processed Data Paths ---
MARKET_CAP_FILE_PATH = PROCESSED_DATA_DIR / 'market_cap_full.csv'
DAILY_RETURN_FILE_PATH = PROCESSED_DATA_DIR / 'daily_return_full.csv'
INDEX_FILE_PATH = PROCESSED_DATA_DIR / 'dynamic_pca_index.csv'
NC_RESULTS_CACHE_PATH = PROCESSED_DATA_DIR / 'nc_results_cache.json'
THREE_FACTOR_FILE_PATH = PROCESSED_DATA_DIR / 'three_factors.csv'

# --- Results Directory ---
RESULTS_DIR = BASE_DIR / 'results'
FIGURES_DIR = RESULTS_DIR / 'figures'

# --- Model & Run Parameters ---
INDEX_START_VALUE = 1000

# Dynamic Index Parameters
TN_LOOKBACK = 90  # Look-back period for choosing components
TN_GAP = 30       # Step size for each iteration
TW_CALCULATION = 60 # Calculation window for the index segment
VARIANCE_THRESHOLD = 0.99