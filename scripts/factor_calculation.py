import pandas as pd
import numpy as np
import pandas_datareader.data as web
from src import config
import os


def prepare_factor_data():
    """
    Loads, merges, and prepares daily returns (from Yahoo) and market cap data
    (from CoinGecko) for factor calculation.

    Returns:
        tuple: A tuple containing two DataFrames:
               - daily_returns (pd.DataFrame): Dates as index, tickers as columns.
               - market_caps (pd.DataFrame): Dates as index, tickers as columns.
    """
    print("Preparing data for factor calculation...")

    # 1. Load Yahoo Finance Daily Data for Returns
    # We assume this data is already downloaded and cached if necessary.
    # Here, we will use the same caching logic for a combined yahoo data file.
    daily_return_path = config.DAILY_RETURN_FILE_PATH

    if os.path.exists(daily_return_path):
        daily_data = pd.read_csv(daily_return_path, index_col='Date', parse_dates=True)
    else:
        # This part should be adapted from your data ingestion scripts.
        # For now, we'll assume a placeholder for creating this cache.
        print(f"Yahoo data cache not found at {daily_return_path}. Please create it first.")

    daily_returns = daily_data.iloc[2:]  # Using placeholder returns

    # 2. Load CoinGecko Data for Market Caps
    market_caps = pd.read_csv(config.MARKET_CAP_FILE_PATH, index_col='Date', parse_dates=True)

    # 3. Load PCA-based Index data
    pca_index = pd.read_csv(config.INDEX_FILE_PATH, index_col='Date', parse_dates=True)

    # 4. Align DataFrames
    # Find the common dates and common tickers between the two datasets
    common_dates = daily_returns.index.intersection(market_caps.index)
    common_tickers = daily_returns.columns.intersection(market_caps.columns)

    # Filter both dataframes to only the common dates and tickers
    daily_returns = daily_returns.loc[common_dates, common_tickers]
    market_caps = market_caps.loc[common_dates, common_tickers]

    # # Convert all datetime index values to string format '%y-%m-%d'
    # daily_returns.index = pd.to_datetime(daily_returns.index, format='%y-%m-%d')
    # market_caps.index = pd.to_datetime(market_caps.index, format='%y-%m-%d')
    # pca_index.index = pd.to_datetime(pca_index.index, format='%y-%m-%d')

    print(
        f"Data prepared with {len(common_tickers)} assets from {common_dates.min().date()} to {common_dates.max().date()}.")

    return daily_returns, market_caps, pca_index


# Add these functions to scripts/factor_calculation.py

def get_risk_free_rate(start_date, end_date):
    """
    Fetches the 3-Month Treasury Bill Rate from FRED as the risk-free rate.
    """
    print("Fetching risk-free rate from FRED...")
    try:
        # The series for 3-Month Treasury Bill Secondary Market Rate
        rf = web.DataReader('DTB3', 'fred', start_date, end_date)
        # Forward-fill missing values and convert to daily rate
        rf = rf.ffill() / 100 / 365
        return rf
    except Exception as e:
        print(f"Could not fetch risk-free rate: {e}. Returning zero series.")
        return pd.Series(0, index=pd.date_range(start=start_date, end=end_date), name='DTB3')


def calculate_factors(pca_index_data, daily_returns, market_caps):
    """
    Calculates the MKT, SMB, and WML (Momentum) factors.
    """
    print("Calculating all three factors...")

    # --- 1. Calculate Risk-Free Rate and MKT Factor ---
    rf_rate = get_risk_free_rate(daily_returns.index.min(), daily_returns.index.max())

    # Align risk-free rate with our data's dates
    rf = rf_rate['DTB3'].reindex(daily_returns.index, method='ffill')

    # Market return is the equal-weighted average of all crypto returns
    market_return = daily_returns.mean(axis=1)
    mkt = market_return - rf

    # --- 2. Calculate SMB (Small Minus Big) Factor ---
    # This loop calculates SMB for each day
    smb_list = []
    portfolio_size = 100  # Define the size of our portfolios

    for date in daily_returns.index:
        # Get market caps and returns for the current day, dropping NaNs
        mcaps_today = market_caps.loc[date].dropna()
        mcaps_today = mcaps_today[mcaps_today > 0]

        returns_today = daily_returns.loc[date].dropna()

        sorted_market_caps = mcaps_today.sort_values(ascending=False)

        if len(mcaps_today) < 2 * portfolio_size:
            smb_list.append(0)
            continue

        # --- Identify Portfolios ---
        # Get the tickers for the top 100 (Big) and bottom 100 (Small)
        big_portfolio_tickers = sorted_market_caps.head(portfolio_size).index.intersection(returns_today.index)
        small_portfolio_tickers = sorted_market_caps.tail(portfolio_size).index.intersection(returns_today.index)

        # --- Get Returns for Each Portfolio ---
        # Select the daily returns for the assets in each portfolio
        big_portfolio_returns = returns_today.loc[big_portfolio_tickers]
        small_portfolio_returns = returns_today.loc[small_portfolio_tickers]

        # --- Calculate SMB Value for the Day ---
        # Calculate the equal-weighted average return for each portfolio
        big_return = big_portfolio_returns.mean()
        small_return = small_portfolio_returns.mean()

        # SMB is the return of the small portfolio minus the return of the big one
        smb_value = small_return - big_return
        smb_list.append(smb_value)

    smb = pd.Series(smb_list, index=daily_returns.index)

    # --- 3. Calculate WML (Winners Minus Losers) Momentum Factor ---
    # This is computationally intensive. We rebalance monthly for performance.
    wml_list = []
    # Get month-end dates for rebalancing
    rebalance_dates = daily_returns.resample('M').last().index

    # Using a 3-month lookback, holding for 1 month
    for i in range(3, len(rebalance_dates)):
        # Define portfolio formation period
        formation_date = rebalance_dates[i - 1]
        start_lookback = rebalance_dates[i - 3]

        # Calculate past returns (momentum)
        past_returns = (daily_returns.loc[start_lookback:formation_date] + 1).prod() - 1
        # Clean any infinities that might arise from the calculation
        past_returns = past_returns.replace([np.inf, -np.inf], np.nan).dropna()

        if len(past_returns) < 10:  # Need enough assets to form portfolios
            continue

        # Define quantile-based portfolios (e.g., top and bottom 30%)
        q30 = past_returns.quantile(0.4)
        q70 = past_returns.quantile(0.6)
        losers = past_returns[past_returns <= q30].index
        winners = past_returns[past_returns >= q70].index

        # Calculate returns for the NEXT month
        holding_period_start = formation_date
        holding_period_end = rebalance_dates[i]

        # This is your new line that drops columns with any NaN in the holding period
        returns_next_month = daily_returns.loc[holding_period_start:holding_period_end].dropna(axis=1, how='any')

        # ========================================================================
        # --- FIX: Find the intersection of desired tickers and available tickers ---
        # ========================================================================
        safe_low_tickers = losers.intersection(returns_next_month.columns)
        safe_high_tickers = winners.intersection(returns_next_month.columns)

        # Now, use these "safe" lists for the calculation
        low_ret_returns = returns_next_month[safe_low_tickers].mean(axis=1)
        high_ret_returns = returns_next_month[safe_high_tickers].mean(axis=1)

        monthly_hml = low_ret_returns - high_ret_returns
        wml_list.append(monthly_hml)

    wml = pd.concat(wml_list)
    wml = wml[~wml.index.duplicated()]

    # Combine all factors into a single DataFrame
    factors = pd.DataFrame({
        'MKT': mkt,
        'SMB': smb,
        'WML': wml
    }).dropna()#.fillna(0)  # Fill any NaNs that may have occurred

    print("Factor calculation complete.")
    return factors


if __name__ == '__main__':
    # 1. Prepare the underlying data
    daily_returns, market_caps, pca_index = prepare_factor_data()

    # 2. Calculate the three factors
    factors_df = calculate_factors(pca_index, daily_returns, market_caps)

    # 3. Save the factors to a CSV file
    factors_df.to_csv(config.THREE_FACTOR_FILE_PATH)
    print(f"\nThree-factor model data successfully saved to: {config.THREE_FACTOR_FILE_PATH}")

    # 4. Display the last 5 rows of the result
    print("\n--- Factor Model Data (Last 5 Days) ---")
    print(factors_df.tail())