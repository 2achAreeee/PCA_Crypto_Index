import pandas as pd

from src import config
import os

if __name__ == '__main__':
    # 1. Load and Prepare Data
    gecko_daily_folder_path = config.GECKO_DAILY_DATA_DIR
    gecko_daily_all_files = [f for f in os.listdir(gecko_daily_folder_path) if f.endswith('.csv')]

    gecko_daily_dataset_series = {}
    for file in gecko_daily_all_files:
        try:
            data = pd.read_csv(os.path.join(gecko_daily_folder_path, file))

            # Remove the last row which is often incomplete
            if not data.empty:
                data = data.iloc[:-1]

            data['Date'] = pd.to_datetime(data['date'])
            data = data.sort_values('Date').drop_duplicates(subset='Date')

            # Use 'log_Return' column
            price_series = pd.to_numeric(data['log_Return'])
            price_series.index = data['Date']

            file_name = file.replace('.csv', '')
            gecko_daily_dataset_series[file_name] = price_series
        except Exception as e:
            print(f"Could not process file {file}: {e}")

    # Merge all series into a single DataFrame
    gecko_daily_full_data = pd.concat(gecko_daily_dataset_series, axis=1)

    print(
        f"Loaded and merged data for {gecko_daily_full_data.shape[1]} assets from {gecko_daily_full_data.index.min().date()} to {gecko_daily_full_data.index.max().date()}.")

    # Ensure the processed data directory exists
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save the newly created DataFrame to the cache file
    gecko_daily_full_data.to_csv(config.DAILY_RETURN_FILE_PATH)

    # 1. Load and Prepare Data
    gecko_mcap_folder_path = config.GECKO_MCAP_DATA_DIR
    gecko_mcap_all_files = [f for f in os.listdir(gecko_mcap_folder_path) if f.endswith('.csv')]

    gecko_mcap_dataset_series = {}
    for file in gecko_mcap_all_files:
        try:
            data = pd.read_csv(os.path.join(gecko_mcap_folder_path, file))

            # Remove the first row to map with log return data
            if not data.empty:
                data = data.iloc[1:]

            data['Date'] = pd.to_datetime(data['date'])
            data = data.sort_values('Date').drop_duplicates(subset='Date')

            # Use 'market_cap' column
            price_series = pd.to_numeric(data['market_cap'])
            price_series.index = data['Date']

            file_name = file.replace('.csv', '')
            gecko_mcap_dataset_series[file_name] = price_series
        except Exception as e:
            print(f"Could not process file {file}: {e}")

    # Merge all series into a single DataFrame
    gecko_mcap_full_data = pd.concat(gecko_mcap_dataset_series, axis=1)

    print(
        f"Loaded and merged data for {gecko_mcap_full_data.shape[1]} assets from {gecko_mcap_full_data.index.min().date()} to {gecko_mcap_full_data.index.max().date()}.")

    # Ensure the processed data directory exists
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save the newly created DataFrame to the cache file
    gecko_mcap_full_data.to_csv(config.MARKET_CAP_FILE_PATH)

