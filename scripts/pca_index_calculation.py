import os
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src import config

# --- Helper Functions (Recreating R's internal logic in Python) ---

def nc_cal_py(dataframe: pd.DataFrame, threshold: float = config.VARIANCE_THRESHOLD) -> dict:
    """
    Determines the number of assets (Nc) to include by finding the smallest
    subset of top assets whose index correlates highly with a "full" index.
    This is a direct translation of the user's R code logic.

    Args:
        dataframe (pd.DataFrame): DataFrame of market caps for a time window.
                                  (Assumes columns are pre-sorted by market cap).
        threshold (float): The correlation threshold to meet (e.g., 0.99).

    Returns:
        dict: A dictionary containing Nc and the list of chosen crypto names.
    """
    num_assets = dataframe.shape[1]

    # This logic requires at least 2 assets to compare.
    if num_assets < 2:
        return {
            "Nc": num_assets,
            "Choosen Crypto": dataframe.columns.tolist()
        }

    # --- Step 1 & 2: Loop through subsets and calculate PC1 index for each ---
    index_series_list = []
    scaler = StandardScaler()

    # The loop starts at 2, creating indices for the top 2, top 3, ..., top N assets
    for i in range(2, num_assets + 1):
        subset_data = dataframe.iloc[:, :i]

        # Scale the data and perform PCA
        scaled_subset = scaler.fit_transform(subset_data)
        pca = PCA(n_components=1)
        pca.fit(scaled_subset)

        # Get PC1 loadings (weights). Ensure consistent direction.
        pc1_loadings = pca.components_[0]
        if np.sum(pc1_loadings) < 0:
            pc1_loadings = -pc1_loadings

        # Calculate the index for this subset (Matrix multiplication)
        # This is equivalent to R's `%*%` operator
        subset_index = subset_data.dot(pc1_loadings)
        index_series_list.append(subset_index)

    # --- Step 3: Calculate Correlations against the "full" index ---
    # The benchmark index is the last one calculated, which used all available assets.
    benchmark_index = index_series_list[-1]

    correlations = []
    for subset_idx in index_series_list:
        # Note: The correlation of the last index with itself will be 1.0
        corr = subset_idx.corr(benchmark_index)
        correlations.append(corr)

    # --- Step 4: Find the first index that meets the correlation threshold ---
    correlations = np.array(correlations)

    # `np.where` finds the indices where the condition is true. We take the first one.
    indices_reaching_threshold = np.where(correlations >= threshold)[0]

    if len(indices_reaching_threshold) > 0:
        # The first element in `correlations` corresponds to an index of 2 assets.
        # So, we add 2 to the found list index to get the true number of assets.
        first_reach_list_index = indices_reaching_threshold[0]
        nc = first_reach_list_index + 2
    else:
        # If no subset meets the threshold, use all assets as a fallback.
        nc = num_assets

    chosen_crypto_list = dataframe.columns[:nc].tolist()

    time_window = [str(dataframe.index[0]), str(dataframe.index[-1])]
    # The main script only needs 'Nc' and 'Choosen Crypto'.
    # We return these in the expected dictionary format.
    return {
        "Nc": int(nc),
        "Choosen Crypto": chosen_crypto_list,
        "Time Window": time_window
    }


def dynamic_index_base_py(dataframe: pd.DataFrame, m_start: float, tw: int = 30) -> pd.Series:
    """
    Calculates the index value for a single period using fixed PCA weights
    from the first half of the window, and chain-links the result.
    This version incorporates the user's more efficient vectorization.

    Args:
        dataframe (pd.DataFrame): The full data window for calculation (e.g., 60 days).
        m_start (float): The last index value from the previous period, used for linking.
        tw (int): The sub-window size for PCA calculation (defaults to 30).

    Returns:
        pd.Series: A pandas Series containing the calculated index values for the
                   second half of the window (e.g., the next 30 days).
    """
    if len(dataframe) < tw + 1:
        return pd.Series(dtype=float)

    # --- Step 1: Perform PCA on the first `tw` days to get fixed weights ---
    pca_window = dataframe.iloc[0:tw].dropna(axis=1, how='any')
    if pca_window.shape[1] < 1:
        return pd.Series(dtype=float)

    scaler = StandardScaler()
    pca = PCA(n_components=1)
    scaled_data = scaler.fit_transform(pca_window)
    pca.fit(scaled_data)

    pc1_loadings = pca.components_[0]
    if np.sum(pc1_loadings) < 0:
        pc1_loadings = -pc1_loadings

    loadings_s = pd.Series(pc1_loadings, index=pca_window.columns)
    aligned_loadings = loadings_s.reindex(dataframe.columns, fill_value=0.0).values

    # --- Step 2: Calculate `a_base` for normalization ---
    base_value_row = dataframe.iloc[tw]
    a_base = base_value_row.dot(aligned_loadings)

    if a_base == 0:
        return pd.Series(dtype=float)

    # --- Step 3: Calculate index values for the second half of the window ---
    index_calc_window = dataframe.iloc[tw:]

    # Calculate the raw values for the entire calculation window
    raw_values = index_calc_window.dot(aligned_loadings)

    # Chain-link all values in a single vectorized operation
    final_values = (raw_values * m_start) / a_base

    # Create the final Series. The name and index are already aligned.
    final_values.name = 'Value'

    return final_values


# --- Main Calculation Function ---

def dynamic_index_cal_py(dataframe: pd.DataFrame, tn: int = 90, tn_gap: int = 30, tw: int = 30,
                         threshold: float = 0.99) -> list:
    """
    The main function to calculate the dynamic index.

    Args:
        dataframe (pd.DataFrame): The full DataFrame of market caps.
        tn (int): The length of the look-back period for choosing components (e.g., 90 days).
        tn_gap (int): The step size for each iteration (e.g., 30 days).
        tw (int): The length of the index calculation window (e.g., 30 days - although the R code uses 60).
                  Note: The R code slices `31:90`, which is 60 days. We use `Tw = 60`.
        threshold (float): Variance threshold for component selection.

    Returns:
        list: A list containing [list_of_nc_results, list_of_index_series].
    """
    Tw = 60  # Correcting based on R code's slice `(31+i*30):(90+i*30)`

    # --- Nc Choosing process ---
    nc_results = []
    n_iter = (len(dataframe) - tn) // tn_gap

    print("Starting Nc (Number of Components) selection process...")
    for i in range(n_iter + 1):
        start_row = i * tn_gap
        end_row = tn + i * tn_gap

        # Slice the 90-day window
        data_90 = dataframe.iloc[start_row:end_row].copy()

        # Data filtering pipeline, similar to R's `select(where(...))`
        # Drop columns that contain any NA or 0 values
        data_90 = data_90.loc[:, (data_90.notna().all() & (data_90 != 0).all())]
        # Drop columns with zero standard deviation
        valid_std = data_90.std() > 0
        data_90 = data_90.loc[:, valid_std]

        if data_90.empty:
            continue

        # Sort columns by the first row's market cap, descending
        sorted_cols = data_90.iloc[0].sort_values(ascending=False).index
        data_90 = data_90[sorted_cols]

        nc_results.append(nc_cal_py(data_90, threshold=threshold))
        print(
            f"  Iteration {i + 1}/{n_iter + 1}: Found {nc_results[-1]['Nc']} components from {len(nc_results[-1]['Choosen Crypto'])} cryptos.")

    # --- Dynamic Index Calculation ---
    dynamic_index_results = []
    m = 1000.0  # Start the index at 1000 for better scaling

    print("\nStarting Dynamic Index calculation...")
    for i, nc_info in enumerate(nc_results):
        # The calculation window starts 30 days into the 90-day selection window
        start_row_60 = i * tn_gap + (tn - Tw)
        end_row_60 = tn + i * tn_gap

        # Select the 60-day window
        data_60 = dataframe.iloc[start_row_60:end_row_60].copy()

        # Filter by selected cryptos from the Nc process
        chosen_cryptos = [c for c in nc_info["Choosen Crypto"] if c in data_60.columns]
        data_60 = data_60[chosen_cryptos]

        # Further clean this specific window
        data_60 = data_60.loc[:, (data_60.notna().all() & (data_60 != 0).all())]

        if len(data_60) < Tw or data_60.shape[1] < 1:
            print(f"  Skipping Iteration {i + 1}: Not enough valid data for index calculation.")
            continue

        index_segment = dynamic_index_base_py(data_60, m_start = m)

        if not index_segment.empty:
            dynamic_index_results.append(index_segment)
            # Update 'm' to the last value of the newly calculated segment
            m = index_segment.iloc[-1]
            print(f"  Iteration {i + 1}: Calculated index segment. New base value m = {m:.2f}")

    return [nc_results, dynamic_index_results]


# --- Main Execution Block ---
if __name__ == '__main__':
    # # 1. Load and Prepare Data
    # folder_path = config.GECKO_DATA_DIR
    # all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    #
    # dataset_series = {}
    # for file in all_files:
    #     try:
    #         data = pd.read_csv(os.path.join(folder_path, file))
    #
    #         # # Remove one the line which download by accident and is incomplete data
    #         # data['date'] = pd.to_datetime(data['date'])
    #         # data = data[data['date'].dt.time == pd.to_datetime('00:00:00').time()]
    #         # data.to_csv(os.path.join(folder_path, file))
    #
    #         # Remove the last row which is often incomplete
    #         if not data.empty:
    #             data = data.iloc[:-1]
    #
    #         data['Date'] = pd.to_datetime(data['date'])
    #         data = data.sort_values('Date').drop_duplicates(subset='Date')
    #
    #         # Use 'market_cap' column
    #         price_series = pd.to_numeric(data['market_cap'])
    #         price_series.index = data['Date']
    #
    #         file_name = file.replace('.csv', '')
    #         dataset_series[file_name] = price_series
    #     except Exception as e:
    #         print(f"Could not process file {file}: {e}")
    #
    # # Merge all series into a single DataFrame
    # full_data = pd.concat(dataset_series, axis=1)
    #
    # print(
    #     f"Loaded and merged data for {full_data.shape[1]} assets from {full_data.index.min().date()} to {full_data.index.max().date()}.")
    #
    # # Ensure the processed data directory exists
    # config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    #
    # # Save the newly created DataFrame to the cache file
    # full_data.to_csv(config.MARKET_CAP_FILE_PATH)

    # 2. Calculate the Dynamic Index
    market_cap_full_data = pd.read_csv(config.MARKET_CAP_FILE_PATH, index_col='Date', parse_dates=True)
    nc_res, index_res = dynamic_index_cal_py(market_cap_full_data)

    output_index = pd.concat(index_res)
    output_index.to_csv(config.INDEX_FILE_PATH)
    # Save the newly calculated results to the cache
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.NC_RESULTS_CACHE_PATH, 'w') as f:
        json.dump(nc_res, f, indent=4)
    print(f"Nc results saved to cache: {config.NC_RESULTS_CACHE_PATH}")

    # 3. Combine and Plot Results
    if index_res:
        # Combine all the index segments into one continuous series
        combined_dynamic_index = pd.concat(index_res)
        # Remove any duplicate index entries that might occur at the seams
        combined_dynamic_index = combined_dynamic_index[~combined_dynamic_index.index.duplicated(keep='first')]

        # For comparison, calculate the total market capitalization
        total_market_cap = market_cap_full_data.sum(axis=1)

        # Align the total market cap data with the index data
        aligned_market_cap = total_market_cap.reindex(combined_dynamic_index.index)

        # Plotting
        fig, ax1 = plt.subplots(figsize=(14, 7))

        # Plot Dynamic Index on the left y-axis
        ax1.plot(combined_dynamic_index.index, combined_dynamic_index, color='blue', label='Dynamic PCA Index')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Dynamic Index Value', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True)

        # Create a second y-axis for the total market capitalization
        ax2 = ax1.twinx()
        ax2.plot(aligned_market_cap.index, aligned_market_cap, color='red', alpha=0.6, linestyle='--',
                 label='Total Market Cap')
        ax2.set_ylabel('Total Market Capitalization', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        # Formatting and Legends
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        fig.autofmt_xdate()
        plt.title('Dynamic PCA Index vs. Total Market Capitalization')

        # Combine legends from both axes
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left')

        plt.show()
    else:
        print("Could not generate index results. The dataset might be too sparse or short.")