import pandas as pd
import os

# Paths
coin_list_path = 'asset_lists/cryptocurrency_list_1200.csv'
data_folder_path = 'data/raw/coingecko_market_cap'

# # Load list of expected coin_ids
# coin_df = pd.read_csv(coin_list_path)
# coin_ids = set(coin_df['coin_id'].astype(str))
#
# # Get list of filenames (remove .csv extension)
# files = os.listdir(data_folder_path)
# existing_ids = set(os.path.splitext(f)[0] for f in files if f.endswith('.csv'))
#
# # Find missing coin_ids
# missing_ids = coin_ids - existing_ids
#
# print("Missing coin_ids (not found in folder):")
# for coin in sorted(missing_ids):
#     print(coin)
#
# print(len(missing_ids))
#
# # Optionally save to a file
# # pd.Series(sorted(missing_ids)).to_csv("missing_coin_ids.csv", index=False)

# Load list of valid coin_ids
coin_df = pd.read_csv(coin_list_path)
valid_coin_ids = set(coin_df['coin_id'].astype(str))

# Get all .csv files in the folder (remove extension for comparison)
all_files = os.listdir(data_folder_path)
all_coin_files = [f for f in all_files if f.endswith('.csv')]
file_ids = set(os.path.splitext(f)[0] for f in all_coin_files)

# Find extra files (files not in the valid coin list)
extra_ids = file_ids - valid_coin_ids

# Delete the extra files
for extra_id in extra_ids:
    file_path = os.path.join(data_folder_path, f"{extra_id}.csv")
    try:
        os.remove(file_path)
        print(f"Removed: {file_path}")
    except Exception as e:
        print(f"Error removing {file_path}: {e}")