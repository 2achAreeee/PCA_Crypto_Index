# PCA-Based Cryptocurrency Index Construction and Analysis

## Introduction

This project implements and extends the methodology for constructing a dynamic, PCA-based cryptocurrency index, inspired by the paper "Principal Component Analysis-Based Construction and Evaluation of a Cryptocurrency Index." The primary goal is to build a representative index from the top 1,200 cryptocurrencies and then rigorously evaluate its behavior and risk exposures using a suite of econometric and machine learning models.

The analysis pipeline involves three major stages:
1.  **Index Construction:** Building a dynamic index using Principal Component Analysis (PCA) on market capitalization data.
2.  **Factor Analysis:** Constructing a three-factor model (MKT, SMB, HML) adapted for the crypto market to explain the index's returns.
3.  **Predictive Modeling:** Applying a range of models—from traditional OLS and regularized regressions (Ridge, Lasso) to advanced machine learning (Random Forest, XGBoost) and time-series forecasting (ARIMAX)—to build a comprehensive understanding of the index.

## Key Features

* **Dynamic PCA Index:** Implements a rolling-window PCA methodology to create an index that adapts to changing market conditions.
* **Crypto-Adapted Factor Model:** Constructs Market (MKT), Size (SMB), and Value (WML, using a short-term reversal proxy) factors from daily market data.
* **Comprehensive Model Comparison:** Applies and compares five different regression/ML models to explain index returns.
* **Time-Series Forecasting:** Utilizes an ARIMAX model to forecast the index's future path based on its own history and external factors.
* **Robust Validation:** Employs time-series cross-validation (`TimeSeriesSplit`) to prevent look-ahead bias and ensure model results are realistic.
* **Efficient Data Handling:** Implements data caching for both raw data and intermediate calculations (`nc_results`) to significantly speed up repeated analyses.

## Project Structure

The repository is organized to separate data, scripts, notebooks, and results for clarity and maintainability.

```
PCA_Index/
│
├── asset_lists/
│   ├── cryptocurrency_list.csv    # List of cryptos for index construction
│   └── crypto_tickers.json        # List of cryptos for factor model
│
├── data/
│   ├── raw/
│   │   ├── coingecko_market_cap/  # Raw daily market cap data
│   │   └── ...
│   └── processed/
│       ├── dynamic_pca_index.csv  # Final calculated index
│       ├── three_factors.csv      # MKT, SMB, WML factor returns
│       ├── engineered_features.csv# Factors with lags and rolling stats
│       ├── full_data_cache.csv    # Cached merged data
│       └── nc_results_cache.json  # Cached PCA component selection results
│
├── notebooks/
│   ├── 1_data_exploration.ipynb
│   └── ... (All analysis notebooks)
│
├── results/
│   ├── figures/
│   │   ├── dynamic_index_pc1.png
│   │   └── factor_model_coefficients.png
│   └── tables/
│
├── scripts/
│   ├── data_ingestion.py        # Scripts to fetch and update data
│   ├── pca_index_calculation.py       # Main script to calculate the dynamic index
│   ├── factor_calculation.py    # Main script to calculate the 3 factors
│   ├── feature_engineering.py   # Script to generate lags and rolling stats
│   └── ml_model_analysis.py     # Script to run final model comparisons
│
├── src/
│   └── config.py                # Central configuration for paths and parameters
│
├── .gitignore
└── README.md
```

## How to Run the Analysis Pipeline

Execute the scripts from the root directory (`PCA_Index/`) in the following order. The scripts use caching, so subsequent runs will be much faster.

**Step 1: Calculate the Dynamic PCA Index**
This script creates `dynamic_pca_index.csv`.
```bash
python scripts/pca_index_calculation.py
```
*Use `python scripts/pca_index_calculation --refresh` to force a rebuild of all caches.*

**Step 2: Calculate the Three Factors**
This script creates `three_factors.csv`.
```bash
python scripts/factor_calculation.py
```

**Step 3: Engineer Additional Features**
This script uses the factors from Step 2 to create `engineered_features.csv`.
```bash
python scripts/feature_engineering.py
```

**Step 4: Model Analysis**
