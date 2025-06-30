import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV, LinearRegression
from sklearn.metrics import r2_score
import os

from src import config
import pandas_datareader.data as web


def get_risk_free_rate(start_date, end_date):
    """
    Fetches the 3-Month Treasury Bill Rate from FRED as the risk-free rate.
    """
    print("Fetching risk-free rate from FRED...")
    try:
        # The series for 3-Month Treasury Bill Secondary Market Rate
        rf = web.DataReader('DTB3', 'fred', start_date, end_date)
        # Forward-fill missing values and convert to daily rate based on 360-day year convention
        rf = rf.ffill() / 100 / 360
        return rf
    except Exception as e:
        print(f"Could not fetch risk-free rate: {e}. Returning zero series.")
        return pd.Series(0, index=pd.date_range(start=start_date, end=end_date), name='DTB3')


def run_regression_analysis():
    """
    Loads the PCA index and factors, runs OLS, Ridge, and Lasso regressions,
    and presents the interpretation of the results.
    """
    # --- 1. Load Data ---
    print("Loading calculated index and factors...")
    try:
        index_df = pd.read_csv(config.INDEX_FILE_PATH, index_col='Date', parse_dates=True)
        factors_df = pd.read_csv(config.THREE_FACTOR_FILE_PATH, index_col='Date', parse_dates=True)
    except FileNotFoundError as e:
        print(f"ERROR: Make sure you have run the calculation scripts first. Missing file: {e.filename}")
        return

    # --- 2. Prepare Data for Regression ---
    print("Preparing data for regression analysis...")
    # Merge the dataframes on their common dates to ensure alignment
    full_df = index_df.join(factors_df, how='inner')

    # Calculate daily LOG returns for the PCA index
    full_df['pca_return'] = np.log(full_df['Value']).diff()

    # Get the risk-free rate for the same period
    rf_rate = get_risk_free_rate(full_df.index.min(), full_df.index.max())
    full_df['rf_rate'] = rf_rate['DTB3'].reindex(full_df.index, method='ffill').fillna(0)

    # Calculate the excess return of the PCA index (this is our dependent variable 'y')
    full_df['pca_excess_return'] = full_df['pca_return'] - full_df['rf_rate']

    # Drop any NaN values that were created from the calculations
    full_df.dropna(inplace=True)

    # Define our independent variables (X) and dependent variable (y)
    # Using HML as the standard name for the value factor
    X = full_df[['MKT', 'SMB', 'WML']]
    y = full_df['pca_excess_return']

    # Scale the features for Ridge and Lasso, as they are sensitive to feature scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)

    # --- 3. Train Regression Models ---
    print("Training models...")
    # --- A. OLS using statsmodels for p-values and detailed stats ---
    # Use unscaled X for statsmodels to keep coefficients interpretable
    X_sm = sm.add_constant(X)  # You must add the intercept manually
    sm_model = sm.OLS(y, X_sm)
    sm_results = sm_model.fit()

    # --- B. Scikit-learn models for prediction and regularized coefficients ---
    # Define a range of alphas for cross-validation
    alphas = np.logspace(-6, 2, 100)

    # Scikit-learn OLS for baseline coefficient comparison
    ols_sk = LinearRegression()
    ols_sk.fit(X_scaled, y)

    # RidgeCV finds the best alpha using cross-validation
    ridge_cv = RidgeCV(alphas=alphas)
    ridge_cv.fit(X_scaled, y)

    # LassoCV finds the best alpha using cross-validation
    lasso_cv = LassoCV(alphas=alphas, cv=5, random_state=42)
    lasso_cv.fit(X_scaled, y)

    print("Model training complete.")

    # --- 4. Analyze and Interpret Results ---
    print("\n" + "=" * 60)
    print("              MODEL INTERPRETATION & RESULTS")
    print("=" * 60)

    # Create a summary DataFrame for easy comparison
    # Note: scikit-learn coefficients are from scaled data, statsmodels are not.
    # We use statsmodels for interpretable coeffs and p-values.
    results_summary = pd.DataFrame({
        'OLS_Coefficient': sm_results.params,
        'P-Value': sm_results.pvalues,
    }).rename(index={'const': 'Intercept'})

    # Add Lasso coefficients (which can be zero)
    lasso_coeffs = pd.Series(lasso_cv.coef_, index=X.columns)
    results_summary['Lasso_Coefficient'] = lasso_coeffs

    # Add Ridge coefficients
    ridge_coeffs = pd.Series(ridge_cv.coef_, index=X.columns)
    results_summary['Ridge_Coefficient'] = ridge_coeffs

    results_summary.fillna('-', inplace=True)  # Fill NaN for intercept row

    print("\n--- Model Coefficients, Betas, and P-Values ---")
    print(results_summary.to_string(formatters={'P-Value': '{:,.4f}'.format, 'OLS_Coefficient': '{:,.6f}'.format}))

    print("\n--- Interpretation Guide ---")
    print("OLS_Coefficient: The beta from standard regression. Use this for magnitude interpretation.")
    print("P-Value:         Tests if the OLS coefficient is statistically different from 0. (p < 0.05 is significant).")
    print("Lasso/Ridge Coeff: These are on scaled data. Primarily observe which factors Lasso sets to zero.")

    print(f"\nOptimal Alpha for Ridge: {ridge_cv.alpha_:.6f}")
    print(f"Optimal Alpha for Lasso: {lasso_cv.alpha_:.6f}")

    # --- Model Performance ---
    r2_ols = sm_results.rsquared_adj  # Adjusted R-squared is a more robust metric
    r2_ridge = r2_score(y, ridge_cv.predict(X_scaled))
    r2_lasso = r2_score(y, lasso_cv.predict(X_scaled))

    print("\n--- Model Performance (R-squared) ---")
    print(f"OLS Adj. R-squared: {r2_ols:.4f}")
    print(f"Ridge R-squared:    {r2_ridge:.4f}")
    print(f"Lasso R-squared:    {r2_lasso:.4f}")
    print(
        f"\nInterpretation: An Adj. R-squared of {r2_ols:.2f} means ~{r2_ols:.0%} of your index's daily variance is explained by these factors.")

    # --- 5. Visualize Coefficients ---
    print("\nGenerating coefficient plot...")

    # Use statsmodels coefficients for visualization as they are more interpretable
    coeffs_to_plot = results_summary[['OLS_Coefficient']].drop('Intercept').reset_index()
    coeffs_to_plot.columns = ['Factor', 'Coefficient']

    plt.figure(figsize=(10, 6))
    sns.barplot(data=coeffs_to_plot, x='Factor', y='Coefficient')
    plt.title('OLS Factor Model Coefficients (Betas)', fontsize=16)
    plt.ylabel('Coefficient Value (Beta)')
    plt.xlabel('Factor')
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the figure
    figure_path = config.FIGURES_DIR / 'factor_model_ols_coefficients.png'
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_path)
    print(f"Plot saved to {figure_path}")
    plt.show()


if __name__ == '__main__':
    run_regression_analysis()