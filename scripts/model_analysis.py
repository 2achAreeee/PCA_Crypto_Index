import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm

from src import config
import pandas_datareader.data as web


def get_risk_free_rate(start_date, end_date):
    """
    Fetches the 3-Month Treasury Bill Rate from FRED as the risk-free rate.
    """
    try:
        rf = web.DataReader('DTB3', 'fred', start_date, end_date)
        rf = rf.ffill() / 100 / 360  # Convert to daily rate
        return rf
    except Exception as e:
        print(f"Could not fetch risk-free rate: {e}. Returning zero series.")
        return pd.Series(0, index=pd.date_range(start=start_date, end=end_date), name='DTB3')


def run_regression_analysis():
    """
    Loads the PCA index and factors, runs Ridge and Lasso regressions,
    and prints the interpretation of the results.
    """
    # --- 1. Load Data ---
    print("Loading calculated index and factors...")
    try:
        index_df = pd.read_csv(config.INDEX_FILE_PATH, index_col='Date', parse_dates=True)
        factors_df = pd.read_csv(config.THREE_FACTOR_FILE_PATH, index_col='Date', parse_dates=True)
    except FileNotFoundError as e:
        print(f"Error: Make sure you have run the calculation scripts first. Missing file: {e.filename}")
        return

    # --- 2. Prepare Data for Regression ---
    print("Preparing data for regression analysis...")
    # Merge the dataframes on their common dates
    full_df = index_df.join(factors_df, how='inner')

    # Calculate daily returns for the PCA index
    full_df['pca_return'] = full_df['Value'].pct_change()

    # Get the risk-free rate for the same period
    rf_rate = get_risk_free_rate(full_df.index.min(), full_df.index.max())
    full_df['rf_rate'] = rf_rate['DTB3'].reindex(full_df.index, method='ffill')

    # Calculate the excess return of the PCA index (this is our 'y')
    full_df['pca_excess_return'] = full_df['pca_return'] - full_df['rf_rate']

    # Drop any NaN values that were created
    full_df.dropna(inplace=True)

    # Define our independent variables (X) and dependent variable (y)
    X = full_df[['MKT', 'SMB', 'WML']]
    y = full_df['pca_excess_return']

    # --- IMPORTANT: Scale the features for Ridge and Lasso ---
    # These models are sensitive to the scale of the input features.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)

    # --- 3. Train Regression Models ---
    print("Training models...")
    # Define a range of alphas for cross-validation
    alphas = np.logspace(-6, 6, 100)

    # Ordinary Least Squares (OLS) for baseline comparison
    ols = LinearRegression()
    ols.fit(X_scaled, y)

    # RidgeCV finds the best alpha using cross-validation
    ridge_cv = RidgeCV(alphas=alphas, store_cv_values=True)
    ridge_cv.fit(X_scaled, y)

    # LassoCV finds the best alpha using cross-validation
    lasso_cv = LassoCV(alphas=alphas, cv=5, random_state=42)
    lasso_cv.fit(X_scaled, y)

    print("Model training complete.")

    # --- 4. Analyze and Interpret Results ---
    print("\n" + "=" * 50)
    print("              MODEL INTERPRETATION")
    print("=" * 50)

    # Create a DataFrame to hold the coefficients for easy comparison
    coeffs = pd.DataFrame({
        'Factor': X.columns,
        'OLS': ols.coef_,
        'Ridge': ridge_cv.coef_,
        'Lasso': lasso_cv.coef_
    }).set_index('Factor')

    print("\n--- Factor Coefficients (Betas) ---")
    print(coeffs)
    print("\nInterpretation:")
    print(" - The coefficient shows how much your index return is expected to change if a factor changes by one unit.")
    print(
        " - MKT (Market Beta): Sensitivity to overall market movements. A value > 1 means it's more volatile than the market.")
    print(
        " - SMB (Size Beta): A positive value indicates a tilt towards smaller cryptos; negative means a tilt towards larger ones.")
    print(
        " - HML (Value Beta): A positive value indicates a tilt towards 'value' cryptos (recent losers); negative means a tilt towards 'growth' cryptos (recent winners).")
    print(
        " - NOTE: Lasso's ability to set coefficients to ZERO suggests it considers that factor unimportant for explaining your index's returns.")

    print(f"\nOptimal Alpha for Ridge: {ridge_cv.alpha_:.6f}")
    print(f"Optimal Alpha for Lasso: {lasso_cv.alpha_:.6f}")

    # --- Model Performance ---
    r2_ols = r2_score(y, ols.predict(X_scaled))
    r2_ridge = r2_score(y, ridge_cv.predict(X_scaled))
    r2_lasso = r2_score(y, lasso_cv.predict(X_scaled))

    print("\n--- Model Performance (R-squared) ---")
    print(f"OLS R-squared:   {r2_ols:.4f}")
    print(f"Ridge R-squared: {r2_ridge:.4f}")
    print(f"Lasso R-squared: {r2_lasso:.4f}")
    print("\nInterpretation:")
    print(
        f" - An R-squared of {r2_ridge:.2f} means that approximately {r2_ridge:.0%} of the daily movements in your PCA index can be explained by the three factors.")

    # --- 5. Visualize Coefficients ---
    print("\nGenerating coefficient plot...")
    coeffs_to_plot = coeffs.reset_index().melt(id_vars='Factor', var_name='Model', value_name='Coefficient')

    plt.figure(figsize=(12, 7))
    sns.barplot(data=coeffs_to_plot, x='Factor', y='Coefficient', hue='Model')
    plt.title('Factor Model Coefficients: OLS vs. Ridge vs. Lasso', fontsize=16)
    plt.ylabel('Coefficient Value (Beta)')
    plt.xlabel('Factor')
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the figure
    figure_path = config.FIGURES_DIR / 'factor_model_coefficients.png'
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_path)
    print(f"Plot saved to {figure_path}")
    plt.show()


if __name__ == '__main__':
    run_regression_analysis()