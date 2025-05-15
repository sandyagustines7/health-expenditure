# Advanced Statistical Analysis 

# The below code builds off my exploratory data analysis, running in-depth time-series analysis (ARIMA), regression and ML techniques.
# The primary goal of this analysis is to predict spending trends and impact on health outcomes. 

import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add database connection parameters 
db_params = {
    "dbname": "health_data",
    "user": "postgres",
    "password": "what",
    "host": "localhost",
    "port": 5432
}

# Establish database connection w/ cursor 
def connect_to_db(params):
    print("Attempting to connect to database...")
    try: 
        conn = psycopg2.connect(**db_params)
        print("Database connection successful!")
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

# Query data 
def fetch_data(query):
    conn = connect_to_db(db_params)
    if not conn:
        return None
    try:
        df = pd.read_sql(query, conn)
        print("Query executed successfully!")
        return df
    except Exception as e: 
        print(f"Error with fetching data: {e}")
        return None
    finally: 
        conn.close()

# Fetch and process data 
query = "SELECT country_id, year, indicator_id, value FROM health_metrics.health_data WHERE indicator_id IN (1,3,6,7,9,14,18,19)"

df = fetch_data(query)
if df is None: 
    print("No data fetched, exiting...")
    exit()

# Convert year to datetime and set index
df["year"] = pd.to_datetime(df["year"],format="%Y")
df.set_index('year', inplace=True)

# 1. Time Series Forecasting (ARIMA)
# Forecasting future CHE (Current Health Expenditure) and Life Expectancy

# Import modules for ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression

# Prepare time series data 
che_series = df[df["indicator_id"] == 7]["value"].resample("YE").mean()
life_exp_series = df[df["indicator_id"] == 18]["value"].resample("YE").mean()

# Stationarity Test 
def check_stationarity(series, name):
    adf_result = adfuller(series.dropna())
    print(f"ADF Test for {name}: p-value = {adf_result[1]}")

check_stationarity(life_exp_series, "Life Expectancy")
check_stationarity(che_series, "CHE per Capita")

def forecast_arima_with_linear(series, title, order, steps=10):
    print(f"\nRunning ARIMA and Linear Regression forecasting for {title}")

    # Fit ARIMA model with optimized parameters
    model = ARIMA(series, order=order)
    arima_result = model.fit()
    forecast_arima = arima_result.forecast(steps=steps)

    # Prepare data for Linear Regression
    series = series.dropna()  # Remove NaNs for regression
    years = np.array(series.index.year).reshape(-1, 1)  # Convert years to numerical format
    values = np.array(series.values).reshape(-1, 1)

    # Train Linear Regression Model
    lr_model = LinearRegression()
    lr_model.fit(years, values)

    # Generate Linear Regression Predictions
    future_dates = pd.date_range(start=series.index[-1], periods=steps+1, freq="YE")[1:].to_pydatetime()
    future_years = np.array([date.year for date in future_dates]).reshape(-1, 1)

    # Make predictions
    forecast_linear = lr_model.predict(future_years)

    # Plot results
    plt.figure(figsize=(10,5))
    plt.plot(series.index, series.values, label="Actual Data", color="blue")
    plt.plot(pd.date_range(start=series.index[-1], periods=steps+1, freq="YE")[1:], 
             forecast_arima, label="ARIMA Forecast", linestyle="dashed", color="red")
    plt.plot(pd.date_range(start=series.index[-1], periods=steps+1, freq="YE")[1:], 
         forecast_linear, label="Linear Regression", linestyle="dotted", color="green")
    plt.xlabel("Year")
    plt.ylabel(title)
    plt.legend()
    plt.title(f"ARIMA vs. Linear Regression Forecast - {title}")
    plt.show()

    return forecast_arima, forecast_linear

# Calling function to forecast CHE per capita (indicator_id = 7)
forecast_arima_with_linear(che_series, "CHE Per Capita", order=(3,1,1), steps=10)

# Calling function to forecast Life Expectancy (indicator_id = 18)
forecast_arima_with_linear(life_exp_series, "Life Expectancy", order=(3,1,2), steps=10)

# 2. Ridge Regression Function 
# Predicting health outcomes based on country spending 

# Import modules for Regression 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Pivot table to create separate columns for indicator values
df_pivot = df.pivot(index=["country_id", "year"], columns="indicator_id", values="value")

# Rename columns for readability 
df_pivot.rename(columns={
    1: "gdp", # Gross Domestic Product (GDP)
    3: "che_total", # Current Health Expenditure (CHE)
    6: "gdp_per_capita", # GDP per Capita
    7: "che_per_capita", # CHE per Capita
    9: "oops_per_capita", # Out-of-Pocket Expenditure (OOPS) per Capita
    18: "life_expectancy", # Life Expectancy (Target Variable)
    19: "under_5_mortality" # Under-5 Mortality
}, inplace=True)

# Reset index 
df_pivot.reset_index(inplace=True)

# Ensure no missing values 
df_pivot.dropna(inplace=True)

# Define Independent and Target variables
X = df_pivot[["gdp", "che_total", "gdp_per_capita", "che_per_capita", "oops_per_capita", "under_5_mortality"]]
y = df_pivot["life_expectancy"]

# Standardize features 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train and split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Fit Ridge Regression Model 
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train,y_train)

# Predict on test data 
y_pred = ridge_model.predict(X_test)

# Evaluate model performance 
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-Squared (R2): {r2:.4f}")

# Plot actual vs predicted values 
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred, alpha=0.6, color="blue", label="Predicted vs Actual")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle="dashed", color="red", label="Ideal Fit")
plt.xlabel("Actual Life Expectancy")
plt.ylabel("Predicted Life Expectancy")
plt.title("Ridge Regression: Actual vs. Predicted Life Expectancy")
plt.legend()
plt.show()

# 3. GMM (Gaussian Mixture Model) Function 
# Identifying and grouping countries with similar economic and health indicators 

# Import modules for GMM
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import seaborn as sns

# Select key indicators for clustering 
cluster_features = ["gdp_per_capita", "che_per_capita", "oops_per_capita", "life_expectancy", "under_5_mortality"]
df_cluster = df_pivot[["country_id", "year"] + cluster_features].dropna()

# Standardize data 
scaler = StandardScaler()
X_cluster = scaler.fit_transform(df_cluster[cluster_features])

# Fit GMM 
num_clusters = 3
gmm = GaussianMixture(n_components=num_clusters, random_state=42)
df_cluster["Cluster"] = gmm.fit_predict(X_cluster)

# Evaluation - Compute Silhouette Score 
silhouette_avg = silhouette_score(X_cluster, df_cluster["Cluster"])
print(f"\nSilhouette Score for {num_clusters} Clusters: {silhouette_avg:.4f}")

# Testing different 'num_clusters' 
for k in range(2,7):
    gmm_test = GaussianMixture(n_components=k, random_state=42)
    cluster_labels = gmm_test.fit_predict(X_cluster)
    silhouette_avg = silhouette_score(X_cluster, cluster_labels)
    print(f"Silhouette Score for {k} Clusters: {silhouette_avg:.4f}")

# Visualize clusters 
plt.figure(figsize=(10,6))
sns.scatterplot(data=df_cluster, x="gdp_per_capita", y="life_expectancy", hue="Cluster", palette="viridis", alpha=0.7) # Change depending on indicators you wish to visualize 
plt.xlabel("GDP per Capita")
plt.ylabel("Life Expectancy")
plt.title("Clustering Analysis: Country Groups based on Economic and Health Indicators")
plt.legend(title="Cluster")
plt.show()



