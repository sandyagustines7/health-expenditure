# Exploratory Data Analysis

# This code will initialize the connection to my database containing relevant health indicators, outcomes and their respective values per country.
# Once initialization is complete, the below code will generate descriptive statistics calculations, correlation and clustering analysis.

import psycopg2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

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
    
# Define reusable query function 
def run_query(query, conn):
    print(f"Running query: {query}")
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            columns = [desc[0] for desc in cur.description] # Fetch column names
            data = pd.DataFrame(cur.fetchall(), columns=columns)
        print("Query executed successfully!")
        return data
    except Exception as e:
        print(f"Error running query: {e}")
        return None

# Create EDA functions
# 1. Inspect the data and drop any missing values
def inspect_data(df): 
    print("\nInspecting the data...")
    print(df.info())
    print(df.describe())
    print("Missing values by column:\n", df.isnull().sum())
    return df.dropna(subset=["value"])

# 2. Descriptive Statistics 
def descriptive_analysis(df): # Calculates descriptive statistics for the 'value' column
    print("\nCalculating descriptive statistics...")
    try:
        # Aggregates 
        agg_stats = df[df["indicator_id"].isin([3,4,5])].groupby(["indicator_id", "year"])["value"].agg(["mean", "median", "min", "max", "std"])
        print("Aggregate descriptive statistics calculated successfully!")

        # Per Capita 
        per_capita_stats = df[df["indicator_id"].isin([7,8,11])].groupby(["indicator_id", "year"])["value"].agg(["mean", "median", "min", "max", "std"])
        print("Per Capita descriptive statistics have been calculated successfully!")
        return agg_stats, per_capita_stats
    except Exception as e:
        print(f"Error in descriptive statistics calculation: {e}")
        return None

# 3. Time Series Analysis 
def plot_trends(df, indicators, title): # Plots trends over time
    print(f"\nPlotting trends over time for {title}...")
    try:
        plt.figure(figsize=(10,6))
        for ind in indicators: 
            subset = df[df["indicator_id"] == ind].groupby("year")["value"].mean()
            plt.plot(subset.index, subset.values, label=f"Indicator {ind}")
        
        plt.xlabel("Year")
        plt.ylabel("Value")
        plt.legend()
        plt.title(title)
        plt.show()
        print("Trend plot generated successfully!")
    except Exception as e:
        print(f"Error in time series analysis: {e}")

# 4. Correlation Analysis 
def correlation_heatmap(df_pivot, indicators, title="Correlation Heatmap"): # Computes correlation matrix for selected indicators
    try: 
        print(f"\nGenerating correlation heatmap for indicators: {indicators}")
        
        # Select relevant indicators
        subset = df_pivot[indicators]

        # Drop missing values
        subset = subset.dropna()

        # Compute correlation matrix 
        corr_matrix = subset.corr()

        # Plot heatmap
        plt.figure(figsize=(8,6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title(title)
        plt.show()

        print("Correlation heatmap generated successfully!")
        return corr_matrix
    except Exception as e:
        print(f"Error in generating correlation heatmap: {e}")
        return None

def correlation_analysis(df, subset_indicators): # Generates correlation heatmap based on the above function
    print("\nPerforming correlation analysis...")
    try:
        # Check for duplicates and resolve 
        if df.duplicated(subset=["country_id", "indicator_id"]).any():
            print("Duplicates found - resolving by taking the mean of duplicate entries.")
            df = df.groupby(["country_id", "indicator_id"], as_index=False)["value"].mean()
        
        # Pivot table to reshape it for correlation calculations
        print("Pivoting the data...")
        df_pivot = df.pivot(index="country_id", columns="indicator_id", values="value")
        
        # Flatten list of indicators from subset dictionary
        indicators = [indicator for group in subset_indicators.values() for indicator in group]

        # Generate correlation heatmap for selected indicators 
        corr_matrix = correlation_heatmap(df_pivot, indicators, title="Correlation Heatmap - Subset of Indicators")
        return corr_matrix
    except Exception as e:
        print(f"Error in correlation analysis: {e}")
        return None

# 5. Clustering 
def cluster_data(df, num_clusters=4): # Applies KMeans Clustering to data
    print("\nPerforming KMeans clustering...")
    try:
        # Check for duplicates and resolve 
        if df.duplicated(subset=["country_id", "indicator_id", "year"]).any():
            print("Duplicates found - resolving by taking the mean of duplicate entries.")
            df = df.groupby(["country_id", "indicator_id", "year"], as_index=False)["value"].mean()

        selected_indicators = [6,7,18] # GDP per capita, CHE per capita, Life Expectancy

        # Ensure indicators used are within the selected indicators only 
        df_filtered = df[df["indicator_id"].isin(selected_indicators)]

        # Select 5 unique country IDs
        unique_countries = df["country_id"].unique()
        sampled_country_ids = pd.Series(unique_countries).sample(5, random_state=42)

        print("\nSelected Country IDs:", sampled_country_ids.tolist())

        # Filter data based on selected indicators and unique countries
        df_filtered=df[df["indicator_id"].isin(selected_indicators) & df["country_id"].isin(sampled_country_ids)]

        # Pivot table to ensure time-series data is retained
        df_pivot = df_filtered.pivot(index=["country_id", "year"], columns="indicator_id", values="value")

        # Handle missing values
        df_pivot =df_pivot.fillna(df_pivot.mean())

        if df_pivot.empty: 
            print("Error: Pivot table is empty. No data available for clustering.")
            return None
    
        # Standardize the data to account for different units 
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_pivot)

        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        df_pivot["Cluster"] = kmeans.fit_predict(df_scaled)
        print("Clustering completed!")
        print(df_pivot.groupby("Cluster").size())

        return df_pivot
    except Exception as e:
        print(f"Error in clustering: {e}")
        return None

# Create main function for query execution and analysis
def main():
    print("Starting EDA script...")
    conn = connect_to_db(db_params)
    if not conn:
        print("Terminating script due to connection error.")
        return

    # Run test query to test connection 
    test_query = "SELECT * FROM health_metrics.health_data LIMIT 10;"
    test_df = run_query(test_query, conn)
    if test_df is None or test_df.empty:
        print("Test query returned no results. Please check database.")
        return conn

    # Print results
    print('\nTest Query Results:')
    print(test_df)

    # Run query to retrieve full health_data table for EDA 
    health_data_query = "SELECT * FROM health_metrics.health_data WHERE indicator_id IN (1,3,4,6,7,8,11,14,18,19) AND value is NOT NULL;"
    health_data_df = run_query(health_data_query, conn)
    if health_data_df is None or health_data_df.empty:
        print("No data retrieved from the health_data table. Terminating script.")
        return conn

    # Data Inspection Results 
    print("\n### Inspecting Data ###")
    health_data_df = inspect_data(health_data_df)

    # Global duplicate check and resolution
    print("\n### Checking for Duplicates ###")
    duplicates = health_data_df[health_data_df.duplicated(subset=["country_id", "indicator_id", "year"], keep=False)]
    if not duplicates.empty: 
        print(f"Found duplicate entries - resolving by taking the mean of duplicate values...")
        print(f"Number of duplicates: {len(duplicates)}")
        health_data_df= health_data_df.groupby(["country_id", "indicator_id", "year"],as_index=False)["value"].mean()
    else:
        print("No duplicate entries found.")

    # EDA Results 
    print("\n### Descriptive Statistics ###")
    agg_stats, per_capita_stats = descriptive_analysis(health_data_df)
    print("\nAggregate Stats:\n", agg_stats)
    print("\nPer Capita Stats:\n", per_capita_stats)

    print("\n### Plotting Trends ###")
    plot_trends(health_data_df, indicators=[3,4,5], title="Aggregate Expenditure Trends Over Time")
    plot_trends(health_data_df, indicators=[7,8,11], title="Per Capita Expenditure Trends Over Time")

    print("\n### Correlation Analysis ###")
    # Define subset of indicators 
    subset_indicators = {
    "Economic Factors": [1,6], #GDP, GDP per capita
    "Health Expenditure": [3,7,14], #CHE, CHE per capita, OOPS as a % of CHE 
    "Health Outcomes": [18,19] # Life Expectancy, Under-5 Mortality Rate 
}
    corr_matrix = correlation_analysis(health_data_df, subset_indicators)
    if corr_matrix is not None: 
        print("\nCorrelation Matrix:", corr_matrix)

    print("\n### Clustering ###")
    sampled_countries = cluster_data(health_data_df)
    if sampled_countries is not None:
        print("\n Sampled Countries for Cluster Visualization:")
        print(sampled_countries.head(10))

        # Visualize clusters
        plt.figure(figsize=(10,7))
        sns.scatterplot(data=sampled_countries, x=6, y=7, hue="Cluster", palette="viridis", style="Cluster", s=100)
        plt.xlabel("GDP per Capita (US$)")
        plt.ylabel("Current Health Expenditure (CHE) per Capita (US$) - Subset of 5 Countries")
        plt.title("GDP vs CHE per Capita")
        plt.show()
    else: 
        print("Clustering failed - no output available.")

    # Close database connection 
    if conn: 
        conn.close()
        print("Database connection closed successfully!")

    print("EDA completed!")

if __name__ == "__main__":
    main()








