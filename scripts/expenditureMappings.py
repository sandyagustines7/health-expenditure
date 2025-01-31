import psycopg2
import pandas as pd

# Database connection parameters
db_params = {
    "dbname": "health_data",
    "user": "postgres",
    "password": "what",
    "host": "localhost",
    "port": 5432
}

# File path to the normalized dataset
file_paths = [
    "Normalized_Aggregates_Dataset.csv",
    "Normalized_CHE_Dataset.csv",
    "Normalized_GDP_Dataset.csv",
    "Normalized_GGE_Dataset.csv",
    "Normalized_Per_Capita_Dataset.csv"
]  

try:
    # Connect to the database
    conn = psycopg2.connect(**db_params)
    cur = conn.cursor()

    # Fetch country_name -> country_id mapping
    cur.execute("SELECT country_id, country_name FROM health_metrics.countries;")
    country_mapping = {row[1]: row[0] for row in cur.fetchall()}

    # Fetch indicator_code -> indicator_id mapping
    cur.execute("SELECT indicator_id, indicator_code FROM health_metrics.indicators;")
    indicator_mapping = {row[1]: row[0] for row in cur.fetchall()}

    # Loop through each file and process it
    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        
        # Load the dataset
        df = pd.read_csv(file_path)

        # Map country_name and indicator_code to their IDs
        df["country_id"] = df["country_name"].map(country_mapping)
        df["indicator_id"] = df["indicator_code"].map(indicator_mapping)

        # Drop rows with unmapped values
        df = df.dropna(subset=["country_id", "indicator_id"])

        # Select only the necessary columns for the database
        df = df[["country_id", "indicator_id", "year", "value"]]

        # Save the processed dataset temporarily (optional)
        processed_file_path = f"processed_{file_path}"
        df.to_csv(processed_file_path, index=False)

        # Import the processed data using the COPY command
        with open(processed_file_path, "r") as f:
            cur.copy_expert("""
                COPY health_metrics.health_data (country_id, indicator_id, year, value)
                FROM STDIN
                WITH CSV HEADER
            """, f)

        print(f"File imported successfully: {file_path}")

    # Commit the transaction after all files are processed
    conn.commit()
    print("All files imported successfully.")

except Exception as e:
    print("An error occurred:", e)
    if 'conn' in locals():
        conn.rollback()

finally:
    # Close the cursor and connection
    if 'cur' in locals():
        cur.close()
    if 'conn' in locals():
        conn.close()
