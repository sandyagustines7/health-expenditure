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

# File paths for datasets
file_paths = [
    "Normalized_LE_Dataset.csv",
    "Normalized_MR_Dataset.csv"
]

try:
    # Connect to the database
    conn = psycopg2.connect(**db_params)
    cur = conn.cursor()

    # Fetch mappings
    cur.execute("SELECT country_id, country_name, region_id FROM health_metrics.countries;")
    country_mapping = {row[1]: (row[0], row[2]) for row in cur.fetchall()}  # country_name -> (country_id, region_id)

    cur.execute("SELECT indicator_id, indicator_code FROM health_metrics.indicators;")
    indicator_mapping = {row[1]: row[0] for row in cur.fetchall()}  # indicator_code -> indicator_id

    # Loop through each file and process it
    for file_path in file_paths:
        print(f"Processing file: {file_path}")

        # Load the dataset
        df = pd.read_csv(file_path)

        # Map country_name to country_id and region_id
        df["country_id"] = df["country_name"].map(lambda x: country_mapping.get(x, (None, None))[0])
        df["region_id"] = df["country_name"].map(lambda x: country_mapping.get(x, (None, None))[1])

        # Map indicator_code to indicator_id
        df["indicator_id"] = df["indicator_code"].map(indicator_mapping)

        # Drop rows with missing or invalid mappings
        before_drop = len(df)
        df = df.dropna(subset=["country_id", "indicator_id", "region_id"])
        rows_dropped = before_drop - len(df)
        print(f"Rows dropped due to unmapped country_name, indicator_code, or region_id: {rows_dropped}")

        # Ensure columns match the health_data table structure
        df = df[["country_id", "indicator_id", "year", "value", "region_id"]]  # Adjusted column order

        # Ensure all columns have correct data types
        df["country_id"] = df["country_id"].astype(int)
        df["indicator_id"] = df["indicator_id"].astype(int)
        df["region_id"] = df["region_id"].astype(int)
        df["year"] = df["year"].astype(int)
        df["value"] = df["value"].astype(float)

        # Save the processed dataset temporarily (optional)
        processed_file_path = f"processed_{file_path.split('/')[-1]}"
        df.to_csv(processed_file_path, index=False)

        # Import the processed data using the COPY command
        with open(processed_file_path, "r") as f:
            cur.copy_expert("""
                COPY health_metrics.health_data (country_id, indicator_id, year, value, region_id)
                FROM STDIN
                WITH CSV HEADER;
            """, f)

        print(f"File imported successfully: {file_path}")

    # Commit the transaction after all files are processed
    conn.commit()
    print("All files imported successfully!")

except Exception as e:
    print(f"An error occurred: {e}")
    if 'conn' in locals():
        conn.rollback()

finally:
    # Close the cursor and connection
    if 'cur' in locals():
        cur.close()
    if 'conn' in locals():
        conn.close()

