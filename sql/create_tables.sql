-- Create regions table
CREATE TABLE health_metrics.regions(
	region_id SERIAL PRIMARY KEY,
	region VARCHAR(255) NOT NULL, 
	sub_region VARCHAR(255)
);

-- Create countries table
CREATE TABLE health_metrics.countries (
	country_id SERIAL PRIMARY KEY,
	country_name VARCHAR(255) NOT NULL,
	region_INTEGER,
	sub_region VARCHAR(255),
	CONSTRAINT fk_region FOREIGN KEY (region_id) REFERENCES health_metrics.regions(region_id)
);

-- Create indicators table 
CREATE TABLE health_metrics.indicators(
	indicator_id SERIAL PRIMARY KEY,
	indicator_code VARCHAR(50),
	indicator_name VARCHAR(255),
	indicator_type VARCHAR(50)
);

-- Create health_data table 
CREATE TABLE health_metrics.health_data(
	id SERIAL PRIMARY KEY,
	country_id INTEGER NOT NULL,
	indicator_id INTEGER NOT NULL,
	year INTEGER NOT NULL,
	value DOUBLE PRECISION,
	region_id INTEGER,
	CONSTRAINT fk_country FOREIGN KEY(country_id) REFERENCES health_metrics.countries(country_id),
	CONSTRAINT fk_indicator FOREIGN KEY(indicator_id) REFERENCES health_metrics.indicators(indicator_id),
	CONSTRAINT fk_region FOREIGN KEY(region_id) REFERENCES health_metrics.regions(region_id)
);

-- Populate the region_id column in health_data table
UPDATE health_metrics.health_data hd
SET region_id = c.region_id
FROM health_metrics.countries c
WHERE hd.country_id = c.country_id;


