-- Placeholder for test_viirs_fire_events table schema
-- Please replace with your actual CREATE TABLE statement.
/*
Example (based on generate_test_data.py):
CREATE TABLE test_viirs_fire_events (
    event_id SERIAL PRIMARY KEY,
    geom GEOMETRY(Point, 4326) NOT NULL,
    detection_timestamp TIMESTAMPTZ NOT NULL,
    brightness_kelvin NUMERIC,
    confidence_percentage NUMERIC,
    satellite_source VARCHAR(50)
);
*/ 