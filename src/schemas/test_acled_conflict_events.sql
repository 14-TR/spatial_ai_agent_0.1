-- Placeholder for test_acled_conflict_events table schema
-- Please replace with your actual CREATE TABLE statement.
/*
Example (based on generate_test_data.py):
CREATE TABLE test_acled_conflict_events (
    event_id VARCHAR(50) PRIMARY KEY,
    event_date DATE NOT NULL,
    geom GEOMETRY(Point, 4326) NOT NULL,
    admin1_name VARCHAR(100),
    admin2_name VARCHAR(100),
    location_name VARCHAR(255),
    event_type VARCHAR(100) NOT NULL,
    sub_event_type VARCHAR(100),
    actor1 VARCHAR(255),
    actor2 VARCHAR(255),
    fatalities INTEGER,
    notes TEXT,
    source VARCHAR(255)
);
*/ 