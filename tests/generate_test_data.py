import datetime
import random
from faker import Faker

# Initialize Faker
fake = Faker()

# --- Configuration ---
NUM_VIIRS_RECORDS = 500
NUM_ACLED_RECORDS = 500
OUTPUT_SQL_FILE = "data/test_data_dump.sql"

# Geographic bounding box for random points (optional, example: Horn of Africa)
# You can adjust this or make it more specific to a region of interest.
MIN_LAT, MAX_LAT = -5.0, 20.0  # Example latitudes
MIN_LON, MAX_LON = 30.0, 55.0  # Example longitudes

ACLED_EVENT_TYPES = [
    "Battles",
    "Explosions/Remote violence",
    "Protests",
    "Riots",
    "Strategic developments",
    "Violence against civilians",
]
ACLED_SUB_EVENT_TYPES = {
    "Battles": [
        "Armed clash",
        "Government regains territory",
        "Non-state actor overtakes territory",
    ],
    "Explosions/Remote violence": [
        "Air/drone strike",
        "Grenade",
        "IED attack",
        "Remote explosive/landmine/IED",
        "Shelling/artillery/missile attack",
    ],
    "Protests": [
        "Peaceful protest",
        "Protest with intervention",
        "Excessive force against protesters",
    ],
    "Riots": ["Mob violence", "Violent demonstration"],
    "Strategic developments": [
        "Agreement",
        "Arrests",
        "Change to group/activity",
        "Disrupted weapons use",
        "Headquarters or base established",
        "Looting/property destruction",
        "Non-violent transfer of territory",
    ],
    "Violence against civilians": [
        "Abduction/forced disappearance",
        "Attack",
        "Sexual violence",
        "Forced recruitment",
    ],
}
SATELLITE_SOURCES = ["Suomi NPP", "NOAA-20", "Aqua", "Terra"]


# --- Helper Functions ---
def random_point_in_bbox():
    lat = random.uniform(MIN_LAT, MAX_LAT)
    lon = random.uniform(MIN_LON, MAX_LON)
    return f"SRID=4326;POINT({lon:.5f} {lat:.5f})"


def generate_viirs_record(event_id):
    return (
        event_id,
        random_point_in_bbox(),
        fake.date_time_between(
            start_date="-2y", end_date="now", tzinfo=datetime.timezone.utc
        ).isoformat(),
        round(random.uniform(300.0, 700.0), 1),  # brightness_kelvin
        round(random.uniform(30.0, 100.0), 1),  # confidence_percentage
        random.choice(SATELLITE_SOURCES),
    )


def generate_acled_record(idx):
    event_type = random.choice(ACLED_EVENT_TYPES)
    sub_event_type = random.choice(ACLED_SUB_EVENT_TYPES[event_type])
    return (
        f"{fake.country_code().upper()}{idx+1000}",  # event_id
        fake.date_between(start_date="-5y", end_date="today").isoformat(),
        random_point_in_bbox(),
        fake.state(),  # admin1_name
        fake.city(),  # admin2_name (using city for simplicity, could be more specific)
        fake.street_address(),  # location_name (using street address for variety)
        event_type,
        sub_event_type,
        fake.company() + " Forces",  # actor1
        (
            fake.company() + " Militia" if random.random() > 0.3 else "Civilian Group"
        ),  # actor2
        random.randint(0, 50),  # fatalities
        fake.paragraph(nb_sentences=2),
        fake.url(),
    )


# --- SQL Definitions ---
SQL_CREATE_VIIRS = """
CREATE TABLE IF NOT EXISTS test_viirs_fire_events (
    event_id SERIAL PRIMARY KEY,
    geom GEOMETRY(Point, 4326) NOT NULL,
    detection_timestamp TIMESTAMPTZ NOT NULL,
    brightness_kelvin NUMERIC,
    confidence_percentage NUMERIC,
    satellite_source VARCHAR(50)
);
"""

SQL_CREATE_ACLED = """
CREATE TABLE IF NOT EXISTS test_acled_conflict_events (
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
"""


# --- Main Script Logic ---
def main():
    sql_statements = []

    # Add CREATE TABLE statements
    sql_statements.append(SQL_CREATE_VIIRS)
    sql_statements.append(SQL_CREATE_ACLED)

    # Generate VIIRS data
    viirs_inserts = []
    for i in range(NUM_VIIRS_RECORDS):
        record = generate_viirs_record(i + 1)
        # Properly format string and geometry values for SQL INSERT
        # SRID=4326;POINT(...) is the EWKT format, which PostGIS understands directly.
        sql_insert_viirs = f"""INSERT INTO test_viirs_fire_events (geom, detection_timestamp, brightness_kelvin, confidence_percentage, satellite_source) VALUES (
            '{record[1]}', '{record[2]}', {record[3]}, {record[4]}, '{record[5]}'
        );"""
        viirs_inserts.append(sql_insert_viirs)
    sql_statements.extend(viirs_inserts)

    # Generate ACLED data
    acled_inserts = []
    for i in range(NUM_ACLED_RECORDS):
        record = generate_acled_record(i + 1)
        sql_insert_acled = f"""INSERT INTO test_acled_conflict_events (event_id, event_date, geom, admin1_name, admin2_name, location_name, event_type, sub_event_type, actor1, actor2, fatalities, notes, source) VALUES (
            '{record[0]}', '{record[1]}', '{record[2]}', '{record[3].replace("'", "''")}', '{record[4].replace("'", "''")}', '{record[5].replace("'", "''")}', '{record[6]}', '{record[7]}', '{record[8].replace("'", "''")}', '{record[9].replace("'", "''")}', {record[10]}, '{record[11].replace("'", "''")}', '{record[12].replace("'", "''")}'
        );"""
        acled_inserts.append(sql_insert_acled)
    sql_statements.extend(acled_inserts)

    # Ensure data directory exists
    os.makedirs(os.path.dirname(OUTPUT_SQL_FILE), exist_ok=True)

    # Write to SQL file
    with open(OUTPUT_SQL_FILE, "w", encoding="utf-8") as f:
        for stmt in sql_statements:
            f.write(stmt + "\n")

    print(
        f"Successfully generated {NUM_VIIRS_RECORDS} VIIRS records and {NUM_ACLED_RECORDS} ACLED records."
    )
    print(f"SQL dump file created at: {OUTPUT_SQL_FILE}")
    print("\nNext steps:")
    print(
        f"1. Ensure your PostgreSQL service is running and you have created the '{os.getenv("POSTGRES_DB", "spatial_ai_db")}' database."
    )
    print(f"2. Connect to your database using psql or pgAdmin.")
    print(
        f"3. Execute the generated SQL script: psql -U {os.getenv("POSTGRES_USER", "postgres")} -d {os.getenv("POSTGRES_DB", "spatial_ai_db")} -a -f {OUTPUT_SQL_FILE}"
    )
    print(
        f"   (Replace username and dbname if different from your .env settings or defaults)"
    )


if __name__ == "__main__":
    # For the script to access POSTGRES_DB and POSTGRES_USER from .env for the print instructions,
    # it needs dotenv if you run it directly in an environment where .env isn't pre-loaded by poetry.
    # However, poetry run python scripts/generate_test_data.py should handle .env loading via python-dotenv if it's a project dependency.
    # Let's add it just in case for direct execution or if poetry's .env loading isn't picking up.
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        print(
            "dotenv library not found, .env variables might not be loaded for print instructions if run outside poetry."
        )
        pass  # Proceed without it, os.getenv will use defaults if not set.
    import os  # Moved import os here to be available after dotenv load attempt

    main()
