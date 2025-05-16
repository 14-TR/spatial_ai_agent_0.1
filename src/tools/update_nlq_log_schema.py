import os
import psycopg
from psycopg.sql import SQL, Identifier
from dotenv import load_dotenv

def update_nlq_agent_log_table():
    """
    Connects to the PostgreSQL database and adds new columns
    to the 'nlq_agent_log' table for evaluator feedback.
    """
    load_dotenv()

    db_name = os.getenv("POSTGRES_DB")
    db_user = os.getenv("POSTGRES_USER")
    db_password = os.getenv("POSTGRES_PASSWORD")
    db_host = os.getenv("POSTGRES_HOST")
    db_port = os.getenv("POSTGRES_PORT", "5432") # Default to 5432 if not set

    if not all([db_name, db_user, db_password, db_host]):
        print("Error: Database connection details (POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST) not found in environment variables.")
        print("Please ensure your .env file is correctly set up.")
        return

    conn_string = f"dbname='{db_name}' user='{db_user}' password='{db_password}' host='{db_host}' port='{db_port}'"
    
    table_name = "nlq_agent_log"
    columns_to_add = {
        "evaluator_overall_success_score": "INTEGER",
        "evaluator_critique_text": "TEXT",
        "schema_context_used": "TEXT"
    }

    try:
        with psycopg.connect(conn_string) as conn:
            with conn.cursor() as cur:
                print(f"Successfully connected to database '{db_name}' on '{db_host}'.")
                
                for col_name, col_type in columns_to_add.items():
                    # Check if column exists before attempting to add it
                    cur.execute(SQL("""
                        SELECT EXISTS (
                            SELECT 1
                            FROM information_schema.columns
                            WHERE table_name = %s AND column_name = %s
                        );
                    """), (table_name, col_name))
                    
                    column_exists = cur.fetchone()[0]

                    if not column_exists:
                        print(f"Column '{col_name}' does not exist in table '{table_name}'. Adding it...")
                        alter_table_sql = SQL("ALTER TABLE {table} ADD COLUMN {column_name} {column_type};").format(
                            table=Identifier(table_name),
                            column_name=Identifier(col_name),
                            column_type=SQL(col_type) # Use SQL() for type as it's part of the DDL
                        )
                        cur.execute(alter_table_sql)
                        print(f"Successfully added column '{col_name}' with type '{col_type}' to table '{table_name}'.")
                    else:
                        print(f"Column '{col_name}' already exists in table '{table_name}'. Skipping.")
                
                conn.commit()
                print("Schema update process completed successfully for relevant columns.")

    except psycopg.Error as e:
        error_message = str(e).lower()
        if "must be owner" in error_message or "permission denied" in error_message:
            print(f"\nDATABASE PERMISSION ERROR for user '{db_user}': {e}")
            print("The script failed because this user does not have permission to alter the table.")
            print("\nTo fix this, please follow these steps:")
            print("1. Connect to your PostgreSQL database using a superuser account (e.g., 'postgres') via psql or a GUI tool.")
            print(f"2. Execute the following SQL command to grant ownership of the table to '{db_user}':")
            print(f"   ALTER TABLE {table_name} OWNER TO \"{db_user}\";") # Ensure db_user is quoted if it contains special chars or is case sensitive
            print("   (If your username is simple, quotes might not be needed: ALTER TABLE nlq_agent_log OWNER TO your_user;)")
            print("3. After successfully running the command, re-run this Python script.")
        else:
            print(f"Database error: {e}")
            print("An error occurred during the schema update.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    print("Attempting to update 'nlq_agent_log' table schema...")
    update_nlq_agent_log_table() 