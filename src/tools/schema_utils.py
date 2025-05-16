# src/utils/schema_utils.py
import pathlib
from typing import List
import os  # Keep os for getenv in case it was used in load_schemas implicitly, though not in current version

SCHEMAS_DIR = pathlib.Path(__file__).resolve().parent.parent / "schemas"


def load_schemas(table_names: List[str]) -> str:
    """Loads schema definitions for the given table names from .sql files in SCHEMAS_DIR."""
    loaded_schema_parts = []
    print(
        f"[Schema Utils] Attempting to load schemas from: {SCHEMAS_DIR}"
    )  # Debug print
    for table_name in table_names:
        schema_file = SCHEMAS_DIR / f"{table_name}.sql"
        try:
            with open(schema_file, "r", encoding="utf-8") as f:
                loaded_schema_parts.append(f.read().strip())
            print(f"[Schema Utils] Successfully loaded: {schema_file}")  # Debug print
        except FileNotFoundError:
            print(
                f"[Schema Utils] Warning: Schema file not found for table '{table_name}' at {schema_file}"
            )
            # Optionally, include a placeholder or skip if a schema is critical
            # loaded_schema_parts.append(f"-- Schema for table '{table_name}' not found. --")
        except Exception as e:
            print(
                f"[Schema Utils] Warning: Error loading schema for table '{table_name}': {e}"
            )

    if not loaded_schema_parts:
        return "No specific table schemas provided. Please infer from the question or general knowledge."
    return "\n\n".join(loaded_schema_parts)
