import dspy

# REMOVED: from dspy import OpenAI
from typing import Optional, Dict, Any, List
import time
import uuid
import os

# import pathlib # No longer needed here if SCHEMAS_DIR and load_schemas are moved

from dotenv import load_dotenv

load_dotenv()

# Import from new utils module
from ..tools.schema_utils import load_schemas

try:
    from ..tools.db import log_nlq_interaction, execute_sql_on_postgis
except ImportError:
    import sys

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.tools.db import log_nlq_interaction, execute_sql_on_postgis

    # Also ensure utils is findable in this fallback path
    from src.tools.schema_utils import load_schemas as load_schemas_fallback

    # This is getting a bit complex; proper packaging or consistent PYTHONPATH is better long-term
    # For now, if the first load_schemas fails, try this one.
    # However, the primary import should work if streamlit is run from root.

# --- Schema Loading Function ---
# MOVED to src/utils/schema_utils.py
# SCHEMAS_DIR = pathlib.Path(__file__).parent.parent / "schemas"
# def load_schemas(table_names: List[str]) -> str:
# ... (function body removed)


# --- DSPy Signatures ---
class NLQToSQL(dspy.Signature):
    """Your task is to translate a natural language question into a SQL query.
    Context: You are given database table schemas in the 'context' field.
    Question: The user's question is in the 'nl_question' field.
    Output: You MUST output ONLY the SQL query in the 'sql_query' field.
    ABSOLUTELY NO EXPLANATIONS, NO INTRODUCTORY TEXT, NO MARKDOWN, NO CODE FENCES (```sql).
    The 'sql_query' field must contain nothing but the SQL statement itself.
    The SQL query must be safe, read-only (SELECT or WITH statements only).
    The SQL query will be run against a PostgreSQL database with PostGIS extension.
    Use only tables and columns defined in the provided schema context.
    If information is missing from the schema, do not infer or invent columns/tables.
    Example Output for 'sql_query': SELECT COUNT(*) FROM my_table WHERE condition = 'value';
    """

    context = dspy.InputField(
        desc="Schema of relevant tables (e.g., CREATE TABLE statements)."
    )
    nl_question = dspy.InputField(desc="Natural language question.")
    sql_query = dspy.OutputField(
        desc="ONLY the SQL query. NO additional text. NO markdown. Example: SELECT * FROM table;"
    )


# --- DSPy Modules ---


class BasicNLQAgent(dspy.Module):
    def __init__(self, agent_version="0.1.0-dev"):
        super().__init__()
        self.llm_model_name = "gpt-4o-mini"
        self.llm = None  # Initialize to None
        self.generate_sql = None  # Initialize to None
        self.agent_version = agent_version

        # --- Detailed Initialization Logging ---
        api_key_status = os.getenv("OPENAI_API_KEY")
        print(f"[BasicNLQAgent.__init__] Checking OPENAI_API_KEY...")
        if not api_key_status:
            print(
                f"[BasicNLQAgent.__init__] OPENAI_API_KEY not found initially. Attempting load_dotenv()."
            )
            load_dotenv()  # Ensure .env is loaded
            api_key_status = os.getenv("OPENAI_API_KEY")

        if api_key_status:
            print(f"[BasicNLQAgent.__init__] OPENAI_API_KEY found.")
        else:
            print(
                f"[BasicNLQAgent.__init__] ERROR: OPENAI_API_KEY IS STILL NOT FOUND AFTER ATTEMPTED LOAD. THIS IS LIKELY THE ISSUE."
            )
            # No point in proceeding if API key is missing
            return

        try:
            print(
                f"[BasicNLQAgent.__init__] Attempting to initialize dspy.LM with model: openai/{self.llm_model_name}"
            )
            self.llm = dspy.LM(
                f"openai/{self.llm_model_name}", max_tokens=300
            )  # Corrected line
            print(
                f"[BasicNLQAgent.__init__] Successfully initialized self.llm for model: openai/{self.llm_model_name}"
            )

            self.generate_sql = dspy.Predict(NLQToSQL)
            print(
                f"[BasicNLQAgent.__init__] Initialized self.generate_sql = dspy.Predict(NLQToSQL). Current self.generate_sql.lm: {getattr(self.generate_sql, 'lm', 'N/A')}"
            )

            # Manually set lm on generate_sql if it's None and self.llm is available
            if (
                hasattr(self.generate_sql, "lm")
                and not self.generate_sql.lm
                and self.llm
            ):
                print(
                    f"[BasicNLQAgent.__init__] self.generate_sql.lm is None. Manually assigning self.llm to it."
                )
                self.generate_sql.lm = self.llm  # Use our created self.llm instance
                print(
                    f"[BasicNLQAgent.__init__] After manual assignment, self.generate_sql.lm: {self.generate_sql.lm}"
                )
            elif hasattr(self.generate_sql, "lm") and self.generate_sql.lm:
                print(
                    f"[BasicNLQAgent.__init__] self.generate_sql.lm was already set upon initialization: {self.generate_sql.lm}"
                )
            else:
                print(
                    f"[BasicNLQAgent.__init__] Could not manually set self.generate_sql.lm (either it has no 'lm' attribute, or self.llm is None). This is unexpected."
                )

        except Exception as e:
            print(f"---------- [BasicNLQAgent.__init__] CRITICAL ERROR ----------")
            print(f"Failed to initialize dspy.LM or dspy.Predict for BasicNLQAgent.")
            print(f"Model attempted: openai/{self.llm_model_name}")
            print(
                f"OPENAI_API_KEY available at exception: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}"
            )
            print(f"Full Exception Type: {type(e).__name__}")
            print(f"Full Exception Message: {e}")
            import traceback

            print(f"Traceback: {traceback.format_exc()}")
            print(f"-------------------------------------------------------------")
            self.llm = None  # Ensure llm is None on failure
            self.generate_sql = None  # Ensure generate_sql is None on failure
        # --- End Detailed Initialization Logging ---

    def forward(
        self,
        natural_language_query: str,
        schema_context: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        start_time = time.time()
        if session_id is None:
            session_id = str(uuid.uuid4())

        generated_sql: Optional[str] = None
        sql_result_data: Optional[List[Dict[str, Any]]] = None
        processed_result_text: Optional[str] = None
        error_message: Optional[str] = None
        prompt_tokens: Optional[int] = None
        completion_tokens: Optional[int] = None
        total_tokens: Optional[int] = None
        cost_usd_cents: Optional[float] = None
        current_llm_model_used = self.llm.model if self.llm else "llm_not_configured"

        try:
            if not self.llm:
                raise RuntimeError(
                    "LLM not configured for BasicNLQAgent. Cannot proceed with query generation."
                )

            # --- Use self.generate_sql initialized in __init__ ---
            if not self.generate_sql:
                print(
                    "[BasicNLQAgent.forward] CRITICAL ERROR: self.generate_sql (Predict module) is None. Initialization failed."
                )
                raise RuntimeError(
                    "SQL generation module (self.generate_sql) not initialized in BasicNLQAgent."
                )

            if not self.generate_sql.lm:
                print(
                    f"[BasicNLQAgent.forward] CRITICAL ERROR: self.generate_sql.lm is None. The Predict module in __init__ did not get an LM."
                )
                print(
                    f"[BasicNLQAgent.forward] Current self.llm in forward: {self.llm} (type: {type(self.llm)})"
                )
                if self.llm:
                    print(
                        f"[BasicNLQAgent.forward] Current self.llm.model: {getattr(self.llm, 'model', 'N/A')}"
                    )
                print(f"[BasicNLQAgent.forward] dspy.settings.lm: {dspy.settings.lm}")
                raise RuntimeError(
                    "SQL generation module (self.generate_sql) does not have an LM configured."
                )

            print(f"\n[BasicNLQAgent.forward] Using self.generate_sql from __init__.")
            print(f"self.generate_sql instance: {self.generate_sql}")
            print(
                f"self.generate_sql.lm instance: {self.generate_sql.lm} (type: {type(self.generate_sql.lm)})"
            )
            if self.generate_sql.lm:
                print(
                    f"self.generate_sql.lm.model: {getattr(self.generate_sql.lm, 'model', 'N/A')}"
                )
            # ---- End __init__ checks ----

            effective_load_schemas = load_schemas
            if "load_schemas_fallback" in globals() and not callable(load_schemas):
                effective_load_schemas = load_schemas_fallback

            if schema_context is None or not schema_context.strip():
                print(
                    "Warning: No specific schema context provided to agent.forward(). Using fallback."
                )
                schema_context = effective_load_schemas(["nlq_agent_log"])
                if (
                    not schema_context.strip()
                    or "No specific table schemas provided" in schema_context
                ):
                    schema_context = "No specific table schemas provided. Please infer table structure from the question."

            response = self.generate_sql(
                nl_question=natural_language_query, context=schema_context
            )
            generated_sql = response.sql_query
            print(f"Generated SQL: {generated_sql}")

            if (
                dspy.settings.lm
                and hasattr(dspy.settings.lm, "history")
                and dspy.settings.lm.history
            ):
                last_call = dspy.settings.lm.history[-1]
                response_data = last_call.get("response", {})
                usage_data = response_data.get("usage", {})
                prompt_tokens = usage_data.get("prompt_tokens")
                completion_tokens = usage_data.get("completion_tokens")
                total_tokens = usage_data.get("total_tokens")
                if prompt_tokens is None:
                    prompt_tokens = last_call.get("prompt_tokens")
                if completion_tokens is None:
                    completion_tokens = last_call.get("completion_tokens")
                if total_tokens is None:
                    total_tokens = last_call.get("total_tokens")

            if generated_sql:
                if not execute_sql_on_postgis:
                    raise RuntimeError(
                        "Database execution tool (execute_sql_on_postgis) not available."
                    )
                print(f"Executing SQL: {generated_sql}")
                sql_result_data = execute_sql_on_postgis(
                    generated_sql, is_agent_query=True
                )
                print(f"SQL Result: {sql_result_data}")
                if sql_result_data is not None:
                    if isinstance(sql_result_data, list) and len(sql_result_data) > 0:
                        processed_result_text = (
                            f"Query executed successfully. Result: {sql_result_data}"
                        )
                    elif (
                        isinstance(sql_result_data, list) and len(sql_result_data) == 0
                    ):
                        processed_result_text = (
                            "Query executed successfully, but no data was returned."
                        )
                    else:
                        processed_result_text = f"Query executed. Unexpected result format: {sql_result_data}"
                else:
                    processed_result_text = (
                        "Error during SQL execution or query returned no data."
                    )
        except Exception as e:
            error_message = str(e)
            print(f"Agent error: {error_message}")
            processed_result_text = f"Error during agent processing: {error_message}"
            if "generated_sql" not in locals() or not isinstance(generated_sql, str):
                generated_sql = None
        finally:
            latency_ms = int((time.time() - start_time) * 1000)
            log_payload_raw_result = None
            if (
                sql_result_data
                and isinstance(sql_result_data, list)
                and len(sql_result_data) > 0
            ):
                log_payload_raw_result = (
                    sql_result_data[0]
                    if isinstance(sql_result_data[0], dict)
                    else {"data": sql_result_data}
                )
            elif sql_result_data:
                log_payload_raw_result = {"data": sql_result_data}
            if log_nlq_interaction:
                log_id = log_nlq_interaction(
                    natural_language_query=natural_language_query,
                    session_id=session_id,
                    generated_sql_query=generated_sql,
                    sql_execution_raw_result=log_payload_raw_result,
                    processed_analysis_result=processed_result_text,
                    agent_version=self.agent_version,
                    llm_model_used=current_llm_model_used,
                    latency_ms=latency_ms,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cost_usd_cents=cost_usd_cents,
                )
                if log_id:
                    print(f"NLQ Agent interaction logged with ID: {log_id}")
                else:
                    print(
                        "NLQ Agent interaction logging FAILED or logger not available."
                    )
            else:
                print("NLQ Agent interaction logging skipped: logger not available.")
        return {
            "natural_language_query": natural_language_query,
            "generated_sql": generated_sql,
            "execution_result": sql_result_data,
            "processed_output": processed_result_text,
            "error": error_message,
            "session_id": session_id,
            "llm_model_used": current_llm_model_used,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "latency_ms": latency_ms,
        }


if __name__ == "__main__":
    print("Initializing BasicNLQAgent...")
    if not os.getenv("OPENAI_API_KEY"):
        print("FATAL ERROR: OPENAI_API_KEY environment variable not set.")
        print("Please set it before running the agent.")
    else:
        agent = BasicNLQAgent()
        if agent.llm:
            # Determine which load_schemas to use (primary or fallback)
            # This logic should ideally be simplified by consistent PYTHONPATH or packaging
            effective_load_schemas = load_schemas
            if "load_schemas_fallback" in globals() and not callable(load_schemas):
                print("Using fallback schema loader for __main__")
                effective_load_schemas = load_schemas_fallback
            elif not callable(load_schemas):
                print(
                    "CRITICAL: load_schemas not callable and no fallback. Schemas won't load for tests."
                )

                # This would be a problem, so we might want to stop or ensure a dummy function.
                def effective_load_schemas(x):
                    return "Error: Schema loader not available."

            print("\n--- Test Case 1: Count logs ---")
            nl_query1 = "How many log entries are there in the nlq_agent_log table?"
            schema_ctx1 = effective_load_schemas(["nlq_agent_log"])
            response1 = agent.forward(
                natural_language_query=nl_query1, schema_context=schema_ctx1
            )
            print(f"Agent Response:\n{response1}\n")

            print("\n--- Test Case 2: List recent logs ---")
            nl_query2 = "Show me the last 2 log entries, just their natural language query and generated sql."
            schema_ctx2 = effective_load_schemas(["nlq_agent_log"])
            response2 = agent.forward(
                natural_language_query=nl_query2, schema_context=schema_ctx2
            )
            print(f"Agent Response:\n{response2}\n")

            print("\n--- Test Case 3: A more complex query on logs ---")
            nl_query3 = "What are the distinct llm_model_used values from the logs where the agent_version is '0.1.0-dev'?"
            schema_ctx3 = effective_load_schemas(["nlq_agent_log"])
            response3 = agent.forward(
                natural_language_query=nl_query3, schema_context=schema_ctx3
            )
            print(f"Agent Response:\n{response3}\n")

            print(
                "\n--- Test Case 4: A query that might generate non-SELECT (should be caught by DB or agent logic) ---"
            )
            nl_query4 = "Delete all logs older than yesterday."
            schema_ctx4 = effective_load_schemas(["nlq_agent_log"])
            response4 = agent.forward(
                natural_language_query=nl_query4, schema_context=schema_ctx4
            )
            print(f"Agent Response:\n{response4}\n")

            print("\n--- Test Case 5: VIIRS - Count fire events ---")
            nl_query5 = "How many fire events are in the test VIIRS table?"
            schema_ctx5 = effective_load_schemas(["test_viirs_fire_events"])
            response5 = agent.forward(
                natural_language_query=nl_query5, schema_context=schema_ctx5
            )
            print(f"Agent Response:\n{response5}\n")

            print("\n--- Test Case 6: VIIRS - List 3 events by Suomi NPP ---")
            nl_query6 = "List 3 fire events detected by Suomi NPP."
            schema_ctx6 = effective_load_schemas(["test_viirs_fire_events"])
            response6 = agent.forward(
                natural_language_query=nl_query6, schema_context=schema_ctx6
            )
            print(f"Agent Response:\n{response6}\n")

            print("\n--- Test Case 7: VIIRS - Brightness and location ---")
            nl_query7 = "Show fire events with brightness above 600 Kelvin, include their detection time and location geometry as text."
            schema_ctx7 = effective_load_schemas(["test_viirs_fire_events"])
            response7 = agent.forward(
                natural_language_query=nl_query7, schema_context=schema_ctx7
            )
            print(f"Agent Response:\n{response7}\n")

            print("\n--- Test Case 9: ACLED - Count conflict events ---")
            nl_query9 = "How many conflict events are in the test ACLED table?"
            schema_ctx9 = effective_load_schemas(["test_acled_conflict_events"])
            response9 = agent.forward(
                natural_language_query=nl_query9, schema_context=schema_ctx9
            )
            print(f"Agent Response:\n{response9}\n")

            print("\n--- Test Case 10: ACLED - List 2 recent events ---")
            nl_query10 = "List the 2 most recent conflict events, showing event date, type, and location name."
            schema_ctx10 = effective_load_schemas(["test_acled_conflict_events"])
            response10 = agent.forward(
                natural_language_query=nl_query10, schema_context=schema_ctx10
            )
            print(f"Agent Response:\n{response10}\n")

            print("\n--- Test Case 11: ACLED - Fatalities in Battles ---")
            nl_query11 = (
                "How many total fatalities were recorded in 'Battles' type events?"
            )
            schema_ctx11 = effective_load_schemas(["test_acled_conflict_events"])
            response11 = agent.forward(
                natural_language_query=nl_query11, schema_context=schema_ctx11
            )
            print(f"Agent Response:\n{response11}\n")

            print(
                "\n--- Test Case 13 (Combined Context): Fires near recent conflicts ---"
            )
            nl_query13 = "Show me fire events that occurred within the last 30 days from test_viirs_fire_events and also list conflict events from test_acled_conflict_events in the same period, focusing on 'Battles'."
            schema_ctx13 = effective_load_schemas(
                ["test_viirs_fire_events", "test_acled_conflict_events"]
            )
            response13 = agent.forward(
                natural_language_query=nl_query13, schema_context=schema_ctx13
            )
            print(f"Agent Response:\n{response13}\n")

        else:
            print("Agent tests skipped as LLM could not be configured.")

    # To test thoroughly:
    # 1. Ensure PostGIS is running and nlq_agent_log table exists.
    # 2. Set OPENAI_API_KEY.
    # 3. Run this script: `poetry run python src/agents/nlq_agent.py` (or similar)
