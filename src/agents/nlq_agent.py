import dspy
import logging

# REMOVED: from dspy import OpenAI
from typing import Optional, Dict, Any, List
import time
import uuid
import os

# import pathlib # No longer needed here if SCHEMAS_DIR and load_schemas are moved

from dotenv import load_dotenv

load_dotenv()

# --- Logging Configuration (Module Level) ---
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT) # Set to DEBUG for dev
logger = logging.getLogger(__name__) # Define module-level logger

# Import from new utils module
from ..tools.schema_utils import load_schemas
from .evaluator_agent import InteractionEvaluator

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
        self.evaluator = None

        # --- Detailed Initialization Logging ---
        api_key_status = os.getenv("OPENAI_API_KEY")
        logger.info("[BasicNLQAgent.__init__] Checking OPENAI_API_KEY...")
        if not api_key_status:
            logger.warning("[BasicNLQAgent.__init__] OPENAI_API_KEY not found initially. Attempting load_dotenv().")
            load_dotenv()  # Ensure .env is loaded
            api_key_status = os.getenv("OPENAI_API_KEY")

        if api_key_status:
            logger.info("[BasicNLQAgent.__init__] OPENAI_API_KEY found.")
        else:
            logger.error("[BasicNLQAgent.__init__] OPENAI_API_KEY IS STILL NOT FOUND AFTER ATTEMPTED LOAD. THIS IS LIKELY THE ISSUE.")
            # No point in proceeding if API key is missing
            return

        try:
            logger.info(f"[BasicNLQAgent.__init__] Attempting to initialize dspy.LM with model: openai/{self.llm_model_name}")
            self.llm = dspy.LM(
                f"openai/{self.llm_model_name}", max_tokens=300
            )  # Corrected line
            logger.info(f"[BasicNLQAgent.__init__] Successfully initialized self.llm for model: openai/{self.llm_model_name}")

            self.generate_sql = dspy.Predict(NLQToSQL)
            logger.info(f"[BasicNLQAgent.__init__] Initialized self.generate_sql = dspy.Predict(NLQToSQL). Current self.generate_sql.lm: {getattr(self.generate_sql, 'lm', 'N/A')}")

            # --- Load optimized predictor state if available ---
            optimized_predictor_path = "compiled_nlq_predictor.json" # Adjust path if needed
            if os.path.exists(optimized_predictor_path):
                try:
                    logger.info(f"[BasicNLQAgent.__init__] Found '{optimized_predictor_path}'. Attempting to load its state.")
                    self.generate_sql.load(optimized_predictor_path)
                    logger.info(f"[BasicNLQAgent.__init__] Successfully loaded optimized state into self.generate_sql from '{optimized_predictor_path}'.")
                    # After loading, the LM might need to be re-assigned if it wasn't saved or if using a different LM instance
                    if not self.generate_sql.lm and self.llm:
                        logger.info("[BasicNLQAgent.__init__] LM was None after loading. Re-assigning self.llm.")
                        self.generate_sql.lm = self.llm
                    logger.debug(f"[BasicNLQAgent.__init__] self.generate_sql.lm after attempting load: {self.generate_sql.lm}")
                except Exception as load_err:
                    logger.error(f"[BasicNLQAgent.__init__] ERROR: Failed to load optimized predictor state from '{optimized_predictor_path}': {load_err}")
                    logger.warning("[BasicNLQAgent.__init__] Falling back to unoptimized predictor.")
            else:
                logger.info(f"[BasicNLQAgent.__init__] '{optimized_predictor_path}' not found. Using unoptimized predictor.")
            # --- End Load optimized predictor state ---

            # Manually set lm on generate_sql if it's None and self.llm is available
            if (
                hasattr(self.generate_sql, "lm")
                and not self.generate_sql.lm
                and self.llm
            ):
                logger.info("[BasicNLQAgent.__init__] self.generate_sql.lm is None. Manually assigning self.llm to it.")
                self.generate_sql.lm = self.llm  # Use our created self.llm instance
                logger.debug(f"[BasicNLQAgent.__init__] After manual assignment, self.generate_sql.lm: {self.generate_sql.lm}")
            elif hasattr(self.generate_sql, "lm") and self.generate_sql.lm:
                logger.debug(f"[BasicNLQAgent.__init__] self.generate_sql.lm was already set upon initialization: {self.generate_sql.lm}")
            else:
                logger.warning("[BasicNLQAgent.__init__] Could not manually set self.generate_sql.lm (either it has no 'lm' attribute, or self.llm is None). This is unexpected.")

        except Exception as e:
            logger.critical(f"---------- [BasicNLQAgent.__init__] CRITICAL ERROR ----------")
            logger.critical(f"Failed to initialize dspy.LM or dspy.Predict for BasicNLQAgent.")
            logger.critical(f"Model attempted: openai/{self.llm_model_name}")
            logger.critical(f"OPENAI_API_KEY available at exception: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}")
            logger.critical(f"Full Exception Type: {type(e).__name__}")
            logger.critical(f"Full Exception Message: {e}")
            import traceback

            logger.critical(f"Traceback: {traceback.format_exc()}")
            logger.critical(f"-------------------------------------------------------------")
            self.llm = None  # Ensure llm is None on failure
            self.generate_sql = None  # Ensure generate_sql is None on failure
        # --- End Detailed Initialization Logging ---

        # --- Initialize Interaction Evaluator ---
        try:
            logger.info(f"[BasicNLQAgent.__init__] Attempting to initialize InteractionEvaluator...")
            self.evaluator = InteractionEvaluator()
            if not self.evaluator.evaluate_interaction:
                logger.warning("[BasicNLQAgent.__init__] WARNING: InteractionEvaluator module failed to initialize properly.")
                self.evaluator = None
            else:
                logger.info("[BasicNLQAgent.__init__] InteractionEvaluator initialized successfully.")
        except Exception as e:
            logger.critical(f"[BasicNLQAgent.__init__] CRITICAL ERROR initializing InteractionEvaluator: {e}")
            import traceback
            logger.critical(traceback.format_exc())
            self.evaluator = None
        # --- End Interaction Evaluator Initialization ---

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
        current_llm_model_used: Optional[str] = None

        # Initialize evaluator feedback variables
        evaluator_score: Optional[int] = None
        evaluator_critique: Optional[str] = None

        try:
            if not self.llm:
                raise RuntimeError(
                    "LLM not configured for BasicNLQAgent. Cannot proceed with query generation."
                )

            # --- Use self.generate_sql initialized in __init__ ---
            if not self.generate_sql:
                logger.critical("[BasicNLQAgent.forward] CRITICAL ERROR: self.generate_sql (Predict module) is None. Initialization failed.")
                raise RuntimeError(
                    "SQL generation module (self.generate_sql) not initialized in BasicNLQAgent."
                )

            if not self.generate_sql.lm:
                logger.critical(f"[BasicNLQAgent.forward] CRITICAL ERROR: self.generate_sql.lm is None. The Predict module in __init__ did not get an LM.")
                logger.debug(f"[BasicNLQAgent.forward] Current self.llm in forward: {self.llm} (type: {type(self.llm)}) (Model: {getattr(self.llm, 'model', 'N/A') if self.llm else 'N/A'})")
                if self.llm:
                    logger.debug(f"[BasicNLQAgent.forward] Current self.llm.model: {getattr(self.llm, 'model', 'N/A')}")
                logger.debug(f"[BasicNLQAgent.forward] dspy.settings.lm: {dspy.settings.lm} (Model: {getattr(dspy.settings.lm, 'model', 'N/A') if dspy.settings.lm else 'N/A'})")
                raise RuntimeError(
                    "SQL generation module (self.generate_sql) does not have an LM configured."
                )

            logger.debug(f"[BasicNLQAgent.forward] Using self.generate_sql from __init__.")
            logger.debug(f"self.generate_sql instance: {self.generate_sql}")
            logger.debug(f"self.generate_sql.lm instance: {self.generate_sql.lm} (type: {type(self.generate_sql.lm)})")
            if self.generate_sql.lm:
                logger.debug(f"self.generate_sql.lm.model: {getattr(self.generate_sql.lm, 'model', 'N/A')}")
            # ---- End __init__ checks ----

            effective_load_schemas = load_schemas
            if "load_schemas_fallback" in globals() and not callable(load_schemas):
                effective_load_schemas = load_schemas_fallback

            if schema_context is None or not schema_context.strip():
                logger.warning("Warning: No specific schema context provided to agent.forward(). Using fallback to load all available schemas.")
                schema_context = effective_load_schemas([])
                if (
                    not schema_context.strip()
                    or "No specific table schemas provided" in schema_context
                ):
                    schema_context = "No table schemas could be loaded. Please ensure .sql files exist in the schemas directory."

            response = self.generate_sql(
                nl_question=natural_language_query, context=schema_context
            )
            generated_sql = response.sql_query
            logger.debug(f"Generated SQL: {generated_sql}")

            # --- Token and Cost Tracking ---
            prompt_tokens: Optional[int] = None
            completion_tokens: Optional[int] = None
            total_tokens: Optional[int] = None
            cost_usd_cents: Optional[float] = None
            current_llm_model_used: Optional[str] = None

            logger.debug("[BasicNLQAgent.forward] Initializing token/cost variables to None.")

            if (
                dspy.settings.lm
                and hasattr(dspy.settings.lm, "history")
                and dspy.settings.lm.history
            ):
                last_call = dspy.settings.lm.history[-1]
                logger.debug(f"[BasicNLQAgent.forward] Last LLM call history entry: {last_call}")
                
                current_llm_model_used = last_call.get("model_name", self.llm.model if self.llm else "llm_not_configured")
                logger.debug(f"[BasicNLQAgent.forward] Token Usage: current_llm_model_used: {current_llm_model_used}")

                # Corrected: 'usage' is a direct key in last_call
                usage_data = last_call.get("usage", {})
                
                if not usage_data: # If usage_data is still empty, log a warning
                    logger.warning(f"[BasicNLQAgent.forward] Token Usage: 'usage' field not found or empty in last_call. last_call content: {last_call}")

                prompt_tokens_raw = usage_data.get("prompt_tokens")
                completion_tokens_raw = usage_data.get("completion_tokens")
                total_tokens_raw = usage_data.get("total_tokens")

                try:
                    if prompt_tokens_raw is not None: prompt_tokens = int(prompt_tokens_raw)
                    if completion_tokens_raw is not None: completion_tokens = int(completion_tokens_raw)
                    if total_tokens_raw is not None: total_tokens = int(total_tokens_raw)
                    logger.debug(f"[BasicNLQAgent.forward] Converted tokens - prompt: {prompt_tokens}, completion: {completion_tokens}, total: {total_tokens}")
                except (ValueError, TypeError) as e:
                    logger.warning(f"[BasicNLQAgent.forward] Could not convert token values to int. Raw values - prompt: {prompt_tokens_raw} (type {type(prompt_tokens_raw)}), completion: {completion_tokens_raw} (type {type(completion_tokens_raw)}), total: {total_tokens_raw} (type {type(total_tokens_raw)}). Error: {e}")
                
                if total_tokens is None: 
                    logger.warning(f"[BasicNLQAgent.forward] total_tokens is still None after all attempts. Usage data: {usage_data}. Full last_call: {last_call}")
            else:
                logger.warning("[BasicNLQAgent.forward] No LLM history found or dspy.settings.lm not configured for token extraction.")

            if generated_sql:
                if not execute_sql_on_postgis:
                    raise RuntimeError(
                        "Database execution tool (execute_sql_on_postgis) not available."
                    )
                logger.info(f"Executing SQL: {generated_sql}")
                sql_result_data = execute_sql_on_postgis(
                    generated_sql, is_agent_query=True
                )
                logger.debug(f"SQL Result: {sql_result_data}")
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
            
            # --- Call Interaction Evaluator ---
            if self.evaluator and self.evaluator.evaluator_llm and generated_sql and processed_result_text:
                # Prepare a summary of the SQL result for the evaluator - MORE CONCISE
                sql_res_summary_for_eval = "SQL query did not return data or execution failed before result processing."
                if sql_result_data is not None:
                    if isinstance(sql_result_data, list):
                        if len(sql_result_data) > 0 and isinstance(sql_result_data[0], dict) :
                            sql_res_summary_for_eval = f"Query returned {len(sql_result_data)} rows. First row keys: {list(sql_result_data[0].keys())}."
                        elif len(sql_result_data) > 0:
                             sql_res_summary_for_eval = f"Query returned {len(sql_result_data)} rows. Data is a list of non-dict items."
                        else:
                            sql_res_summary_for_eval = "Query returned 0 rows."
                    else:
                         sql_res_summary_for_eval = f"Query returned data of type: {type(sql_result_data)}."

                # Ensure schema_context for evaluator is the one used for SQL generation
                schema_for_evaluator = schema_context
                if not schema_for_evaluator:
                    temp_effective_load_schemas = load_schemas
                    if "load_schemas_fallback" in globals() and not callable(load_schemas):
                        temp_effective_load_schemas = load_schemas_fallback
                    schema_for_evaluator = temp_effective_load_schemas([])
                    if not schema_for_evaluator.strip() or "No specific table schemas provided" in schema_for_evaluator:
                         schema_for_evaluator = "No specific table schemas provided. Please infer table structure from the question."

                logger.info(f"[BasicNLQAgent.forward] Calling InteractionEvaluator with NLQ: '{natural_language_query[:50]}...'")
                
                # Use dspy.settings to temporarily set the LM for the evaluator call
                # Store original lm to restore it, just in case, though context manager should handle it.
                # original_global_lm = dspy.settings.lm 
                with dspy.settings.context(lm=self.evaluator.evaluator_llm):
                    logger.debug(f"[BasicNLQAgent.forward] DEBUG: Temporarily set dspy.settings.lm for evaluator: {dspy.settings.lm} (Model: {getattr(dspy.settings.lm, 'model', 'N/A')})")
                    eval_response = self.evaluator.forward(
                        natural_language_query=natural_language_query,
                        schema_context=schema_for_evaluator, 
                        generated_sql=generated_sql,
                        sql_query_result_summary=sql_res_summary_for_eval,
                        final_answer_to_user=processed_result_text
                    )
                # print(f"[BasicNLQAgent.forward] DEBUG: Restored dspy.settings.lm after evaluator call: {dspy.settings.lm} (Model: {getattr(dspy.settings.lm, 'model', 'N/A') if dspy.settings.lm else 'N/A'})")
                
                evaluator_score = eval_response.get("overall_success_score")
                evaluator_critique = eval_response.get("critique_text")
                logger.info(f"[BasicNLQAgent.forward] Evaluator response: Score={evaluator_score}, Critique='{str(evaluator_critique)[:100]}...'")
            else:
                logger.info("[BasicNLQAgent.forward] Evaluator not available or not called.")

        except Exception as e:
            logger.error(f"Agent error in main try-except: {e}", exc_info=True)
            error_message = str(e)
            processed_result_text = f"Error during agent processing: {e}"
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
                    evaluator_overall_success_score=evaluator_score,
                    evaluator_critique_text=evaluator_critique,
                    schema_context_used=schema_context
                )
                if log_id:
                    logger.info(f"NLQ Agent interaction logged with ID: {log_id}")
                else:
                    logger.warning("NLQ Agent interaction logging FAILED or logger not available.")
            else:
                logger.warning("NLQ Agent interaction logging skipped: logger not available.")
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
    logger.info("Initializing BasicNLQAgent...")
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("FATAL ERROR: OPENAI_API_KEY environment variable not set.")
        logger.error("Please set it before running the agent.")
    else:
        agent = BasicNLQAgent()
        if agent.llm:
            # Determine which load_schemas to use (primary or fallback)
            # This logic should ideally be simplified by consistent PYTHONPATH or packaging
            effective_load_schemas = load_schemas
            if "load_schemas_fallback" in globals() and not callable(load_schemas):
                logger.info("Using fallback schema loader for __main__")
                effective_load_schemas = load_schemas_fallback
            elif not callable(load_schemas):
                logger.critical(
                    "CRITICAL: load_schemas not callable and no fallback. Schemas won't load for tests."
                )

                # This would be a problem, so we might want to stop or ensure a dummy function.
                def effective_load_schemas(x):
                    return "Error: Schema loader not available."

            logger.info("\n--- Test Case 1: Count logs ---")
            nl_query1 = "How many log entries are there in the nlq_agent_log table?"
            schema_ctx1 = effective_load_schemas(["nlq_agent_log"])
            response1 = agent.forward(
                natural_language_query=nl_query1, schema_context=schema_ctx1
            )
            logger.info(f"Agent Response:\n{response1}\n")

            logger.info("\n--- Test Case 2: List recent logs ---")
            nl_query2 = "Show me the last 2 log entries, just their natural language query and generated sql."
            schema_ctx2 = effective_load_schemas(["nlq_agent_log"])
            response2 = agent.forward(
                natural_language_query=nl_query2, schema_context=schema_ctx2
            )
            logger.info(f"Agent Response:\n{response2}\n")

            logger.info("\n--- Test Case 3: A more complex query on logs ---")
            nl_query3 = "What are the distinct llm_model_used values from the logs where the agent_version is '0.1.0-dev'?"
            schema_ctx3 = effective_load_schemas(["nlq_agent_log"])
            response3 = agent.forward(
                natural_language_query=nl_query3, schema_context=schema_ctx3
            )
            logger.info(f"Agent Response:\n{response3}\n")

            logger.info(
                "\n--- Test Case 4: A query that might generate non-SELECT (should be caught by DB or agent logic) ---"
            )
            nl_query4 = "Delete all logs older than yesterday."
            schema_ctx4 = effective_load_schemas(["nlq_agent_log"])
            response4 = agent.forward(
                natural_language_query=nl_query4, schema_context=schema_ctx4
            )
            logger.info(f"Agent Response:\n{response4}\n")

            logger.info("\n--- Test Case 5: VIIRS - Count fire events ---")
            nl_query5 = "How many fire events are in the test VIIRS table?"
            schema_ctx5 = effective_load_schemas(["test_viirs_fire_events"])
            response5 = agent.forward(
                natural_language_query=nl_query5, schema_context=schema_ctx5
            )
            logger.info(f"Agent Response:\n{response5}\n")

            logger.info("\n--- Test Case 6: VIIRS - List 3 events by Suomi NPP ---")
            nl_query6 = "List 3 fire events detected by Suomi NPP."
            schema_ctx6 = effective_load_schemas(["test_viirs_fire_events"])
            response6 = agent.forward(
                natural_language_query=nl_query6, schema_context=schema_ctx6
            )
            logger.info(f"Agent Response:\n{response6}\n")

            logger.info("\n--- Test Case 7: VIIRS - Brightness and location ---")
            nl_query7 = "Show fire events with brightness above 600 Kelvin, include their detection time and location geometry as text."
            schema_ctx7 = effective_load_schemas(["test_viirs_fire_events"])
            response7 = agent.forward(
                natural_language_query=nl_query7, schema_context=schema_ctx7
            )
            logger.info(f"Agent Response:\n{response7}\n")

            logger.info("\n--- Test Case 9: ACLED - Count conflict events ---")
            nl_query9 = "How many conflict events are in the test ACLED table?"
            schema_ctx9 = effective_load_schemas(["test_acled_conflict_events"])
            response9 = agent.forward(
                natural_language_query=nl_query9, schema_context=schema_ctx9
            )
            logger.info(f"Agent Response:\n{response9}\n")

            logger.info("\n--- Test Case 10: ACLED - List 2 recent events ---")
            nl_query10 = "List the 2 most recent conflict events, showing event date, type, and location name."
            schema_ctx10 = effective_load_schemas(["test_acled_conflict_events"])
            response10 = agent.forward(
                natural_language_query=nl_query10, schema_context=schema_ctx10
            )
            logger.info(f"Agent Response:\n{response10}\n")

            logger.info("\n--- Test Case 11: ACLED - Fatalities in Battles ---")
            nl_query11 = (
                "How many total fatalities were recorded in 'Battles' type events?"
            )
            schema_ctx11 = effective_load_schemas(["test_acled_conflict_events"])
            response11 = agent.forward(
                natural_language_query=nl_query11, schema_context=schema_ctx11
            )
            logger.info(f"Agent Response:\n{response11}\n")

            logger.info(
                "\n--- Test Case 13 (Combined Context): Fires near recent conflicts ---"
            )
            nl_query13 = "Show me fire events that occurred within the last 30 days from test_viirs_fire_events and also list conflict events from test_acled_conflict_events in the same period, focusing on 'Battles'."
            schema_ctx13 = effective_load_schemas(
                ["test_viirs_fire_events", "test_acled_conflict_events"]
            )
            response13 = agent.forward(
                natural_language_query=nl_query13, schema_context=schema_ctx13
            )
            logger.info(f"Agent Response:\n{response13}\n")

        else:
            logger.info("Agent tests skipped as LLM could not be configured.")

    # To test thoroughly:
    # 1. Ensure PostGIS is running and nlq_agent_log table exists.
    # 2. Set OPENAI_API_KEY.
    # 3. Run this script: `poetry run python src/agents/nlq_agent.py` (or similar)
