"""
Evaluation harness for the Natural Language Query (NLQ) Agent.

This script runs a predefined set of test cases against the NLQ agent,
logs its performance, and calculates basic metrics such as SQL generation success,
latency, and token usage.
"""

import os
import sys
import logging
import time
import json
import re
from typing import List, Dict, Any, Optional, Tuple

from dotenv import load_dotenv

# Add project root to sys.path for module imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from src.agents.nlq_agent import BasicNLQAgent
    from src.tools.schema_utils import load_schemas
    from src.tools.db import execute_sql_on_postgis # For validation queries if needed
except ImportError as e:
    print(f"Error importing modules: {e}. Ensure PYTHONPATH is set correctly or run from project root.")
    sys.exit(1)

# --- Logging Configuration ---
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# --- Evaluation Configuration ---
# Adapted from src/testing/batch_test_nlq_agent.py and project goals
EVALUATION_CASES: List[Dict[str, Any]] = [
    {
        "id": "VIIRS_COUNT_NPP",
        "nlq": "How many fires did the 'Suomi NPP' satellite detect in test_viirs_fire_events?",
        "target_tables": ["test_viirs_fire_events"],
        "expected_sql_pattern": r"SELECT COUNT\(\*\) FROM test_viirs_fire_events WHERE satellite_source = 'Suomi NPP'",
        "description": "Basic count with a WHERE clause on VIIRS data.",
    },
    {
        "id": "VIIRS_BRIGHTEST_FIRES",
        "nlq": "Tell me when and how hot the top 3 brightest fires were from test_viirs_fire_events.",
        "target_tables": ["test_viirs_fire_events"],
        "expected_sql_pattern": r"SELECT\s+.*?detection_timestamp.*?,\s*.*?brightness_kelvin.*?\s+FROM\s+test_viirs_fire_events\s+ORDER\s+BY\s+brightness_kelvin\s+DESC\s+LIMIT\s+3",
        "description": "Query with ORDER BY and LIMIT on VIIRS data.",
    },
    {
        "id": "ACLED_TOTAL_FATALITIES_PROTESTS",
        "nlq": "How many people died in total during 'Protests' in test_acled_conflict_events?",
        "target_tables": ["test_acled_conflict_events"],
        "expected_sql_pattern": r"SELECT SUM\(fatalities\) FROM test_acled_conflict_events WHERE event_type = 'Protests'",
        "description": "Aggregation (SUM) with a WHERE clause on ACLED data.",
    },
    {
        "id": "ACLED_RECENT_CONFLICTS",
        "nlq": "List the date, type of event, and place for 2 recent conflicts from test_acled_conflict_events.",
        "target_tables": ["test_acled_conflict_events"],
        "expected_sql_pattern": r"SELECT\s+.*?event_date.*?,\s*.*?event_type.*?,\s*.*?location_name.*?\s+FROM\s+test_acled_conflict_events\s+ORDER\s+BY\s+event_date\s+DESC\s+LIMIT\s+2",
        "description": "Query with multiple selected columns, ORDER BY, and LIMIT on ACLED data.",
    },
    {
        "id": "LOG_COUNT_TOTAL",
        "nlq": "How many interactions have been logged in total by the NLQ agent in nlq_agent_log?",
        "target_tables": ["nlq_agent_log"],
        "expected_sql_pattern": r"SELECT COUNT\(\*\) FROM nlq_agent_log",
        "description": "Basic count on the agent's log table.",
    },
    {
        "id": "LOG_HIGH_LATENCY",
        "nlq": "Show me log entries from nlq_agent_log with latency over 2000 milliseconds.",
        "target_tables": ["nlq_agent_log"],
        "expected_sql_pattern": r"SELECT\s+.*?\s+FROM\s+nlq_agent_log\s+WHERE\s+(latency_ms|latency)\s*>\s*2000",
        "description": "Query with a WHERE clause involving a numeric comparison on log data.",
    },
    {
        "id": "VIIRS_INVALID_QUERY_TYPE",
        "nlq": "Delete all VIIRS fire events older than a year from test_viirs_fire_events.",
        "target_tables": ["test_viirs_fire_events"],
        "expected_sql_pattern": None, # Expecting agent to refuse or DB to block
        "description": "Attempt a destructive query (DELETE) which should be blocked.",
        "expect_failure_or_rejection": True,
    },
]

def run_evaluation_suite() -> bool:
    """
    Runs the NLQ agent evaluation suite.

    Returns:
        bool: True if all tests passed based on defined criteria (e.g., accuracy threshold), False otherwise.
              For this bare-bones version, focuses on execution and basic metrics.
    """
    logger.info("--- Starting NLQ Agent Evaluation Suite ---")
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("FATAL ERROR: OPENAI_API_KEY environment variable not set.")
        logger.error("Please set it in your .env file before running the evaluation.")
        return False

    logger.info("Initializing BasicNLQAgent...")
    try:
        agent = BasicNLQAgent()
        if not agent.llm or not agent.generate_sql:
            logger.error("BasicNLQAgent did not initialize its LLM or SQL generation components correctly.")
            return False
    except Exception as e:
        logger.error(f"Error initializing BasicNLQAgent: {e}", exc_info=True)
        return False
    logger.info("BasicNLQAgent initialized successfully.")

    results: List[Dict[str, Any]] = []
    total_tests = len(EVALUATION_CASES)
    sql_generated_count = 0
    pattern_matched_count = 0
    total_latency_ms = 0
    total_tokens_used = 0
    failed_due_to_rejection = 0


    for i, case in enumerate(EVALUATION_CASES):
        logger.info(f"--- Running Case {i+1}/{total_tests}: {case['id']} ---")
        logger.info(f"Description: {case['description']}")
        logger.info(f"NLQ: {case['nlq']}")

        case_result: Dict[str, Any] = {
            "id": case["id"],
            "nlq": case["nlq"],
            "target_tables": case["target_tables"],
            "generated_sql": None,
            "processed_output": None,
            "error": None,
            "latency_ms": 0,
            "total_tokens": 0,
            "pattern_match_success": None,
        }

        start_time = time.time()
        try:
            schema_context = load_schemas(case["target_tables"])
            if "No specific table schemas provided" in schema_context and case["target_tables"]:
                logger.warning(f"Could not load schema for tables: {case['target_tables']}. Agent will infer.")
            
            agent_response = agent.forward(
                natural_language_query=case["nlq"], schema_context=schema_context
            )

            case_result["generated_sql"] = agent_response.get("generated_sql")
            case_result["processed_output"] = agent_response.get("processed_output")
            case_result["error"] = agent_response.get("error")
            case_result["latency_ms"] = agent_response.get("latency_ms", 0)
            
            # Handle tokens, ensuring they are ints for accumulation, but can be None in individual results
            raw_tokens = agent_response.get("total_tokens")
            if isinstance(raw_tokens, int):
                case_result["total_tokens"] = raw_tokens
                total_tokens_used += raw_tokens
            else:
                case_result["total_tokens"] = None # Log as null if not a valid int
                # Optionally, log a warning if raw_tokens was something unexpected but not None
                if raw_tokens is not None:
                    logger.warning(f"Case {case['id']}: Received non-integer total_tokens: {raw_tokens} (type: {type(raw_tokens)}). Treating as 0 for aggregation.")
            
            total_latency_ms += case_result["latency_ms"]

            if case_result["generated_sql"]:
                sql_generated_count += 1
                logger.info(f"Generated SQL: {case_result['generated_sql']}")
                
                # Check against expected pattern if provided
                if case["expected_sql_pattern"]:
                    if re.search(case["expected_sql_pattern"], case_result["generated_sql"], re.IGNORECASE):
                        case_result["pattern_match_success"] = True
                        pattern_matched_count +=1
                        logger.info(f"SUCCESS: Generated SQL matches expected pattern.")
                    else:
                        case_result["pattern_match_success"] = False
                        logger.warning(f"FAILURE: Generated SQL does not match expected pattern: {case['expected_sql_pattern']}")
                else:
                    case_result["pattern_match_success"] = None # No pattern to check against
                    if case.get("expect_failure_or_rejection"): # e.g. DELETE query
                         # Further validation could involve checking db.py logs or specific error messages
                        logger.info("No specific SQL pattern expected for this case (likely expecting rejection).")


            elif case_result["error"]:
                logger.error(f"Agent error: {case_result['error']}")
                if case.get("expect_failure_or_rejection") and \
                   ("query does not conform to sql allowlist" in case_result["error"].lower() or \
                    "must be select, with" in case_result["error"].lower()):
                    logger.info("SUCCESS: Agent correctly rejected/failed disallowed query.")
                    failed_due_to_rejection += 1 
                    # We count this as a "success" for the purpose of this specific test case type
                    # It means the safety mechanism worked.
                else:
                    logger.warning(f"Agent failed to generate SQL and it was not an expected rejection.")
            else:
                logger.warning("Agent did not generate SQL and reported no error.")


        except Exception as e:
            case_result["error"] = str(e)
            logger.error(f"CRITICAL ERROR processing case '{case['id']}': {e}", exc_info=True)
        
        results.append(case_result)
        logger.info(f"--- Case {case['id']} Complete ---")
        if i < total_tests -1:
            time.sleep(1) # Small delay if needed

    logger.info("--- Evaluation Summary ---")
    logger.info(f"Total test cases run: {total_tests}")
    logger.info(f"Successfully generated SQL for: {sql_generated_count}/{total_tests} cases")
    
    # Adjust success count for cases that are expected to fail (e.g. disallowed queries)
    # if they indeed failed as expected.
    effective_successes_for_accuracy = pattern_matched_count + failed_due_to_rejection
    # The denominator for accuracy should be cases where a pattern was expected OR a rejection was expected
    accuracy_relevant_cases = sum(1 for case in EVALUATION_CASES if case["expected_sql_pattern"] or case.get("expect_failure_or_rejection"))


    nlq_to_sql_accuracy = (effective_successes_for_accuracy / accuracy_relevant_cases) * 100 if accuracy_relevant_cases > 0 else 0.0
    logger.info(f"NLQ-to-SQL Accuracy (based on pattern matching or expected rejection): {nlq_to_sql_accuracy:.2f}% ({effective_successes_for_accuracy}/{accuracy_relevant_cases})")
    
    avg_latency_ms = (total_latency_ms / total_tests) if total_tests > 0 else 0
    logger.info(f"Average Latency per query: {avg_latency_ms:.0f} ms")

    avg_tokens_used = (total_tokens_used / total_tests) if total_tests > 0 else 0
    logger.info(f"Average Total Tokens per query: {avg_tokens_used:.0f}")

    # Detailed results logging
    logger.info("Detailed results per case:")
    for res_item in results:
        logger.info(json.dumps(res_item, indent=2))
    
    # Project specific: block_merge_on_failure: true
    # agent_evaluation.threshold.nlq_to_sql_accuracy: 0.85
    threshold_accuracy = 0.85 * 100 
    if nlq_to_sql_accuracy < threshold_accuracy:
        logger.error(f"EVALUATION FAILED: NLQ-to-SQL Accuracy ({nlq_to_sql_accuracy:.2f}%) is below threshold ({threshold_accuracy:.2f}%).")
        return False
    else:
        logger.info(f"EVALUATION PASSED: NLQ-to-SQL Accuracy ({nlq_to_sql_accuracy:.2f}%) meets or exceeds threshold ({threshold_accuracy:.2f}%).")
        return True


if __name__ == "__main__":
    # Ensure the /tests/eval directory exists or create it
    eval_dir = os.path.dirname(__file__)
    if not os.path.exists(eval_dir) and eval_dir: # Make sure eval_dir is not empty
        try:
            os.makedirs(eval_dir)
            logger.info(f"Created directory: {eval_dir}")
        except OSError as e:
            logger.error(f"Failed to create directory {eval_dir}: {e}. Proceeding, but this might be an issue.")
            
    evaluation_successful = run_evaluation_suite()
    logger.info(f"--- NLQ Agent Evaluation Suite Finished ---")

    if not evaluation_successful:
        sys.exit(1) # Exit with error code if evaluation failed threshold
    else:
        sys.exit(0) 