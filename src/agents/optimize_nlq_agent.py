import dspy
import os
from dotenv import load_dotenv
import psycopg # Added for database connection
from psycopg.rows import dict_row # To get results as dictionaries

# Assuming NLQToSQL signature and load_schemas are in nlq_agent.py and schema_utils.py respectively
# Adjust imports based on your final project structure and how you run this script
try:
    from .nlq_agent import NLQToSQL # If running as part of a package
    from ..tools.schema_utils import load_schemas
    from ..tools.db import DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT # Import DB constants
except ImportError:
    # Fallback for running script directly from src/agents/ or project root with PYTHONPATH set
    import sys
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.agents.nlq_agent import NLQToSQL
    from src.tools.schema_utils import load_schemas
    from src.tools.db import DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT


# --- Constants for fetching training data ---
MIN_SCORE_FOR_TRAINING = 4  # Corrected: Minimum evaluator score (1-5 scale) to consider an example "good"
MAX_EXAMPLES_FROM_LOG = 20 # Max number of examples to fetch from the log


def sql_exact_match_metric(gold, pred, trace=None):
    """Checks if the predicted SQL query exactly matches the gold SQL query."""
    # gold is a dspy.Example, pred is typically a dspy.Prediction object or a dict
    # Accessing the predicted sql_query field from the prediction
    predicted_sql = pred.sql_query if hasattr(pred, 'sql_query') else ""
    return gold.sql_query.strip() == predicted_sql.strip()

def validate_signature_Ávila_metric(gold, pred, trace=None):
    """A more comprehensive validation metric could be added later."""
    # For now, just use exact match.
    # Future: check for SQL syntax errors, presence of key clauses, etc.
    return sql_exact_match_metric(gold, pred, trace)

def fetch_training_examples_from_db() -> list[dspy.Example]:
    """
    Fetches "good" training examples from the nlq_agent_log table.
    """
    conn_string = f"dbname='{DB_NAME}' user='{DB_USER}' password='{DB_PASSWORD}' host='{DB_HOST}' port='{DB_PORT}'"
    examples = []
    
    # SQL to fetch good examples:
    # - Must have a good evaluator score
    # - Must have a generated SQL query (this will be our "gold" SQL)
    # - Must have the schema context that was used
    # - Order by timestamp to get recent examples if we limit
    query = f"""
        SELECT 
            natural_language_query, 
            generated_sql_query, 
            schema_context_used
        FROM 
            nlq_agent_log
        WHERE 
            evaluator_overall_success_score >= %s
            AND generated_sql_query IS NOT NULL 
            AND TRIM(generated_sql_query) != ''
            AND schema_context_used IS NOT NULL
            AND TRIM(schema_context_used) != ''
        ORDER BY 
            timestamp DESC
        LIMIT %s;
    """
    
    try:
        with psycopg.connect(conn_string, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(query, (MIN_SCORE_FOR_TRAINING, MAX_EXAMPLES_FROM_LOG))
                log_entries = cur.fetchall()
                
                for entry in log_entries:
                    if entry['natural_language_query'] and entry['generated_sql_query'] and entry['schema_context_used']:
                        # We are assuming generated_sql_query is the "gold" standard if score is high
                        example = dspy.Example(
                            nl_question=entry['natural_language_query'],
                            context=entry['schema_context_used'], 
                            sql_query=entry['generated_sql_query'] 
                        ).with_inputs("nl_question", "context")
                        examples.append(example)
                print(f"Fetched {len(examples)} training examples from the database.")
    except psycopg.Error as e:
        print(f"Database error while fetching training examples: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while fetching training examples: {e}")
        
    return examples

def get_static_fallback_examples() -> list[dspy.Example]:
    """
    Returns a small list of hardcoded fallback examples if DB fetching fails or yields too few.
    """
    print("Using static fallback training examples.")
    # Re-using schemas loaded globally for these static examples
    # This assumes viirs_schema, acled_schema, all_schemas_context are available if called from main_optimizer
    # For robustness, this function could load them if they aren't passed or found globally.
    
    # Attempt to load schemas here as well, in case this function is called standalone or schemas aren't global
    try:
        current_viirs_schema = load_schemas(["test_viirs_fire_events"])
        current_acled_schema = load_schemas(["test_acled_conflict_events"])
        current_all_schemas_context = load_schemas([])
    except Exception as e:
        print(f"Error loading schemas for static examples: {e}. Static examples might be incomplete.")
        current_viirs_schema = "CREATE TABLE test_viirs_fire_events (brightness_kelvin FLOAT, detection_timestamp TIMESTAMP);" # Minimal
        current_acled_schema = "CREATE TABLE test_acled_conflict_events (event_date DATE, event_type TEXT, fatalities INT);" # Minimal
        current_all_schemas_context = ""

    return [
        dspy.Example(
            nl_question="What is the average brightness of the 5 most recent fires?",
            context=current_viirs_schema,
            sql_query="""WITH MostRecentFires AS (
    SELECT brightness_kelvin
    FROM test_viirs_fire_events
    ORDER BY detection_timestamp DESC
    LIMIT 5
)
SELECT AVG(brightness_kelvin)
FROM MostRecentFires;"""
        ).with_inputs("nl_question", "context"),
        dspy.Example(
            nl_question="How many fire events are in the VIIRS table?",
            context=current_viirs_schema,
            sql_query="SELECT COUNT(*) FROM test_viirs_fire_events;"
        ).with_inputs("nl_question", "context"),
    ]


def main_optimizer():
    """
    Main function to configure DSPy, define training data,
    run the optimizer, and save the compiled module.
    """
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please ensure your .env file is correctly set up with OPENAI_API_KEY.")
        return

    # 1. Configure DSPy LM
    # Ensure this model name matches what you use in BasicNLQAgent
    llm_model_name = "gpt-4o-mini"
    # turbo = dspy.OpenAI(model=llm_model_name, max_tokens=300) # Old way
    turbo = dspy.LM(f"openai/{llm_model_name}", max_tokens=300) # Corrected line
    dspy.settings.configure(lm=turbo)
    print(f"DSPy LM configured with: openai/{llm_model_name}")

    optimized_predictor_path = "compiled_nlq_predictor.json" # Define path early

    # 2. Load Schemas for Training Data Context
    # We need the actual schema text that would be provided to the agent for these examples.
    # For simplicity, let's assume load_schemas([]) loads all relevant schemas.
    # Or, specify the exact tables needed for each example.
    
    viirs_schema = load_schemas(["test_viirs_fire_events"])
    acled_schema = load_schemas(["test_acled_conflict_events"])
    all_schemas_context = load_schemas([]) # For generic questions or if specific tables aren't known

    # 3. Define Training Data (dspy.Example)
    # Each example needs: nl_question, sql_query (gold standard), and context (schema)
    
    # --- Dynamically Fetch Training Data ---
    train_examples = fetch_training_examples_from_db()

    if not train_examples or len(train_examples) < 2: # Need at least a couple for BootstrapFewShot
        print(f"Not enough examples fetched from DB ({len(train_examples)}). Augmenting with/falling back to static examples.")
        static_examples = get_static_fallback_examples()
        # Combine or replace. For now, let's add static if DB examples are too few.
        # A more sophisticated strategy could be used (e.g. ensure no duplicates by nl_question)
        train_examples.extend(static_examples) 
        # Remove duplicates based on nl_question if any - simple approach
        seen_questions = set()
        unique_train_examples = []
        for ex in train_examples:
            if ex.nl_question not in seen_questions:
                unique_train_examples.append(ex)
                seen_questions.add(ex.nl_question)
        train_examples = unique_train_examples

    if not train_examples:
        print("CRITICAL: No training examples available (neither from DB nor static fallbacks). Aborting optimization.")
        return
        
    print(f"Using a total of {len(train_examples)} training examples for optimization.")

    # 4. Initialize the DSPy Module to be Optimized
    # We are optimizing the 'generate_sql' part of the agent, which uses NLQToSQL signature.
    # We'll create a simple dspy.Predict module for the NLQToSQL signature.
    student_predictor = dspy.Predict(NLQToSQL)

    # 5. Configure and Run the Optimizer
    # We'll use BootstrapFewShot, which creates few-shot prompts.
    # The metric function will guide the selection of these few-shot examples.
    config = dict(max_bootstrapped_demos=min(4, len(train_examples)), max_labeled_demos=min(4, len(train_examples))) 
    # Ensure max_bootstrapped_demos and max_labeled_demos don't exceed available examples.
    # DSPy's BootstrapFewShot might require at least 1 demo for bootstrapping.
    if len(train_examples) == 0:
        print("No training examples available. Skipping BootstrapFewShot compilation.")
        # Save the unoptimized predictor? Or just exit?
        # For now, let's skip saving if no optimization happened.
        return 
    
    # Adjust config if train_examples is very small (e.g., 1) as BootstrapFewShot might need more.
    # BootstrapFewShot typically needs at least 2 examples for meaningful bootstrapping if it samples for demos.
    # If only 1 example, it might not be able to create a few-shot prompt.
    if len(train_examples) < 2 and (config['max_bootstrapped_demos'] > 0 or config['max_labeled_demos'] > 0) :
        print("Warning: Very few training examples. BootstrapFewShot might not be effective or might error.")
        # Decide: either skip, or try with what's available, or adjust config.
        # For now, we'll let it try, but it's a point of attention.
        # config['max_bootstrapped_demos'] = min(config['max_bootstrapped_demos'], len(train_examples))
        # config['max_labeled_demos'] = min(config['max_labeled_demos'], len(train_examples))


    optimizer = dspy.BootstrapFewShot(metric=validate_signature_Ávila_metric, **config)
    
    print("Compiling (optimizing) the predictor...")
    try:
        optimized_predictor = optimizer.compile(student=student_predictor, trainset=train_examples)
        print("Compilation complete.")

        optimized_predictor.save(optimized_predictor_path)
        print(f"Optimized predictor saved to {optimized_predictor_path}")
    except IndexError as ie: # Common if trainset is too small for chosen demos
        print(f"ERROR during compilation, possibly due to too few training examples for BootstrapFewShot: {ie}")
        print("Consider adding more diverse examples to your nlq_agent_log with good scores, or to static fallbacks.")
    except Exception as e:
        print(f"An unexpected error occurred during compilation: {e}")


    print("\nTo use the optimized predictor in your BasicNLQAgent:")
    print("1. In BasicNLQAgent.__init__, instead of `self.generate_sql = dspy.Predict(NLQToSQL)`," )
    print("   you would initialize it and then load the saved state:")
    print("   `self.generate_sql = dspy.Predict(NLQToSQL) # Or your base module`")
    print(f"   `self.generate_sql.load(\'{optimized_predictor_path}\')`")
    print("Ensure the .json file is accessible by the agent (e.g., in the same directory or correct path specified).")

if __name__ == "__main__":
    main_optimizer() 