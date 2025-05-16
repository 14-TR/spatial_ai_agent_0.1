import time
import os
import sys
from dotenv import load_dotenv

# Adjust path to import BasicNLQAgent and other necessary modules
# This assumes the script is in src/testing/ and needs to go up two levels for project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.agents.nlq_agent import BasicNLQAgent
    # If your agent or its dependencies require dspy.settings to be configured globally first:
    # import dspy
    # from src.agents.evaluator_agent import InteractionEvaluator # If you wanted to inspect it separately
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure that the script is run from a location where 'src' is accessible,")
    print("or that PYTHONPATH is set up correctly.")
    sys.exit(1)

# --- Configuration ---
# You can expand this list with up to 100 questions or load them from a file.
TEST_QUESTIONS = [
    # --- VIIRS Table Questions ---
    "How many fires did the 'Suomi NPP' satellite detect?",
    "Tell me when and how hot the top 3 brightest fires were.",
    "Which satellites have reported fire events?",
    "Show me all fires seen since the start of 2023.",
    "What's the average temperature of fires reported by 'NOAA-20'?",
    "Are there any fire events with a confidence below 50 percent?",
    "List the 5 oldest fire detections we have.",
    "What is the maximum brightness recorded for any fire?",
    "Count VIIRS fire events for each satellite source.",
    "Show fire events with brightness above 600 Kelvin and confidence over 80%.",

    # --- ACLED Table Questions ---
    "How many people died in total during 'Protests'?",
    "List the date, type of event, and place for 5 recent conflicts.",
    "Count the number of conflicts that happened in the year 2022.",
    "What are the different types of conflict events recorded in ACLED?",
    "Show conflicts in 'Syria' involving 'Explosions/Remote violence'.",
    "Find the conflict event with the most fatalities recorded.",
    "Which locations have the highest number of reported conflict events?",
    "List all actors involved in 'Battles' in 2023.",
    "What is the sum of fatalities for all events in 'Yemen'?",
    "How many distinct admin1 regions have recorded ACLED events?",

    # --- NLQ Agent Log Table Questions ---
    "Show the original question and the SQL for interactions that the evaluator scored a perfect 5.",
    "What's the typical score the evaluator gives our logged interactions?",
    "List up to 5 questions that caused an error when the agent processed them.",
    "How many interactions have been logged in total by the NLQ agent?",
    "Which LLM model is most frequently used by the agent according to its logs?",
    # "Display logs from sessions where human feedback indicated the agent was incorrect.", # Needs human_feedback fields
    "What was the schema context for the latest logged agent interaction?",
    "Show me log entries with latency over 2000 milliseconds.",
    "How many times has the agent version '0.1.0-dev' been logged?",
    "List natural language queries that resulted in SQL queries longer than 300 characters."
]
QUESTIONS_TO_RUN = TEST_QUESTIONS # Can be a subset for testing: TEST_QUESTIONS[:5]
DELAY_BETWEEN_QUESTIONS_SECONDS = 2 # To be respectful of API rate limits

def run_batch_test():
    print("--- Starting NLQ Agent Batch Test ---")
    print(f"WARNING: This script will make approximately {len(QUESTIONS_TO_RUN) * 2} LLM API calls.")
    print("This will incur costs and take some time. Monitor your API usage.")
    print(f"A delay of {DELAY_BETWEEN_QUESTIONS_SECONDS} seconds will be applied between questions.")
    print("Press Ctrl+C to interrupt the test early if needed.")
    print("---------------------------------------------------\n")

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("FATAL ERROR: OPENAI_API_KEY environment variable not set.")
        print("Please set it in your .env file before running the batch test.")
        return

    print("Initializing BasicNLQAgent...")
    try:
        # If your agent needs global DSPy settings (some older setups might)
        # Example: lm = dspy.OpenAI(model="gpt-3.5-turbo-instruct", max_tokens=150)
        # dspy.settings.configure(lm=lm)
        agent = BasicNLQAgent() # Assumes agent_version is defaulted in __init__
        if not agent.llm or not agent.generate_sql or not agent.evaluator:
             print("ERROR: BasicNLQAgent did not initialize all its components (LLM, SQL generator, or Evaluator).")
             print("Please check the agent's __init__ method and API key.")
             return
    except Exception as e:
        print(f"Error initializing BasicNLQAgent: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("Agent initialized. Starting to process questions...\n")

    successful_interactions = 0
    failed_interactions = 0

    for i, question in enumerate(QUESTIONS_TO_RUN):
        print(f"--- Question {i+1}/{len(QUESTIONS_TO_RUN)} ---")
        print(f"NLQ: {question}")
        
        try:
            # Using agent's default schema loading (all schemas if context is None)
            response = agent.forward(natural_language_query=question)
            
            generated_sql = response.get("generated_sql")
            processed_output = response.get("processed_output")
            error_msg = response.get("error")
            
            # The evaluator's score and critique are now part of the logging within agent.forward,
            # but not directly returned by agent.forward itself.
            # We rely on checking the database log later for scores.
            # For immediate feedback, we could modify agent.forward to return them.
            # For now, we'll just confirm processing.

            print(f"Generated SQL: {generated_sql if generated_sql else 'N/A'}")
            if error_msg:
                print(f"Agent Error: {error_msg}")
                failed_interactions += 1
            else:
                print(f"Processed Output: {str(processed_output)[:200]}...") # Print a snippet
                successful_interactions += 1
            
            # The detailed log, including evaluator score, is in the database.
            print(f"Interaction processed. Check 'nlq_agent_log' table for full details including evaluator score.")

        except Exception as e:
            print(f"CRITICAL ERROR processing question '{question}': {e}")
            import traceback
            traceback.print_exc()
            failed_interactions += 1
        
        print("------------------------------------\n")
        if i < len(QUESTIONS_TO_RUN) - 1: # Don't sleep after the last question
            time.sleep(DELAY_BETWEEN_QUESTIONS_SECONDS)

    print("--- Batch Test Complete ---")
    print(f"Total questions processed: {len(QUESTIONS_TO_RUN)}")
    print(f"Successful interactions (agent processed without critical error): {successful_interactions}")
    print(f"Failed interactions (agent raised critical error during processing): {failed_interactions}")
    print("Review the 'nlq_agent_log' table for detailed outcomes and evaluator scores.")

if __name__ == "__main__":
    # Create the testing directory if it doesn't exist
    testing_dir = os.path.dirname(__file__)
    if not os.path.exists(testing_dir) and testing_dir: # Ensure testing_dir is not empty
        try:
            os.makedirs(testing_dir)
            print(f"Created directory: {testing_dir}")
        except OSError as e:
            print(f"Failed to create directory {testing_dir}: {e}")
            # Decide if script should exit or try to continue
            # For now, it will try to continue, but this might be an issue if it's essential
            
    run_batch_test() 