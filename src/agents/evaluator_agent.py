import dspy
from typing import Optional

# --- DSPy Signatures ---
class EvaluateNLQInteraction(dspy.Signature):
    """
    Evaluates the quality of an NLQ-to-SQL interaction.
    Your goal is to provide a success score (1-5, where 5 is best) and a constructive critique.
    Consider the accuracy of the SQL, the relevance of the data to the question,
    and the completeness/correctness of the final answer provided to the user.
    """
    natural_language_query = dspy.InputField(desc="The user's original question.")
    schema_context = dspy.InputField(desc="The database schema context provided to the NLQ agent.")
    generated_sql = dspy.InputField(desc="The SQL query produced by the NLQ agent.")
    sql_query_result_summary = dspy.InputField(desc="A summary or sample of the data returned by the SQL query.")
    final_answer_to_user = dspy.InputField(desc="The natural language answer/summary provided to the user.")

    overall_success_score = dspy.OutputField(desc="An integer score from 1 (very poor) to 5 (excellent) indicating the overall success of the interaction.")
    critique_text = dspy.OutputField(desc="A textual explanation for the score, highlighting good points or areas for improvement. Be specific.")
    # Optional: suggested_correction_sql = dspy.OutputField(desc="If the SQL was flawed, a corrected version. Otherwise, N/A.")

class InteractionEvaluator(dspy.Module):
    def __init__(self, evaluation_model_name: str = "gpt-3.5-turbo", max_tokens: int = 500):
        super().__init__()
        self.evaluator_llm_name = evaluation_model_name # Should be just "gpt-3.5-turbo" if using openai prefix in dspy.LM
        self.evaluator_llm = None
        self.evaluate_interaction = None
        try:
            # Construct the full model string for dspy.LM
            # Ensure "openai/" prefix is not duplicated and is present
            clean_model_name = self.evaluator_llm_name.replace('openai/', '')
            full_model_identifier = f"openai/{clean_model_name}"
            
            print(f"[InteractionEvaluator.__init__] Attempting to initialize LLM with identifier: {full_model_identifier}")
            self.evaluator_llm = dspy.LM(full_model_identifier, max_tokens=max_tokens)
            print(f"[InteractionEvaluator.__init__] Successfully initialized self.evaluator_llm (type: {type(self.evaluator_llm)}). Model: {getattr(self.evaluator_llm, 'model', 'N/A')}")

            self.evaluate_interaction = dspy.ChainOfThought(EvaluateNLQInteraction)
            self.evaluate_interaction.lm = self.evaluator_llm # Crucial assignment
            print(f"[InteractionEvaluator.__init__] Assigned self.evaluate_interaction.lm. Effective LM: {self.evaluate_interaction.lm}")

        except Exception as e:
            print(f"[InteractionEvaluator.__init__] CRITICAL ERROR initializing evaluator: {e}")
            import traceback
            print(traceback.format_exc())
            self.evaluator_llm = None
            self.evaluate_interaction = None

    def forward(self,
                natural_language_query: str,
                schema_context: str,
                generated_sql: str,
                sql_query_result_summary: str,
                final_answer_to_user: str):
        if not self.evaluate_interaction:
            print("[InteractionEvaluator.forward] ERROR: Evaluator module not initialized.")
            return {"overall_success_score": 0, "critique_text": "Evaluator module not initialized."}

        print(f"[InteractionEvaluator.forward] DEBUG: self.evaluate_interaction.lm IS: {self.evaluate_interaction.lm} (Model: {getattr(self.evaluate_interaction.lm, 'model', 'N/A') if self.evaluate_interaction.lm else 'N/A'})")
        print(f"[InteractionEvaluator.forward] DEBUG: dspy.settings.lm IS: {dspy.settings.lm} (Model: {getattr(dspy.settings.lm, 'model', 'N/A') if dspy.settings.lm else 'N/A'})")

        try:
            print(f"[InteractionEvaluator.forward] Evaluating interaction for NLQ: '{natural_language_query[:50]}...'")
            evaluation = self.evaluate_interaction(
                natural_language_query=natural_language_query,
                schema_context=schema_context,
                generated_sql=generated_sql,
                sql_query_result_summary=sql_query_result_summary,
                final_answer_to_user=final_answer_to_user
            )
            print(f"[InteractionEvaluator.forward] Evaluation complete. Score: {evaluation.overall_success_score}")
            return {
                "overall_success_score": int(evaluation.overall_success_score),
                "critique_text": evaluation.critique_text
            }
        except Exception as e:
            print(f"[InteractionEvaluator.forward] ERROR during evaluation: {e}")
            # Attempt to parse the score if it's a common issue like "ValueError: invalid literal for int()"
            # This is a basic workaround; robust error handling would be more involved.
            try:
                if hasattr(e, 'prediction') and hasattr(e.prediction, 'overall_success_score'): # DSPy 0.4+
                    score_str = str(e.prediction.overall_success_score).strip()
                    score = int(score_str.split()[0]) # Try to get first number if LLM includes text
                    return {"overall_success_score": score, "critique_text": f"Error during evaluation, but extracted score. Detail: {e}"}
            except:
                pass # Fall through if score extraction fails
            return {"overall_success_score": 0, "critique_text": f"Error during evaluation: {e}"}

if __name__ == '__main__':
    # This is a basic test and requires OPENAI_API_KEY to be set in the environment
    # and dspy.settings to be configured with a base LLM if not using the local one.
    # For this test, we'll assume OPENAI_API_KEY is set and try to use the local evaluator_llm.

    print("Testing InteractionEvaluator...")
    # Ensure dotenv is loaded if running this file directly for testing
    from dotenv import load_dotenv
    load_dotenv()
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("FATAL: OPENAI_API_KEY not set. Cannot run evaluator test.")
    else:
        # Configure DSPy globally for this test script
        # This is often necessary for standalone DSPy module testing.
        try:
            # Use a specific model for evaluation here for the global setting
            # This will be the LM that evaluate_interaction picks up if its own .lm isn't overriding.
            print(f"[__main__] Attempting to configure global dspy.settings.lm with openai/gpt-3.5-turbo")
            test_eval_llm = dspy.LM("openai/gpt-3.5-turbo", max_tokens=500) # Matching default in InteractionEvaluator for consistency
            dspy.configure(lm=test_eval_llm)
            print(f"[__main__] Configured dspy.settings.lm globally for test: {dspy.settings.lm} (Model: {getattr(dspy.settings.lm, 'model', 'N/A')})")
        except Exception as e:
            print(f"[__main__] CRITICAL ERROR configuring global DSPy settings for test: {e}")
            import traceback
            print(traceback.format_exc())
            # Exit if global DSPy setup fails, as evaluator likely won't work.
            exit()

        evaluator = InteractionEvaluator() # Keep evaluator_model_name default or specify
        # The InteractionEvaluator __init__ will still try to create its own evaluator_llm
        # and assign it to self.evaluate_interaction.lm.
        # We want to see if the explicit assignment inside InteractionEvaluator
        # overrides the global dspy.settings.lm we just set.

        if evaluator.evaluate_interaction and evaluator.evaluator_llm:
            print(f"[__main__] Evaluator's internal LLM (evaluator.evaluator_llm): {evaluator.evaluator_llm} (Model: {getattr(evaluator.evaluator_llm, 'model', 'N/A')})")
            print(f"[__main__] Evaluator's ChainOfThought module's LM (evaluator.evaluate_interaction.lm): {evaluator.evaluate_interaction.lm} (Model: {getattr(evaluator.evaluate_interaction.lm, 'model', 'N/A')})")
            
            test_nlq = "How many fire events were recorded by 'Suomi NPP' satellite last week?"
            test_schema = """
            CREATE TABLE test_viirs_fire_events (
                event_id SERIAL PRIMARY KEY,
                geom GEOMETRY(Point, 4326) NOT NULL,
                detection_timestamp TIMESTAMPTZ NOT NULL,
                brightness_kelvin NUMERIC,
                confidence_percentage NUMERIC,
                satellite_source VARCHAR(50)
            );
            """
            test_sql = "SELECT COUNT(*) FROM test_viirs_fire_events WHERE satellite_source = 'Suomi NPP' AND detection_timestamp >= '2023-10-16' AND detection_timestamp < '2023-10-23';"
            test_result_summary = "Query returned: [{'count': 42}]"
            test_final_answer = "There were 42 fire events recorded by 'Suomi NPP' satellite last week."

            evaluation_result = evaluator.forward(
                natural_language_query=test_nlq,
                schema_context=test_schema,
                generated_sql=test_sql,
                sql_query_result_summary=test_result_summary,
                final_answer_to_user=test_final_answer
            )
            print("\nEvaluation Result:")
            print(f"  Score: {evaluation_result.get('overall_success_score')}")
            print(f"  Critique: {evaluation_result.get('critique_text')}")

            test_nlq_2 = "List all fires."
            test_sql_2 = "SELECT * FROM fires;" # Table "fires" does not exist in schema
            test_result_summary_2 = "Error: table fires not found"
            test_final_answer_2 = "I could not find a table named 'fires'."
            evaluation_result_2 = evaluator.forward(
                natural_language_query=test_nlq_2,
                schema_context=test_schema, # Schema does not contain 'fires'
                generated_sql=test_sql_2,
                sql_query_result_summary=test_result_summary_2,
                final_answer_to_user=test_final_answer_2
            )
            print("\nEvaluation Result 2 (Flawed SQL):")
            print(f"  Score: {evaluation_result_2.get('overall_success_score')}")
            print(f"  Critique: {evaluation_result_2.get('critique_text')}")
        else:
            print("Evaluator module could not be initialized for testing.") 