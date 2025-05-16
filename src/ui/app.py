"""Main Streamlit application for the Spatial AI Agent."""

import streamlit as st
import os
import sys
from typing import List
import dspy

# Adjust sys.path to ensure modules in 'src' can be found
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- Global DSPy Configuration (run once at app startup) ---
try:
    print("Attempting to configure global DSPy settings (once)...")
    # Minimal global configuration. Agents will create and use their own LM instances explicitly.
    dspy.settings.configure(
        model="gpt-4o-mini",  # A default model, though agents specify theirs
        max_tokens=1024,  # Default max_tokens for the dspy.settings context
    )
    print("SUCCESS: Global DSPy settings configured with default model name.")
except Exception as e:
    if (
        "already configured" in str(e).lower()
        or "can only be changed by the thread" in str(e).lower()
    ):
        print(
            f"INFO: Global DSPy settings likely already configured on a previous run/thread: {e}"
        )
    else:
        critical_error_msg = (
            f"CRITICAL ERROR: Failed to configure global DSPy LM settings: {e}"
        )
        print(critical_error_msg)
# --- End Global DSPy Configuration ---

try:
    from src.agents.nlq_agent import BasicNLQAgent
    from src.tools.schema_utils import load_schemas
except ImportError as e:
    st.error(
        f"""
    **FATAL ERROR: Failed to import agent or utility modules.**
    This usually means the application can't find the `src.agents.nlq_agent` or `src.tools.schema_utils` module.
    Please ensure you are running Streamlit from the project root directory:
    ```bash
    cd /path/to/your/spatial_ai_agent 
    streamlit run src/ui/app.py
    ```
    **Details:** {e}
    Current sys.path: {sys.path}
    PROJECT_ROOT attempted: {PROJECT_ROOT}
    """
    )
    st.stop()


# --- Agent Initialization ---
@st.cache_resource
def get_nlq_agent_cached():
    """Initializes and caches the BasicNLQAgent (for answering questions)."""
    if not os.getenv("OPENAI_API_KEY"):
        st.error(
            "FATAL ERROR: OPENAI_API_KEY environment variable not set. Please set it in your .env file located in the project root."
        )
        return None
    try:
        agent_instance = BasicNLQAgent()
        if not agent_instance.llm:
            st.error(
                "NLQ Agent LLM (gpt-4o-mini) could not be configured. Check console logs from agent init."
            )
            return None
        return agent_instance
    except Exception as e:
        st.error(f"Error initializing BasicNLQAgent: {e}")
        return None


# --- Load Schemas and Generate Sample Questions (Cached) ---
@st.cache_data  # Cache the generated questions to avoid re-generating on every interaction
def load_and_generate_sample_questions(table_list: List[str], num_samples: int = 3):
    """Loads schemas and generates a few sample questions."""
    qgen_agent = get_qgen_agent_cached()
    if not qgen_agent:
        return [], "Error: Question generation agent not available."

    schema_context = load_schemas(table_list)
    if (
        "Error loading schema" in schema_context
        or "Schema file not found" in schema_context
        or not schema_context.strip()
    ):
        return (
            [],
            f"Warning: Could not load all schemas for question generation. Context: {schema_context[:200]}",
        )

    try:
        sample_questions = qgen_agent.forward(
            schema_context=schema_context, num_questions=num_samples
        )
        return sample_questions, None  # Return None for error message if successful
    except Exception as e:
        return [], f"Error generating sample questions: {e}"


# --- Main UI ---
def main():
    st.set_page_config(page_title="Spatial AI Agent", layout="wide")
    st.title("🤖 Spatial AI Agent 🛰️")
    st.write(
        "Ask questions about your spatial data. Currently configured with test VIIRS fire events, ACLED conflict data, and agent logs."
    )

    nlq_agent = get_nlq_agent_cached()
    if not nlq_agent:
        st.warning(
            "NLQ Query Agent could not be initialized. Application cannot process questions."
        )
        # No hard return, allow app to load to show errors.

    default_table_schemas_to_load = [
        "nlq_agent_log",
        "test_viirs_fire_events",
        "test_acled_conflict_events",
    ]

    try:
        current_schema_context_string = load_schemas(default_table_schemas_to_load)
        if (
            "Schema file not found" in current_schema_context_string
            or "Error loading schema" in current_schema_context_string
        ):
            st.warning(
                f"NLQ Agent Context: Issues loading some table schemas. Context preview: {current_schema_context_string[:200]}..."
            )
        elif (
            "No specific table schemas provided" in current_schema_context_string
            and any(default_table_schemas_to_load)
        ):
            st.warning("NLQ Agent Context: No table schemas were loaded.")
    except Exception as e:
        st.error(f"NLQ Agent Context: Critical error loading schemas: {e}")
        current_schema_context_string = "Error loading schemas for NLQ agent."

    with st.form("nlq_form"):
        if "nl_question_text_area" not in st.session_state:
            st.session_state.nl_question_text_area = ""

        nl_question = st.text_area(
            "Enter your natural language question:",
            value=st.session_state.nl_question_text_area,
            height=100,
            key="nl_question_text_area_input",
            placeholder="e.g., How many fire events were there last week? Show me conflict events in 'Sample Province'.",
        )
        submitted = st.form_submit_button("Ask Agent")

    if submitted and nl_question:
        if not nlq_agent:
            st.error("NLQ Query Agent is not available. Cannot process the question.")
            return

        with st.spinner("Agent is thinking... 🧠 Please wait."):
            try:
                response = nlq_agent.forward(
                    natural_language_query=nl_question,
                    schema_context=current_schema_context_string,
                )
            except Exception as e:
                st.error(
                    f"An error occurred while the agent was processing your request: {e}"
                )
                response = None

        if response:
            st.markdown("---")
            st.subheader("Agent's Interpretation & Actions:")
            col_q, col_sql = st.columns(2)
            with col_q:
                st.markdown(f"**Your Question:**")
                st.markdown(f"> {response.get('natural_language_query', 'N/A')}")
            with col_sql:
                generated_sql = response.get("generated_sql")
                if generated_sql:
                    st.markdown(f"**Generated SQL:**")
                    st.code(generated_sql, language="sql")
                else:
                    st.markdown("**Generated SQL:** `None or error during generation.`")
            st.markdown("---")
            st.subheader("Agent's Response & Data:")
            processed_output = response.get("processed_output", "No processed output.")
            error_msg = response.get("error")
            if error_msg:
                st.error(f"**Agent Error Message:** {error_msg}")
            st.markdown(f"**Processed Output:**")
            st.markdown(f"{processed_output}")
            execution_result = response.get("execution_result")
            if execution_result is not None:
                with st.expander("Raw Execution Result (JSON)", expanded=False):
                    st.json(execution_result)
            else:
                with st.expander("Raw Execution Result (JSON)", expanded=False):
                    st.write("No direct execution result available.")
            st.markdown("---")
            st.subheader("Interaction Details (for debugging):")
            details_cols = st.columns(3)
            details_cols[0].metric(
                label="LLM Model Used", value=str(response.get("llm_model_used", "N/A"))
            )
            details_cols[0].metric(
                label="Agent Version",
                value=(
                    str(nlq_agent.agent_version)
                    if hasattr(nlq_agent, "agent_version")
                    else "N/A"
                ),
            )
            details_cols[1].metric(
                label="Total Latency (ms)", value=str(response.get("latency_ms", "N/A"))
            )
            details_cols[1].metric(
                label="Session ID", value=str(response.get("session_id", "N/A"))
            )
            with details_cols[2]:
                st.text("Token Usage:")
                st.caption(
                    f"Prompt: {response.get('prompt_tokens', 'N/A')}\nCompletion: {response.get('completion_tokens', 'N/A')}\nTotal: {response.get('total_tokens', 'N/A')}"
                )
        elif submitted:
            st.error(
                "Agent did not return a response. Check for errors above or in the console."
            )
    elif submitted and not nl_question:
        st.warning("Please enter a question before submitting.")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    main()
