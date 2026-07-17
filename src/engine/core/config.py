import os
import json
from dotenv import load_dotenv

# Load environment variables from the root .env file
load_dotenv()

# =================================================================
# 🌐 CORE NETWORK GATEWAYS
# =================================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# =================================================================
# 🧠 COMPONENT DEFINITIONS (Models + Prompts + Formatters)
# =================================================================
class Architect:
    """High-level structural planning & task checklists (Enforces MD Contracts)"""
    MODEL = "openai/gpt-oss-120b"
    ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

    SYSTEM_PROMPT = """
    You are a Principal Data Science Architect. Your job is to translate statistical metadata and user objectives into a rigorous, production-grade machine learning blueprint saved as a markdown contract file.
    You DO NOT write raw python code. You DO NOT write conversational filler. 
    Every section must contain actionable constraints for an execution agent.

    SECURITY & RELATIVE PATH PATHING RULES:
    Assume the execution environment is a strictly jailed workspace directory.
    All file reads, writes, and model persistence steps MUST use relative paths (e.g., 'data/active/copy_test_data.csv' or 'artifacts/best_model.joblib'). Do not use absolute paths or attempt directory traversal.
    """

    @staticmethod
    def get_user_prompt(profile_data: dict, user_objective: str) -> str:
        # Dynamically isolate likely target fields from metadata if present
        return f"""
        🎯 USER PROBLEM OBJECTIVE:
        {user_objective}

        📊 DATASET PROFILE SPECTRA:
        {json.dumps(profile_data, indent=2)}

        Design an end-to-end data science execution contract blueprint. You must output the entire response in clean Markdown formatting exactly matching the schema template below.

        CRITICAL PIPELINE ARCHITECTURE INSTRUCTIONS:
        1. Target Variable Isolation: Explicitly enforce that the target column (e.g., 'CUSTOMER_SEGMENT') must be parsed out into a target vector 'y' and completely stripped from the feature matrix 'X' BEFORE fitting any transformers or pipelines to prevent 1.0 F1 score data leakage.
        2. Categorical Targets: Explicitly mandate using `sklearn.preprocessing.LabelEncoder` on the target string vector to ensure full compatibility with multiclass algorithms like XGBoost.
        3. Strict Validation Architecture: Mandate a clean hold-out partition (e.g., 20%) isolated at the very start of the modeling stage, entirely separate from any Stratified K-Fold cross-validation loops.
        4. Model Diversity: Force a comparative evaluation using standard parameters across a baseline Logistic Regression, Random Forest, and a multiclass-configured XGBClassifier (without binary pos_weight parameters).

        OUTPUT STRUCTURE LAYOUT REQUIREMENTS:
        Your response must start immediately with the title and contain these exact structural headers:
        
        # 🚀 Data Science Pipeline Blueprint
        ## 🎯 1. User Objective
        [Insert clean analytical translation of the user's objective here]

        ## 📋 2. Core System Requirements & AI Guardrails
        - **Target Isolation Constraint:** [Explicit target separation instruction]
        - **Target Encoding Requirements:** [Explicit LabelEncoder instructions]
        - **Validation Strategy Contract:** [Strict holdout partition and Stratified K-Fold requirements]
        - **Pathing Boundaries:** All active dataset adjustments must target 'data/active/copy_test_data.csv' and all serializations must target the 'artifacts/' folder.

        ## 🛠️ 3. Execution Engine Roadmap
        For every step in the pipeline roadmap, use an H3 header structure matching:
        ### Step 1: [Step Name]
        - **Action:** [Provide 2-3 detailed technical execution bullet points outlining data transformations, drops, or training configurations]
        - **Contract:** Input: [relative input path] | Output: [relative output path or artifact path]

        ## 📦 4. Python Package Requirements
        Analyze the pipeline you just created. List ONLY the third-party pip-installable packages required to execute the code (e.g., pandas, scikit-learn). 
        - Do NOT list standard Python libraries (like os, json, math).
        - Format them as a strict bulleted list of just the package names (e.g., `- scikit-learn`).

        Make sure the step headings use exactly '### Step X:' so the system parsing engine can locate them. Limit the execution plan to 5-8 highly comprehensive steps. Do not append conversational text before or after the markdown content.
        """


class Coder:
    """Autonomous Python script generator for Phase 2 Execution."""
    # llama-3.3-70b-versatile was deprecated by Groq on 2026-06-17 (free/dev tiers).
    # gpt-oss-120b is the recommended migration and already powers the Architect.
    # MODEL = os.getenv("CODER_MODEL", "llama-3.3-70b-versatile")
    MODEL = "qwen2.5:7b-instruct"
    # ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
    ENDPOINT = "http://localhost:11434/v1/chat/completions"
    TEMPERATURE = 0.1
    TEMPERATURE = 0.1
    MAX_TOKENS = 4096
    TIMEOUT_SECONDS = 120

    SYSTEM_PROMPT = """
    You are an expert Data Science AI Programmer operating within a sandboxed workspace.
    CRITICAL RULES:
    1. OUTPUT ONLY THE PURE PYTHON CODE BLOCK. No markdown wraps (unless specified), no conversation.
    2. Ensure the code is strictly non-interactive (save plots using plt.savefig(), never plt.show()).
    3. All data reads/writes must use relative paths targeting 'data/active/copy_test_data.csv' or the 'artifacts/' folder.
    4. PROTECT CONTEXT TOKENS: Never print entire DataFrames or arrays. Only print df.shape, df.head(2), or specific metrics.
    5. NEVER SWALLOW ERRORS: Do NOT wrap your imports or file loading in `try...except` blocks. Let the script crash natively with full tracebacks so the system orchestrator can catch them.
    6. STATE HANDOFF: If (and only if) you overwrite the active CSV, the LAST line of the script must print the new state exactly like: print(f"STATE shape={df.shape} columns={list(df.columns)}")
    """

    @staticmethod
    def get_user_prompt(step_title: str, step_details: str,
                        full_plan: str, summary_txt: str, memory_str: str,
                        live_state: str = "(live probe unavailable)") -> str:
        return f"""
        ========================================================================
        📊 LIVE DATA STATE (ground truth, probed from disk RIGHT NOW):
        {live_state}

        THIS SECTION IS AUTHORITATIVE. These are the exact files, columns, and
        dtypes that exist at this moment. Use ONLY these column names.
        ========================================================================

        🧠 SANDBOX MEMORY (console output from previous steps):
        {memory_str}
        ========================================================================

        📈 ORIGINAL DATASET PROFILE (statistics from the RAW data at ingestion;
        column layout may have changed since — trust LIVE DATA STATE for schema):
        {summary_txt}
        ========================================================================

        📋 PIPELINE MARKDOWN CONTRACT (Your master instructions):
        {full_plan}

        ------------------------------------------------------------------------
        🎯 CURRENT FOCUS TASK:
        Write the complete Python script for this step ONLY:
        ### {step_title}
        {step_details}
        ------------------------------------------------------------------------

        IMPLEMENTATION INSTRUCTIONS:
        - Output ONLY pure python code inside a single ```python block.
        - Column names MUST come from LIVE DATA STATE. Never invent or assume columns.
        - Use the ORIGINAL PROFILE only for statistics (skew, ranges), never for schema.
        - Read the PIPELINE MARKDOWN CONTRACT to ensure you do not violate the Target Isolation or Validation Boundaries.
        """

    @staticmethod
    def get_fix_prompt(broken_code: str, traceback: str,
                       step_details: str = "", live_state: str = "") -> str:
        return f"""
        The previously generated Python script crashed during sandboxed execution.

        📊 LIVE DATA STATE (ground truth, probed from disk AFTER the crash —
        a crashed script may have already mutated files, so trust this, not the code):
        {live_state}

        🎯 THE STEP THIS SCRIPT MUST ACCOMPLISH:
        {step_details}

        [BROKEN SCRIPT]:
        ```python
        {broken_code}
        ```

        [INTERCEPTED RUNTIME TRACEBACK]:
        {traceback}

        Analyze the traceback, find the root cause, and rewrite the ENTIRE corrected
        script. Column names MUST come from LIVE DATA STATE — if the traceback is a
        KeyError, the fix is almost always using a column that actually exists there.
        Preserve the step's intent and its input/output file contracts.
        Output ONLY the fixed script inside a single ```python block.
        """

class Analyst:
    """Senior executive-grade summary reporting (Phase 3 final call)."""
    # _override = os.getenv("ANALYST_MODEL", "").strip()
    # if _override.startswith(("gsk_", "sk-", "key", "Bearer")) or " " in _override:
    #     print("[config] WARNING: ANALYST_MODEL looks like a credential — ignoring.")
    #     _override = ""
    # MODEL = _override or "openai/gpt-oss-120b"
    # del _override
    MODEL = os.getenv("CODER_MODEL", "llama-3.3-70b-versatile")
    ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
 
    SYSTEM_PROMPT = """
    You are a Senior Data Science Lead Analyst writing the final report of an
    automated pipeline run for both technical and executive readers.
 
    NON-NEGOTIABLE RULES:
    1. EVERY number in your report (F1, accuracy, shapes, row counts) must come
       verbatim from the evidence provided. NEVER invent, estimate, or round
       beyond what the logs show. If a value is absent, write "not recorded".
    2. If any step shows FAILED status or suspicious output, call it out
       honestly in a Limitations section — do not paper over it.
    3. Output clean Markdown only. No conversational preamble or sign-off chatter.
    """
 
    @staticmethod
    def get_report_prompt(context: str) -> str:
        return f"""
        Below is the complete evidence package from an autonomous data science
        pipeline run. Synthesize the final report.
 
        {context}
 
        REQUIRED REPORT STRUCTURE (Markdown):
        # <Concise title naming the task and dataset>
        ## Executive Summary            (3-5 sentences: goal, approach, headline result)
        ## Dataset Overview             (ground it in the provided dataset overview)
        ## Methodology                  (walk the executed steps; note repairs/attempts)
        ## Model Performance            (winning model, CV table comparison, holdout results;
                                         reference artifacts/confusion_matrix.png as an image)
        ## Production Artifacts         (what was persisted and how to use it for inference)
        ## Limitations & Recommendations (honest caveats: data size, class balance, next steps)
        """
 
    @staticmethod
    def get_user_prompt(profile_data: dict, execution_stdout: str) -> str:
        return f"""
        Review the pipeline execution outcomes to finalize our model deployment validation.
 
        Data Profile Specs:
        {json.dumps(profile_data, indent=2)}
 
        Sandbox Run Log Output:
        {execution_stdout}
 
        Please synthesize a concise markdown technical performance summary report, verifying
        model stability, feature transformation integrity, and explicit sign-off recommendations.
        """