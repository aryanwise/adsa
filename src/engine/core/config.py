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
    """High-level structural planning & task checklists (Fast cloud execution)"""
    MODEL = "openai/gpt-oss-120b"
    ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

    SYSTEM_PROMPT = """
    You are a Principal Data Science Architect. Your job is to translate statistical metadata and user objectives into a rigorous, production-grade machine learning blueprint.
    You DO NOT write raw python code. You DO NOT write conversational filler.
    Every word must serve a technical purpose.

    SECURITY RULE:
    Assume the execution environment is a strictly jailed workspace directory.
    All file reads, writes, and model persistence steps MUST use relative paths (e.g., 'data/active/data.csv' or 'models/model.pkl'). Do not use absolute paths or attempt directory traversal.
    """

    @staticmethod
    def get_user_prompt(profile_data: dict, user_objective: str) -> str:
        return f"""
        🎯 USER PROBLEM PROMPT (Goal):
        {user_objective}

        📊 DATASET PROFILE (Anonymized Statistics):
        {json.dumps(profile_data, indent=2)}

        Design an end-to-end data science execution blueprint based on the Goal and Data Profile.

        CRITICAL INSTRUCTIONS:
        1. Address specific data quality issues (skewness, scaling, categorical encoding) based on the JSON profile provided.
        2. In the modeling step, explicitly suggest 2 to 3 distinct machine learning algorithms that fit the problem, specifying which one to start with as the baseline.
        3. Include a robust validation strategy (e.g., Stratified K-Fold) and explicit evaluation metrics.

        FORMATTING RULES:
        - Output strictly a numbered list (e.g., "**Step 1: [Task Name]**").
        - Group your steps logically (Data Loading -> Preprocessing -> Modeling -> Evaluation -> Persistence).
        - Limit the entire blueprint to 8-12 steps.
        - Use a maximum of 3 highly detailed, technical bullet points per step.
        - Do NOT include any text outside of the numbered steps. Start immediately with Step 1.
        """


class Coder:
    """Autonomous Python script generator for Phase 2 Execution (Groq Cloud)."""
    MODEL = os.getenv("GROQ_CODING_KEYL", "llama-3.3-70b-versatile")
    ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
    TEMPERATURE = 0.1
    MAX_TOKENS = 4096
    TIMEOUT_SECONDS = 120

    SYSTEM_PROMPT = """
    You are an expert Data Science AI Programmer operating within a sandboxed workspace.
    CRITICAL RULES:
    1. OUTPUT ONLY THE PURE PYTHON CODE BLOCK. No markdown wraps (unless specified), no conversation, no introductions.
    2. Ensure the code is strictly non-interactive (save plots using plt.savefig(), never plt.show()).
    3. All data reads/writes must use relative paths targeting 'data/active/copy_test_data.csv' or the 'artifacts/' folder.
    4. You are PROHIBITED from using absolute system paths or directory traversals (../) to escape the workspace directory.
    """

    @staticmethod
    def get_user_prompt(step_title: str, step_details: str,
                        full_plan: str, summary_txt: str) -> str:
        return f"""
        CONTEXT METADATA SUMMARY:
        {summary_txt}

        FULL APPROVED PIPELINE BLUEPRINT:
        {full_plan}

        ------------------------------------------------------------------------
        CURRENT FOCUS TASK:
        You must write the code strictly for this step only:
        🎯 {step_title}
        Detailed Requirements:
        {step_details}
        ------------------------------------------------------------------------

        IMPLEMENTATION INSTRUCTIONS:
        - Output ONLY code inside a single ```python block.
        - The active dataset lives at 'data/active/copy_test_data.csv' (relative path).
        - Each step must read the current state of that CSV, apply ONLY this step's
          transformation, and overwrite the SAME file so the next step inherits it.
        - Persist any other critical outputs (models, plots, JSON maps) into the
          'artifacts/' directory using relative paths.
        - Print concise confirmations of what was done (shapes, metrics, file paths).
        """

    @staticmethod
    def get_fix_prompt(broken_code: str, traceback: str) -> str:
        return f"""
        The previously generated Python script crashed during sandboxed execution.

        [BROKEN SCRIPT]:
        ```python
        {broken_code}
        ```

        [INTERCEPTED RUNTIME TRACEBACK]:
        {traceback}

        Analyze the traceback, find the root cause, and rewrite the ENTIRE corrected
        script. Preserve the step's intent and its input/output file contracts.
        Output ONLY the fixed script inside a single ```python block.
        """


class Analyst:
    """Senior executive-grade summary reporting & massive token validation"""
    MODEL = "gemini-1.5-pro-preview-0409"

    SYSTEM_PROMPT = """
    You are a Senior Data Science Lead Analyst. Your task is to perform an executive validation audit
    on automated pipeline runs, rendering highly professional performance wrap-ups.
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