import json
from pathlib import Path
from groq import Groq
from engine.core.config import Architect, GROQ_API_KEY

class DataAnalyzer:
    """
    Reads the token-efficient profile data and communicates with the LLM 
    to generate an intelligent, human-readable summary of the dataset.
    """
    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)
        self.data_info_dir = self.workspace_path / "data_info"
        
        if not GROQ_API_KEY:
            raise ValueError("❌ GROQ_API_KEY environment variable is missing.")
        
        self.client = Groq(api_key=GROQ_API_KEY)

    def generate_summary(self, profile_data: dict) -> str:
        """Sends the JSON profile to Groq to generate a professional data summary."""
        
        system_prompt = (
            "You are an elite Lead Data Scientist. Analyze the provided statistical dataset profile. "
            "Provide a concise, highly professional summary of what this dataset represents, "
            "the potential data types of the columns, any critical data quality warnings (like nulls or skew), "
            "and 3 immediate analytical goals that could be built using this data. "
            "Format your response in clean Markdown."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Here is the statistical profile of the dataset:\n{json.dumps(profile_data, indent=2)}"}
        ]

        response = self.client.chat.completions.create(
            model=Architect.MODEL, 
            messages=messages,
            temperature=0.3,
            max_tokens=1024
        )
        
        summary = response.choices[0].message.content
        return summary