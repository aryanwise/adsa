"""
coder.py — The Cloud AI Generator (Phase 2 Execution).

Generates and repairs Python scripts via the Groq completions endpoint.
Security posture: the API key is loaded EXCLUSIVELY from the workspace's own
.env file (dotenv_path + override=True) — never from ambient system env vars —
so each workspace jail carries its own credential.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from ..core.config import Coder


class AICoderError(RuntimeError):
    """Raised when the cloud generation call cannot be completed."""


class AICoder:
    """Groq-backed autonomous Python script generator."""

    _FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)

    def __init__(self, workspace_path: Path) -> None:
        self.workspace_path: Path = Path(workspace_path).resolve()

        env_path = self.workspace_path / ".env"
        if not env_path.exists():
            raise FileNotFoundError(
                f"Workspace credential file missing: {env_path}. "
                "Run Phase 1 first (it provisions the workspace .env), or create "
                "the file with GROQ_CODING_KEY=<your key>.")

        # Explicit dotenv_path + override: workspace jail beats global env.
        load_dotenv(dotenv_path=env_path, override=True)
        self.api_key: str = os.getenv("GROQ_CODING_KEY") or os.getenv("GROQ_API_KEY") or ""
        if not self.api_key:
            raise FileNotFoundError(
                f"No GROQ_CODING_KEY (or GROQ_API_KEY) found inside {env_path}.")

    # ------------------------------------------------------------ private
    @classmethod
    def _extract_code(cls, raw: str) -> str:
        """Strip markdown wrappers; return a pure compilable script string."""
        matches = cls._FENCE_RE.findall(raw)
        if matches:
            return "\n\n".join(m.strip() for m in matches).strip()
        # No fences: model obeyed "pure code" literally; strip stray backticks.
        return raw.replace("```", "").strip()

    def _chat(self, user_prompt: str) -> str:
        """Single chat completion with retry/backoff for 429 and 5xx."""
        payload: dict[str, Any] = {
            "model": Coder.MODEL,
            "temperature": Coder.TEMPERATURE,
            "max_tokens": Coder.MAX_TOKENS,
            "messages": [
                {"role": "system", "content": Coder.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}

        last_error: str = ""
        for attempt in range(4):
            try:
                response = requests.post(Coder.ENDPOINT, json=payload,
                                         headers=headers,
                                         timeout=Coder.TIMEOUT_SECONDS)
                if response.status_code == 429 or response.status_code >= 500:
                    last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                    time.sleep(2 ** (attempt + 1))
                    continue
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except requests.RequestException as exc:
                last_error = str(exc)
                time.sleep(2 ** attempt)
        raise AICoderError(f"Groq generation failed after retries: {last_error}")

    # ------------------------------------------------------------- public
    def generate_script(self, step_title: str, step_details: str,
                        full_blueprint: str, profile_data: dict, memory_str: str) -> str:
        """Compile one blueprint step into a pure Python script string."""
        summary_txt = json.dumps(profile_data, indent=2, default=str)
        prompt = Coder.get_user_prompt(
            step_title=step_title,
            step_details=step_details,
            full_plan=full_blueprint,
            summary_txt=summary_txt,
            memory_str=memory_str  
        )
        return self._extract_code(self._chat(prompt))

    def fix_code(self, broken_code: str, traceback: str) -> str:
        """Feed a crashed script + traceback back to Groq for a repaired variant."""
        prompt = Coder.get_fix_prompt(broken_code=broken_code, traceback=traceback)
        return self._extract_code(self._chat(prompt))
