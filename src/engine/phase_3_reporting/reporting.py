"""
reporting.py — The Analyst Engine (Phase 3 Reporting).

Assembles every meaningful artifact from Phases 1 & 2 into a single
token-budgeted context, makes ONE cloud Analyst call, and writes
final_report.md into the workspace.

Sources consumed (all optional — the report degrades gracefully):
  session_state.json             objective + per-step stdout tails (STATE lines, metrics)
  data_info/llm_summary.txt      human-readable dataset overview (Phase 1 Analyzer)
  data_info/blueprint.md         the approved pipeline contract
  artifacts/manifest.json        model type, CV scores, feature list
  artifacts/model_cv_report.csv  per-model cross-validation comparison
  artifacts/confusion_matrix.png embedded as an image link
  artifacts/*                    inventoried by name for the appendix
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import requests
from dotenv import load_dotenv
from rich.console import Console

from ..core.config import Analyst

console = Console(color_system="standard")


class ReporterError(RuntimeError):
    """Raised when the Analyst cloud call cannot be completed."""


class Phase3Reporter:
    """Gathers pipeline evidence and synthesizes the final analyst report."""

    # Per-source character budgets (~4 chars/token) keep the single Analyst
    # call comfortably inside context limits even for long pipelines.
    BUDGETS: dict[str, int] = {
        "llm_summary": 3000,
        "blueprint": 4000,
        "step_logs": 8000,
        "cv_report": 2000,
        "manifest": 2000,
    }

    def __init__(self, workspace_path: str,
                 chat_fn: Optional[Any] = None) -> None:
        self.workspace_path: Path = Path(workspace_path).resolve()
        self.state_file: Path = self.workspace_path / "session_state.json"
        self.artifacts_dir: Path = self.workspace_path / "artifacts"
        self.data_info_dir: Path = self.workspace_path / "data_info"
        # Injectable for testing: chat_fn(system, user) -> str
        self._chat_fn = chat_fn
        self.state: dict = {}

    # ------------------------------------------------------------ state io
    def _load_state(self) -> None:
        if not self.state_file.exists():
            raise FileNotFoundError(
                f"session_state.json not found in {self.workspace_path}")
        with open(self.state_file, "r", encoding="utf-8") as f:
            self.state = json.load(f)

    def _persist_state(self) -> None:
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)

    # ---------------------------------------------------- context assembly
    def _read_capped(self, path: Path, cap: int) -> str:
        """Read a text file, truncating from the middle if over budget."""
        if not path.exists():
            return ""
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if len(text) <= cap:
            return text
        half = cap // 2
        return text[:half] + "\n\n[... truncated for token budget ...]\n\n" + text[-half:]

    def _gather_step_logs(self) -> str:
        """Per-step stdout tails from session state — the run's evidence trail."""
        results: dict = self.state.get("Phase_2_Execution", {}).get("step_results", {})
        plan: dict = self.state.get("Phase_2_Execution", {}).get("execution_plan", {})
        if not results:
            return "(no step results recorded)"
        blocks: list[str] = []
        for step_id, data in results.items():
            title = plan.get(step_id, {}).get("title", step_id)
            title = title.replace("#", "").replace("*", "").strip()
            status = "SUCCESS" if data.get("success") else "FAILED"
            attempts = data.get("attempts", "?")
            tail = (data.get("stdout_tail") or data.get("stderr_tail") or "").strip()
            blocks.append(f"[{step_id} | {title} | {status} | attempts={attempts}]\n{tail}")
        text = "\n\n".join(blocks)
        cap = self.BUDGETS["step_logs"]
        return text if len(text) <= cap else text[-cap:]

    def _gather_manifest(self) -> tuple[str, dict]:
        """Manifest text for the prompt + parsed dict for session metrics."""
        path = self.artifacts_dir / "manifest.json"
        if not path.exists():
            return "(no manifest.json found)", {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return self._read_capped(path, self.BUDGETS["manifest"]), {}
        return json.dumps(data, indent=1)[: self.BUDGETS["manifest"]], data

    def _gather_artifact_inventory(self) -> str:
        if not self.artifacts_dir.exists():
            return "(no artifacts directory)"
        rows = []
        for f in sorted(self.artifacts_dir.iterdir()):
            if f.is_file():
                rows.append(f"- {f.name} ({f.stat().st_size:,} bytes)")
        return "\n".join(rows) if rows else "(artifacts directory is empty)"

    def _extract_metrics(self, manifest: dict) -> dict:
        """Pull numeric scores out of the manifest for session_state metrics."""
        metrics: dict = {}

        def walk(obj: Any, prefix: str = "") -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, dict):
                        walk(v, f"{prefix}{k}.")
                    elif isinstance(v, (int, float)) and not isinstance(v, bool):
                        key = f"{prefix}{k}"
                        if any(t in key.lower() for t in
                               ("f1", "score", "accuracy", "auc", "precision",
                                "recall", "cv", "loss")):
                            metrics[key] = v

        walk(manifest)
        if isinstance(manifest.get("model_type"), str):
            metrics["model_type"] = manifest["model_type"]
        return metrics

    def build_context(self) -> tuple[str, dict]:
        """Assemble the full evidence package for the Analyst."""
        objective = self.state.get("Phase_1_Planning", {}).get("prompt", "(not recorded)")
        llm_summary = self._read_capped(self.data_info_dir / "llm_summary.txt",
                                        self.BUDGETS["llm_summary"]) or "(not available)"
        blueprint = self._read_capped(self.data_info_dir / "blueprint.md",
                                      self.BUDGETS["blueprint"]) or "(not available)"
        cv_report = self._read_capped(self.artifacts_dir / "model_cv_report.csv",
                                      self.BUDGETS["cv_report"]) or "(not available)"
        manifest_txt, manifest = self._gather_manifest()

        context = f"""
🎯 USER OBJECTIVE:
{objective}

📖 DATASET OVERVIEW (written by the Phase 1 analyzer):
{llm_summary}

📋 APPROVED PIPELINE BLUEPRINT:
{blueprint}

🧾 EXECUTION EVIDENCE (per-step console output from the sandbox):
{self._gather_step_logs()}

🏆 MODEL MANIFEST (artifacts/manifest.json):
{manifest_txt}

📊 CROSS-VALIDATION REPORT (artifacts/model_cv_report.csv):
{cv_report}

🗃️ ARTIFACT INVENTORY:
{self._gather_artifact_inventory()}
"""
        return context, manifest

    # -------------------------------------------------------- cloud call
    def _default_chat(self, system: str, user: str) -> str:
        env_path = self.workspace_path / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=True)
        api_key = os.getenv("GROQ_CODING_KEY") or os.getenv("GROQ_API_KEY") or ""
        if not api_key:
            raise ReporterError(
                f"No GROQ_CODING_KEY / GROQ_API_KEY found (checked {env_path}).")

        payload = {
            "model": Analyst.MODEL,
            "temperature": 0.3,
            "max_tokens": 4096,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        }
        headers = {"Authorization": f"Bearer {api_key}"}
        last = ""
        for attempt in range(4):
            try:
                r = requests.post(Analyst.ENDPOINT, json=payload,
                                  headers=headers, timeout=180)
                if r.status_code == 429 or r.status_code >= 500:
                    last = f"HTTP {r.status_code}"
                    time.sleep(2 ** (attempt + 1))
                    continue
                if r.status_code in (400, 404):
                    body = r.text[:300].replace(api_key, "<redacted>")
                    raise ReporterError(
                        f"Groq rejected the Analyst call (HTTP {r.status_code}) "
                        f"for model '{Analyst.MODEL}'. Body: {body}")
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            except requests.RequestException as exc:
                last = str(exc)
                time.sleep(2 ** attempt)
        raise ReporterError(f"Analyst call failed after retries: {last}")

    # ------------------------------------------------------------- engine
    def execute(self) -> bool:
        try:
            self._load_state()
        except FileNotFoundError as exc:
            console.print(f"  [bold red]└─ Phase 3 Failed:[/bold red] {exc}")
            return False

        p2_status = self.state.get("Phase_2_Execution", {}).get("status")
        if p2_status != "COMPLETED":
            console.print(f"  [bold red]└─ Phase 3 blocked:[/bold red] Phase 2 "
                          f"status is '{p2_status}', not COMPLETED. Finish the "
                          f"pipeline before reporting.")
            return False

        phase3: dict = self.state.setdefault("Phase_3_Reporting", {})
        phase3["status"] = "RUNNING"
        self._persist_state()

        console.print("  [white]├─ Assembling report evidence "
                      "(logs, manifest, CV table, dataset summary)...[/white]")
        context, manifest = self.build_context()

        console.print(f"  [white]├─ Analyst model synthesizing final report "
                      f"([cyan]{Analyst.MODEL}[/cyan])...[/white]")
        try:
            chat = self._chat_fn or self._default_chat
            report: str = chat(Analyst.SYSTEM_PROMPT, Analyst.get_report_prompt(context))
        except ReporterError as exc:
            console.print(f"  [bold red]└─ Phase 3 Failed:[/bold red] {exc}")
            phase3["status"] = "HALTED"
            self._persist_state()
            return False

        # Embed the confusion matrix if the Analyst didn't reference it.
        cm = self.artifacts_dir / "confusion_matrix.png"
        if cm.exists() and "confusion_matrix.png" not in report:
            report += ("\n\n## Confusion Matrix\n"
                       "![Confusion Matrix](artifacts/confusion_matrix.png)\n")

        report_path = self.workspace_path / "final_report.md"
        report_path.write_text(report, encoding="utf-8")

        phase3["status"] = "COMPLETED"
        phase3["final_report_path"] = "final_report.md"
        phase3["metrics"] = self._extract_metrics(manifest)
        self._persist_state()

        console.print(f"  [white]│    ↳[/white] [green]Report written → "
                      f"{report_path.name}[/green]")
        if phase3["metrics"]:
            headline = ", ".join(f"{k}={v}" for k, v in
                                 list(phase3["metrics"].items())[:4])
            console.print(f"  [white]│      {headline[:110]}[/white]")
        console.print("  [bold green]└─ Phase 3 Completed! Session closed.[/bold green]")
        return True
