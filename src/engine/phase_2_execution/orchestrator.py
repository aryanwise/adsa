"""
orchestrator.py — The Stateful Memory Engine (Phase 2 Execution).

Reads the dynamic execution graph from session_state.json. Features:
  - Pre-flight Dependency Installer (reads requirements.txt before starting).
  - Token-Safe Sliding Window Memory (feeds past step logs to prevent AI amnesia).
  - Generates, executes, and repairs via 3-attempt cycle.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from rich.console import Console

from .coder import AICoder, AICoderError
from .sandbox import SandboxExecutor

console = Console(color_system="standard")

MAX_ATTEMPTS_PER_STEP: int = 3   # 1 initial generation + up to 2 repair cycles


class Phase2Orchestrator:
    """Drives the generate -> execute -> repair loop over the execution plan."""

    def __init__(self, workspace_path: str,
                 coder: Optional[AICoder] = None,
                 sandbox: Optional[SandboxExecutor] = None) -> None:
        self.workspace_path: Path = Path(workspace_path).resolve()
        self.state_file: Path = self.workspace_path / "session_state.json"
        self.scripts_dir: Path = self.workspace_path / "generated_scripts"
        self.scripts_dir.mkdir(parents=True, exist_ok=True)

        # Injectable for testing; real instances built lazily in execute().
        self._coder: Optional[AICoder] = coder
        self._sandbox: SandboxExecutor = sandbox or SandboxExecutor(self.workspace_path)

        self.state: dict = {}

    # ------------------------------------------------------------ state io
    def _load_state(self) -> None:
        if not self.state_file.exists():
            raise FileNotFoundError(f"session_state.json not found in {self.workspace_path}")
        with open(self.state_file, "r", encoding="utf-8") as f:
            self.state = json.load(f)

    def _persist_state(self) -> None:
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)

    # ------------------------------------------------------- plan context
    def _load_planning_artifacts(self) -> tuple[str, dict]:
        """Returns (full_blueprint_text, profile_data_dict) from Phase 1 outputs."""
        planning: dict = self.state.get("Phase_1_Planning", {})

        blueprint_rel = planning.get("blueprint_path", "data_info/blueprint.md") # Ensure .md extension
        blueprint_file = self.workspace_path / blueprint_rel
        full_blueprint = blueprint_file.read_text(encoding="utf-8") \
            if blueprint_file.exists() else "\n".join(planning.get("blueprint", []))

        profile_rel = planning.get("data_profile_path", "data_info/profiler_summary.txt")
        profile_file = self.workspace_path / profile_rel
        profile_data: dict = {}
        if profile_file.exists():
            try:
                profile_data = json.loads(profile_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                profile_data = {"raw_profile_text": profile_file.read_text(encoding="utf-8")[:4000]}
        return full_blueprint, profile_data

    @staticmethod
    def _split_blueprint_details(full_blueprint: str) -> dict[str, str]:
        """Map each '### Step N: ...' header to its detail block (header + bullets)."""
        blocks: dict[str, str] = {}
        current_title: Optional[str] = None
        current_lines: list[str] = []
        for line in full_blueprint.splitlines():
            # Support both Markdown H3 and legacy Bold formats
            if line.strip().startswith("### Step") or line.strip().startswith("**Step"):
                if current_title is not None:
                    blocks[current_title] = "\n".join(current_lines).strip()
                current_title = Phase2Orchestrator._normalize_title(line)
                current_lines = [line]
            elif current_title is not None:
                current_lines.append(line)
        if current_title is not None:
            blocks[current_title] = "\n".join(current_lines).strip()
        return blocks

    @staticmethod
    def _normalize_title(title: str) -> str:
        return re.sub(r"\s+", " ", title.replace("*", "").replace("#", "").strip()).lower()

    # ------------------------------------------------------------- engine
    
    def _get_safe_history(self, step_results: dict, max_chars: int = 1500) -> str:
        """
        Token-Safe Memory: Truncates historical stdout/stderr to prevent API bloat 
        while preserving the state and shape of the data for the next step.
        """
        if not step_results:
            return "No prior steps executed. You are on Step 1."
            
        history = ""
        for step_id, data in step_results.items():
            if data.get("success"):
                tail = data.get('stdout_tail', '')[-max_chars:]
                history += f"✅ {step_id} SUCCESS:\n...{tail}\n\n"
            else:
                tail = data.get('stderr_tail', '')[-max_chars:]
                history += f"❌ {step_id} FAILED:\n...{tail}\n\n"
        return history

    def execute(self) -> bool:
        try:
            self._load_state()
        except FileNotFoundError as exc:
            console.print(f"  [bold red]└─ Phase 2 Failed:[/bold red] {exc}")
            return False

        if self.state.get("Phase_1_Planning", {}).get("status") != "COMPLETED":
            console.print("  [bold red]└─ Phase 2 blocked:[/bold red] "
                          "Phase 1 Planning has not completed for this workspace.")
            return False

        phase2: dict = self.state.setdefault("Phase_2_Execution", {})
        plan: dict = phase2.get("execution_plan", {})
        if not plan:
            console.print("  [bold red]└─ Phase 2 Failed:[/bold red] execution_plan is empty.")
            return False

        phase2["status"] = "RUNNING"
        phase2.setdefault("step_results", {})
        self._persist_state()

        # Build the cloud coder lazily
        try:
            coder = self._coder or AICoder(self.workspace_path)
        except (FileNotFoundError, AICoderError) as exc:
            console.print(f"  [bold red]└─ Phase 2 Failed:[/bold red] {exc}")
            phase2["status"] = "HALTED"
            self._persist_state()
            return False

        # --- PRE-FLIGHT INSTALLATION ---
        req_path = self.workspace_path / "requirements.txt"
        if req_path.exists():
            console.print("  [white]├─ Checking sandbox dependencies...[/white]")
            console.print("  [white]│    ↳ Pre-installing requirements.txt...[/white]")
            success = self._sandbox._pip_install_requirements()
            if success:
                console.print("  [white]│    ↳[/white] [green]Dependencies fully loaded.[/green]")
            else:
                console.print("  [white]│    ↳[/white] [yellow]Warning: Pre-install had some errors. Sandbox will auto-heal if needed.[/yellow]")
        # -------------------------------

        full_blueprint, profile_data = self._load_planning_artifacts()
        details_map = self._split_blueprint_details(full_blueprint)

        console.print(f"  [white]├─ Execution graph loaded:[/white] "
                      f"[cyan]{len(plan)} steps[/cyan]")

        for index, (step_id, step) in enumerate(plan.items(), start=1):
            if step.get("status") == "COMPLETED":
                console.print(f"  [white]├─ {step_id}:[/white] "
                              f"[green]already completed — skipping[/green]")
                continue

            phase2["current_step_id"] = step_id
            phase2["current_step_index"] = index
            title: str = step.get("title", step_id)
            details: str = details_map.get(self._normalize_title(title), title)
            script_rel: str = step.get("script_path", f"generated_scripts/{step_id}.py")
            script_abs: Path = self.workspace_path / script_rel

            console.print(f"  [white]├─ Executing {step_id}:[/white] "
                          f"[cyan]{title.replace('*', '').replace('#', '').strip()}[/cyan]")

            success = self._run_single_step(
                coder=coder, step=step, title=title, details=details,
                full_blueprint=full_blueprint, profile_data=profile_data,
                script_abs=script_abs, script_rel=script_rel,
                step_id=step_id, phase2=phase2,
            )

            if not success:
                console.print(f"  [bold red]└─ Engine halted at {step_id} after "
                              f"{step.get('attempts', 0)} attempt(s). "
                              f"Inspect {script_rel} and stderr above.[/bold red]")
                phase2["status"] = "HALTED"
                self._persist_state()
                return False

        phase2["status"] = "COMPLETED"
        self._persist_state()
        console.print("  [bold green]└─ Phase 2 Completed! "
                      "All steps executed safely.[/bold green]")
        return True

    # ----------------------------------------------------- per-step logic
    def _run_single_step(self, *, coder: AICoder, step: dict, title: str,
                         details: str, full_blueprint: str, profile_data: dict,
                         script_abs: Path, script_rel: str, step_id: str,
                         phase2: dict) -> bool:
        step["status"] = "IN_PROGRESS"
        step.setdefault("attempts", 0)
        self._persist_state()

        current_code: str = ""
        last_result: dict = {}

        # Fetch sliding-window memory of all prior steps cleanly from the session state
        memory_str = self._get_safe_history(phase2.get("step_results", {}))

        while step["attempts"] < MAX_ATTEMPTS_PER_STEP:
            step["attempts"] += 1
            attempt: int = step["attempts"]
            self._persist_state()

            try:
                if attempt == 1:
                    console.print(f"  [white]│    ↳ generating script "
                                  f"(attempt {attempt}) ...[/white]")
                    
                    # UPDATED: Pass full_blueprint cleanly and add memory_str as its own argument
                    current_code = coder.generate_script(
                        step_title=title, step_details=details,
                        full_blueprint=full_blueprint, profile_data=profile_data,
                        memory_str=memory_str)
                else:
                    console.print(f"  [white]│    ↳ repairing script "
                                  f"(attempt {attempt}) ...[/white]")
                    current_code = coder.fix_code(
                        broken_code=current_code,
                        traceback=last_result.get("stderr", "")[-3000:])
            except AICoderError as exc:
                console.print(f"  [white]│    ↳[/white] [red]generation failed: {exc}[/red]")
                return False

            script_abs.parent.mkdir(parents=True, exist_ok=True)
            script_abs.write_text(current_code, encoding="utf-8")

            try:
                last_result = self._sandbox.execute(str(script_abs))
            except PermissionError as exc:
                console.print(f"  [white]│    ↳[/white] [red]jail violation: {exc}[/red]")
                return False

            if last_result["success"]:
                step["status"] = "COMPLETED"
                phase2["step_results"][step_id] = {
                    "success": True,
                    "script": script_rel,
                    "attempts": attempt,
                    "stdout_tail": last_result["stdout"][-1500:],
                }
                self._persist_state()
                console.print(f"  [white]│    ↳[/white] [green]Success![/green]")
                stdout_lines = [l for l in last_result["stdout"].splitlines() if l.strip()]
                if stdout_lines:
                    console.print(f"  [white]│      {stdout_lines[-1][:110]}[/white]")
                return True

            error_type: str = last_result.get("error_type", "UnknownError")
            console.print(f"  [white]│    ↳[/white] [red]crashed "
                          f"({error_type}) on attempt {attempt}[/red]")

        step["status"] = "FAILED"
        phase2["step_results"][step_id] = {
            "success": False,
            "script": script_rel,
            "attempts": step["attempts"],
            "error_type": last_result.get("error_type", "UnknownError"),
            "stderr_tail": last_result.get("stderr", "")[-1500:],
        }
        self._persist_state()
        return False