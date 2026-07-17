"""
sandbox.py — The Shield & Dependency Manager (Phase 2 Execution).

Runs generated scripts in an isolated subprocess with:
  - Virtual-env inheritance via sys.executable (the interpreter running the CLI)
  - Workspace jailing: any path outside the workspace raises PermissionError
  - Self-healing dependencies: ModuleNotFoundError -> append to requirements.txt
    -> pip install -r requirements.txt -> automatic re-run
  - Hard wall-clock timeout so an infinite loop can't hang the engine
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console(color_system="standard")

# "No module named 'xgboost'" / "No module named xgboost"
_MISSING_MODULE_RE = re.compile(r"No module named ['\"]?([A-Za-z0-9_\.]+)['\"]?")

# import name -> pip package name, for the usual suspects
_PIP_ALIASES: dict[str, str] = {
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "PIL": "pillow",
    "yaml": "pyyaml",
    "dotenv": "python-dotenv",
    "bs4": "beautifulsoup4",
}

# Final exception line, e.g. "KeyError: 'MONTH_NUM'"
_ERROR_TYPE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception|Warning|Interrupt))\b",
                            re.MULTILINE)


class SandboxExecutor:
    """Executes workspace scripts inside a jailed, self-healing subprocess."""

    def __init__(self, workspace_path: Path, timeout_seconds: int = 900,
                 max_heal_cycles: int = 3) -> None:
        self.workspace_path: Path = Path(workspace_path).resolve()
        self.timeout_seconds: int = timeout_seconds
        self.max_heal_cycles: int = max_heal_cycles
        self.requirements_path: Path = self.workspace_path / "requirements.txt"

    def _get_python_executable(self) -> str:
        """Resolves the isolated workspace venv python binary, falling back to sys.executable."""
        import platform
        import sys
        
        if platform.system() == "Windows":
            venv_python = self.workspace_path / "venv" / "Scripts" / "python.exe"
        else:
            venv_python = self.workspace_path / "venv" / "bin" / "python"
            
        if venv_python.exists():
            return str(venv_python)
            
        # Fallback only if the user deleted the workspace venv
        return sys.executable

    # ------------------------------------------------------------ jailer
    def _is_path_safe(self, path_str: str) -> bool:
        """True only if the resolved path lives strictly inside the workspace."""
        try:
            requested = Path(path_str).resolve()
        except Exception:
            return False
        return requested == self.workspace_path or \
            self.workspace_path in requested.parents

    # ------------------------------------------------- dependency healing
    def _parse_missing_module(self, stderr: str) -> Optional[str]:
        match = _MISSING_MODULE_RE.search(stderr)
        if not match:
            return None
        return match.group(1).split(".")[0]

    def _append_requirement(self, package: str) -> None:
        existing: set[str] = set()
        content = ""
        
        if self.requirements_path.exists():
            content = self.requirements_path.read_text(encoding="utf-8")
            existing = {
                line.strip().split("==")[0].lower()
                for line in content.splitlines()
                if line.strip() and not line.strip().startswith("#")
            }
            
        if package.lower() not in existing:
            with open(self.requirements_path, "a", encoding="utf-8") as f:
                # Safely ensure we are on a new line before appending
                if content and not content.endswith("\n"):
                    f.write("\n")
                f.write(f"{package}\n")

    def _pip_install_requirements(self) -> bool:
        """Bulk install; on failure, fall back to per-package installs and PRUNE
        unresolvable entries (e.g. truncated names like 'sc') from
        requirements.txt so one junk line can never poison the whole file."""
        python_bin = self._get_python_executable()
        base_cmd = [python_bin, "-m", "pip", "install", "-r",
                    str(self.requirements_path)]

        result = subprocess.run(base_cmd, capture_output=True, text=True)
        if result.returncode != 0 and "externally-managed-environment" in result.stderr:
            result = subprocess.run(base_cmd + ["--break-system-packages"],
                                    capture_output=True, text=True)
        if result.returncode == 0:
            return True

        # ---- per-package salvage pass ----
        console.print("      [yellow]⚕ Bulk install failed — resolving packages "
                      "individually...[/yellow]")
        lines = [l.strip() for l in
                 self.requirements_path.read_text(encoding="utf-8").splitlines()
                 if l.strip() and not l.strip().startswith("#")]
        keep: list[str] = []
        all_ok = True
        for pkg in lines:
            r = subprocess.run([python_bin, "-m", "pip", "install", pkg, "--quiet"],
                               capture_output=True, text=True)
            if r.returncode != 0 and "externally-managed-environment" in r.stderr:
                r = subprocess.run([python_bin, "-m", "pip", "install", pkg,
                                    "--quiet", "--break-system-packages"],
                                   capture_output=True, text=True)
            if r.returncode == 0:
                keep.append(pkg)
            elif ("No matching distribution" in r.stderr
                  or "Invalid requirement" in r.stderr
                  or "not find a version" in r.stderr):
                console.print(f"      [yellow]⚕ pruned unresolvable requirement:"
                              f" '{pkg}'[/yellow]")
            else:
                # Real install failure (build error, network): keep the line,
                # report the tail, and flag the run as degraded.
                keep.append(pkg)
                all_ok = False
                console.print(f"      [bold red]Pip failed for '{pkg}':[/bold red] "
                              f"{r.stderr.strip().splitlines()[-1] if r.stderr.strip() else 'unknown'}")
        self.requirements_path.write_text("\n".join(keep) + ("\n" if keep else ""),
                                          encoding="utf-8")
        return all_ok

    def _attempt_self_heal(self, stderr: str) -> bool:
        """Detect a missing module, install it, and report whether to re-run."""
        module = self._parse_missing_module(stderr)
        if module is None:
            return False
        package = _PIP_ALIASES.get(module, module)
        console.print(f"      [yellow]⚕ Missing dependency detected:[/yellow] "
                      f"[white]{module}[/white] → pip package [white]{package}[/white]")
        self._append_requirement(package)
        console.print(f"      [yellow]⚕ Auto-installing from requirements.txt ...[/yellow]")
        if self._pip_install_requirements():
            console.print(f"      [green]⚕ Dependency installed. Re-running script.[/green]")
            return True
        console.print(f"      [red]⚕ pip install failed for '{package}'.[/red]")
        return False

    # --------------------------------------------------------- execution
    def _classify_error(self, stderr: str) -> str:
        matches = _ERROR_TYPE_RE.findall(stderr)
        return matches[-1] if matches else ("UnknownError" if stderr.strip() else "")

    def execute(self, script_path: str) -> dict:
        """Run a workspace script. Returns
        {"success": bool, "stdout": str, "stderr": str, "error_type": str}."""
        script = (self.workspace_path / script_path) \
            if not Path(script_path).is_absolute() else Path(script_path)

        if not self._is_path_safe(str(script)):
            raise PermissionError(
                f"Workspace jail violation: '{script_path}' resolves outside "
                f"'{self.workspace_path}'.")
        if not script.exists():
            return {"success": False, "stdout": "",
                    "stderr": f"Script not found: {script}",
                    "error_type": "FileNotFoundError"}

        heal_cycles = 0
        while True:
            try:
                python_bin = self._get_python_executable()
                proc = subprocess.run(
                    [python_bin, str(script)],
                    capture_output=True,
                    text=True,
                    cwd=str(self.workspace_path),   # relative paths resolve in-jail
                    timeout=self.timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                return {"success": False,
                        "stdout": (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
                        "stderr": f"Execution exceeded {self.timeout_seconds}s wall-clock limit.",
                        "error_type": "TimeoutExpired"}

            if proc.returncode == 0:
                return {"success": True, "stdout": proc.stdout,
                        "stderr": proc.stderr, "error_type": ""}

            if heal_cycles < self.max_heal_cycles and self._attempt_self_heal(proc.stderr):
                heal_cycles += 1
                continue  # dependency fixed -> immediately re-run the script

            return {"success": False, "stdout": proc.stdout,
                    "stderr": proc.stderr,
                    "error_type": self._classify_error(proc.stderr)}
