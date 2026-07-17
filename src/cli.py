import sys
from pathlib import Path

# Force the directory containing cli.py into the system path so 'engine' can be imported
sys.path.insert(0, str(Path(__file__).parent.absolute()))

import os
import json
import subprocess
import shutil
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.json import JSON
from rich.prompt import Confirm

console = Console(color_system="standard")

def print_custom_help(ctx):
    """Prints a custom-styled help page matching the exact uv color palette."""
    console.print("A.D.S.A (Autonomous Data Science Agent)")
    
    console.print("\n[green]Usage:[/green] adsa [cyan]<COMMAND>[/cyan] [cyan][OPTIONS][/cyan]")
    
    console.print("\n[green]Commands:[/green]")
    
    table = Table(show_header=False, show_edge=False, box=None, padding=(0, 4, 0, 0))
    table.add_column("Command", style="cyan")
    table.add_column("Description", style="white")

    table.add_row("init", "Initializes a sandboxed agent workspace with an isolated virtual environment")
    table.add_row("list", "Lists all active workspaces and their current Phase progress")
    table.add_row("run", "Runs the pipeline from src/api/main.py")
    table.add_row("session", "Displays all project information from the session_state.json")
    table.add_row("progress", "Displays only the completed tasks from the session state json")
    table.add_row("del", "Deletes a project workspace completely with all its contents")
    
    console.print(table)

    console.print("\n[green]Command Arguments & Options:[/green]")
    arg_table = Table(show_header=False, show_edge=False, box=None, padding=(0, 4, 0, 0))
    arg_table.add_column("Option", style="cyan")
    arg_table.add_column("Description", style="white")
    
    arg_table.add_row("<PROJECT_NAME>", "The target workspace name (Required for all commands)")
    arg_table.add_row("--phase <1|2|3>", "Execute a specific phase (Used with the 'run' command)")
    arg_table.add_row("--step <INT>", "Execute a specific step inside Phase 2 (Used with the 'run' command)")
    arg_table.add_row("--dataset <STR>", "Target dataset name for Phase 1 (Used with the 'run' command)")
    arg_table.add_row("--objective <STR>", "The main goal for the agent (Used with the 'run' command)")
    
    console.print(arg_table)
    
    console.print("\n[green]Global Options:[/green]")
    opt_table = Table(show_header=False, show_edge=False, box=None, padding=(0, 4, 0, 0))
    opt_table.add_column("Option", style="cyan")
    opt_table.add_column("Description", style="white")
    opt_table.add_row("-h, --help", "Show this message and exit")
    
    console.print(opt_table)
    ctx.exit()

# Set custom context options to override default help behaviour
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.group(context_settings=CONTEXT_SETTINGS, invoke_without_command=True)
@click.pass_context
def main(ctx):
    """Autonomous Data Science Agent - Command Line Interface."""
    if ctx.invoked_subcommand is None:
        print_custom_help(ctx)

# ==========================================================
# 1. INIT COMMAND
# ==========================================================
@main.command()
@click.argument('project_name')
def init(project_name):
    """Initializes a sandboxed agent workspace with an isolated virtual environment."""
    if sys.platform == "win32":
        os.system('chcp 65001 >nul')

    target_dir = Path("workspaces") / project_name

    # Create the new segmented data directory structure
    (target_dir / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (target_dir / "data" / "active").mkdir(parents=True, exist_ok=True)
    (target_dir / "data_info").mkdir(parents=True, exist_ok=True)
    (target_dir / "generated_scripts").mkdir(parents=True, exist_ok=True)
    (target_dir / "templates").mkdir(parents=True, exist_ok=True)
    (target_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    venv_dir = target_dir / "venv"
    uv_available = subprocess.run(["where" if sys.platform == "win32" else "which", "uv"], 
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0

    if uv_available:
        subprocess.run(["uv", "venv", str(venv_dir), "--seed"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        env_type = "uv venv"
    else:
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        env_type = "standard venv"

    (target_dir / "requirements.txt").touch()

    # We will use this .env file to store the API key later if needed
    env_content = f"# Local Engine Configuration\nFRONTEND_API_URL=http://localhost:8000\nWORKSPACE_NAME={project_name}\n"
    with open(target_dir / ".env", "w", encoding="utf-8") as f:
        f.write(env_content)

    session_state = {
        "project_name": project_name,
        "Phase_1_Planning": {"status": "PENDING", "prompt": "", "data_profile_path": "data/schema_summary.txt", "blueprint": []},
        "Phase_2_Execution": {"status": "PENDING", "current_step_index": 1, "step_results": {}},
        "Phase_3_Reporting": {"status": "PENDING", "final_report_path": "", "metrics": {}}
    }
    
    with open(target_dir / "session_state.json", "w", encoding="utf-8") as f:
        json.dump(session_state, f, indent=2)

    python_exec = "venv\\Scripts\\python.exe" if sys.platform == "win32" else "venv/bin/python"

    console.print("Autonomous Data Science Agent workspace deployment package.")
    console.print(f"\n[green]Usage:[/green] adsa [cyan]init[/cyan] [cyan]{project_name}[/cyan]")
    console.print("\n[green]Workspace Details:[/green]")

    table = Table(show_header=False, show_edge=False, box=None, padding=(0, 4, 0, 0))
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("project", project_name)
    table.add_row("directory", str(target_dir))
    table.add_row("environment", env_type)
    table.add_row("executable", str(target_dir / python_exec))
    table.add_row("config", ".env + session_state.json generated")
    console.print(table)

# ==========================================================
# 2. RUN COMMAND
# ==========================================================
# ==========================================================
# 2. RUN COMMAND
# ==========================================================
@main.command()
@click.argument('project_name')
@click.option('--phase', type=click.Choice(['1', '2', '3']), help="Execute a specific phase.")
@click.option('--step', type=int, help="Execute a specific step inside Phase 2.")
def run(project_name, phase, step):
    """Runs the full pipeline or specific phases/steps."""
    target_dir = Path("workspaces") / project_name
    
    if not target_dir.exists():
        console.print(f"[bold red]Error:[/bold red] Workspace '{project_name}' not found.")
        return

    console.print(f"\n[green]Starting Pipeline:[/green] [cyan]{project_name}[/cyan]")
    
    if phase == '1':
        # Dynamic import to keep CLI fast
        try:
            from engine.phase_1_planning.orchestrator import Phase1Orchestrator
        except ImportError as e:
            console.print(f"[bold red]Error importing Phase 1 Orchestrator:[/bold red] {e}")
            sys.exit(1)
            
        orchestrator = Phase1Orchestrator(workspace_path=str(target_dir))
        success = orchestrator.execute()  # We no longer pass args here!
        
        if not success:
            sys.exit(1)
            
    elif phase == '2':
        try:
            from engine.phase_2_execution.orchestrator import Phase2Orchestrator
        except ImportError as e:
            console.print(f"[bold red]Error importing Phase 2 Orchestrator:[/bold red] {e}")
            sys.exit(1)
            
        orchestrator = Phase2Orchestrator(workspace_path=str(target_dir))
        success = orchestrator.execute()
        
        if not success:
            sys.exit(1)
        
    elif phase == '3':
        console.print(f"[white]Executing Phase 3...[/white]")
        console.print("[bold yellow]Development Notice:[/bold yellow] Phase 3 orchestrator pending.")
        
    else:
        console.print("[white]Executing Full Pipeline (Phases 1 -> 2 -> 3)...[/white]")
        console.print("[bold yellow]Development Notice:[/bold yellow] Full pipeline sequence pending.")

# ==========================================================
# 3. SESSION COMMAND
# ==========================================================
@main.command()
@click.argument('project_name')
def session(project_name):
    """Displays all information about the project from session_state.json."""
    state_file = Path("workspaces") / project_name / "session_state.json"
    if not state_file.exists():
        console.print(f"[bold red]Error:[/bold red] Workspace '{project_name}' not found.")
        return

    with open(state_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    console.print(f"\n[green]Session State for:[/green] [cyan]{project_name}[/cyan]")
    console.print(JSON.from_data(data))

# ==========================================================
# 4. PROGRESS COMMAND
# ==========================================================
@main.command()
@click.argument('project_name')
def progress(project_name):
    """Displays only the completed tasks from json session state."""
    state_file = Path("workspaces") / project_name / "session_state.json"
    if not state_file.exists():
        console.print(f"[bold red]Error:[/bold red] Workspace '{project_name}' not found.")
        return

    with open(state_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    console.print(f"\n[green]Progress Report:[/green] [cyan]{project_name}[/cyan]")
    
    table = Table(show_header=True, header_style="cyan")
    table.add_column("Phase/Step")
    table.add_column("Status")
    
    # Check Phase 1
    p1_status = data.get("Phase_1_Planning", {}).get("status", "UNKNOWN")
    status_color = "green" if p1_status == "COMPLETED" else "yellow" if p1_status == "IN_PROGRESS" else "white"
    table.add_row("Phase 1: Planning", f"[{status_color}]{p1_status}[/{status_color}]")

    # Check Phase 2
    p2_status = data.get("Phase_2_Execution", {}).get("status", "UNKNOWN")
    status_color = "green" if p2_status == "COMPLETED" else "yellow" if p2_status == "IN_PROGRESS" else "white"
    table.add_row("Phase 2: Execution", f"[{status_color}]{p2_status}[/{status_color}]")
    
    step_results = data.get("Phase_2_Execution", {}).get("step_results", {})
    for step_idx, step_data in step_results.items():
        if step_data.get("success"):
            table.add_row(f"  └─ Step {step_idx} ({step_data.get('script')})", "[green]COMPLETED[/green]")

    # Check Phase 3
    p3_status = data.get("Phase_3_Reporting", {}).get("status", "UNKNOWN")
    status_color = "green" if p3_status == "COMPLETED" else "yellow" if p3_status == "IN_PROGRESS" else "white"
    table.add_row("Phase 3: Reporting", f"[{status_color}]{p3_status}[/{status_color}]")

    console.print(table)

# ==========================================================
# 5. DEL COMMAND
# ==========================================================
@main.command(name="del")
@click.argument('project_name')
def delete_workspace(project_name):
    """Deletes a project workspace completely with all its contents."""
    target_dir = Path("workspaces") / project_name
    
    if not target_dir.exists():
        console.print(f"[bold red]Error:[/bold red] Workspace '{project_name}' not found.")
        return

    console.print(f"\n[bold red]⚠️  WARNING:[/bold red] You are about to completely delete the workspace [cyan]{project_name}[/cyan].")
    console.print("[red]This will permanently destroy all data, generated scripts, virtual environments, and session states inside this project.[/red]\n")
    
    is_confirmed = Confirm.ask(f"[bold red]Are you absolutely sure you want to delete {project_name}?[/bold red]")
    
    if is_confirmed:
        try:
            shutil.rmtree(target_dir)
            console.print(f"\n  [bold green]✓[/bold green] Workspace [cyan]{project_name}[/cyan] has been successfully deleted.")
        except Exception as e:
            console.print(f"\n[bold red]Error deleting workspace:[/bold red] {e}")
    else:
        console.print("\n[yellow]Deletion cancelled. The workspace remains untouched.[/yellow]")

# ==========================================================
# 6. LIST COMMAND
# ==========================================================
@main.command(name="list")
def list_projects():
    """Lists all active workspaces and their current progress."""
    workspaces_path = Path("workspaces")
    
    if not workspaces_path.exists() or not workspaces_path.is_dir():
        console.print("[yellow]⚠️ No 'workspaces' directory found. You don't have any projects yet.[/yellow]")
        return
        
    projects = [d for d in workspaces_path.iterdir() if d.is_dir()]
    
    if not projects:
        console.print("[yellow]📂 No active projects found in the workspaces directory.[/yellow]")
        return

    console.print("\n[green]📁 Active ADSA Projects[/green]")
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Project Name", style="cyan", no_wrap=True)
    table.add_column("Phase 1 (Planning)", justify="center")
    table.add_column("Phase 2 (Execution)", justify="center")
    
    for proj in sorted(projects):
        project_name = proj.name
        state_file = proj / "session_state.json"
        
        p1_status = "[dim]NOT STARTED[/dim]"
        p2_status = "[dim]NOT STARTED[/dim]"
        
        if state_file.exists():
            try:
                with open(state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                    
                p1_raw = state.get("Phase_1_Planning", {}).get("status", "PENDING")
                p2_raw = state.get("Phase_2_Execution", {}).get("status", "PENDING")
                
                # Color code Phase 1
                p1_status = f"[green]{p1_raw}[/green]" if p1_raw == "COMPLETED" else f"[yellow]{p1_raw}[/yellow]"
                
                # Color code Phase 2
                if p2_raw == "COMPLETED":
                    p2_status = f"[green]{p2_raw}[/green]"
                elif p2_raw == "IN_PROGRESS":
                    p2_status = f"[blue]{p2_raw}[/blue]"
                else:
                    p2_status = f"[yellow]{p2_raw}[/yellow]"
                    
            except Exception:
                p1_status = "[red]ERROR[/red]"
                p2_status = "[red]ERROR[/red]"
                
        table.add_row(project_name, p1_status, p2_status)
        
    console.print(table)
    console.print()

if __name__ == '__main__':
    main()