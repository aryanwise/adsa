import json
from pathlib import Path
from rich.console import Console
from rich.prompt import Prompt

# Import all Phase 1 components
from .ingestion import DataIngestion
from .profiler import DataProfiler
from .analyzer import DataAnalyzer
from .planner import PipelinePlanner
import os
from dotenv import load_dotenv, set_key

console = Console(color_system="standard")

class Phase1Orchestrator:
    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)
        self.data_info_dir = self.workspace_path / "data_info"
        
        # Ensure data_info exists
        self.data_info_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_workspace_env(self):
        """Ensures a .env file exists in the workspace and contains the Groq Key."""
        env_path = self.workspace_path / ".env"
        
        # If the file doesn't exist, create it empty
        if not env_path.exists():
            env_path.touch()
            
        load_dotenv(dotenv_path=env_path)
        
        # Check if the keys exist specifically in this environment
        if not os.getenv("GROQ_CODING_KEY"):
            console.print("\n[bold yellow]🔒 Workspace Security: No Groq API Key found for this workspace.[/bold yellow]")
            api_key = Prompt.ask("[bold white]🔑 Please enter your Groq API Key (it will be saved to this workspace's .env)[/bold white]", password=True)
            set_key(str(env_path), "GROQ_CODING_KEY", api_key)
            
            # Reload to apply changes
            load_dotenv(dotenv_path=env_path)
            console.print("[green]✅ Key securely saved to workspace .env file![/green]")

    def _is_path_safe(self, path_str: str) -> bool:
        """Ensures that the output path is strictly within the workspace jail."""
        try:
            requested_path = Path(path_str).resolve()
            workspace_root = self.workspace_path.resolve()
            return workspace_root in requested_path.parents or requested_path == workspace_root
        except Exception:
            return False
        
    def execute(self) -> bool:
        """Runs the full Phase 1 sequence interactively"""
        try:
            console.print(f"\n[cyan]⚙️ Initiating Phase 1: Planning[/cyan]")

            self._ensure_workspace_env()
            
            # --- INTERACTIVE PROMPTS ---
            raw_dir = self.workspace_path / "data" / "raw"
            available_files = [f.name for f in raw_dir.glob("*.csv")]
            
            if available_files:
                console.print(f"\n[green]📂 Found existing dataset(s) in workspace:[/green] [bold white]{', '.join(available_files)}[/bold white]")
                dataset_path = Prompt.ask("[bold yellow]📁 Enter filename from above, OR provide a new absolute path[/bold yellow]").strip()
                
                # If the user just typed the filename, resolve it to the full raw path
                if dataset_path in available_files:
                    dataset_path = str(raw_dir / dataset_path)
            else:
                dataset_path = Prompt.ask("\n[bold yellow]📁 Enter the absolute path to your dataset (CSV)[/bold yellow]").strip()
                
            objective = Prompt.ask("[bold yellow]🎯 Enter your objective for this dataset[/bold yellow]").strip()
            console.print("") # formatting newline
            
            # 1. INGESTION
            console.print("  [white]├─ Running Data Ingestion...[/white]")
            ingestion = DataIngestion(str(self.workspace_path))
            active_file_path = ingestion.ingest_csv(file_path_or_name=dataset_path)
            
            # 2. PROFILING 
            console.print("  [white]├─ Generating Statistical Data Profile...[/white]")
            profiler = DataProfiler(csv_path=active_file_path)
            profile_data = profiler.profile_dict()
            
            profiler_output_path = self.data_info_dir / "profiler_summary.txt"
            with open(profiler_output_path, "w", encoding="utf-8") as f:
                json.dump(profile_data, f, indent=4)
            
            # 3. ANALYZER 
            console.print("  [white]├─ Generating LLM Data Summary...[/white]")
            analyzer = DataAnalyzer(str(self.workspace_path))
            llm_summary = analyzer.generate_summary(profile_data=profile_data)
            
            analyzer_output_path = self.data_info_dir / "llm_summary.txt"
            with open(analyzer_output_path, "w", encoding="utf-8") as f:
                f.write(llm_summary)
                
            # 4. PLANNING
            console.print("  [white]├─ Drafting Execution Blueprint Markdown...[/white]")
            planner = PipelinePlanner()
            blueprint = planner.build_plan(profile_data=profile_data, user_objective=objective)
            
            # 5. SAVE BLUEPRINT AS MARKDOWN FILE
            blueprint_path = self.data_info_dir / "blueprint.md"
            with open(blueprint_path, "w", encoding="utf-8") as f:
                f.write(blueprint)

            # 5. SAVE BLUEPRINT AS MARKDOWN FILE
            blueprint_path = self.data_info_dir / "blueprint.md"
            with open(blueprint_path, "w", encoding="utf-8") as f:
                f.write(blueprint)
                
            # --- EXTRACT DYNAMIC REQUIREMENTS ---
            console.print("  [white]├─ Generating requirements.txt...[/white]")
            requirements = []
            in_reqs_section = False
            
            for line in blueprint.split('\n'):
                clean_line = line.strip()
                
                if "4. Python Package Requirements" in clean_line:
                    in_reqs_section = True
                    continue
                
                if in_reqs_section:
                    if clean_line.startswith('#') and "4." not in clean_line:
                        break
                    
                    if clean_line.startswith(('-', '*', '•')):
                        # Use lstrip to ONLY remove the bullet points at the start of the line
                        # Then split to grab the word, and strip any backticks
                        stripped_line = clean_line.lstrip('-*• ').strip()
                        parts = stripped_line.split()
                        
                        if len(parts) > 0:
                            pkg_name = parts[0].replace('`', '').lower()
                            if pkg_name:
                                requirements.append(pkg_name)
            
            req_path = self.workspace_path / "requirements.txt"
            with open(req_path, "w", encoding="utf-8") as f:
                f.write("\n".join(requirements))
            console.print(f"  [green]└─ Saved {len(requirements)} dependencies to requirements.txt[/green]")
                
            # 6. UPDATE SESSION STATE (With Markdown Heading Extraction Graph)
            state_file = self.workspace_path / "session_state.json"
            if state_file.exists():
                with open(state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                
                # UPDATED PARSER: Extracts steps matching the standard Markdown H3 syntax '### Step X:'
                steps = [
                    line.replace('###', '').strip() 
                    for line in blueprint.split('\n') 
                    if line.strip().startswith('### Step')
                ]
                
                # Fallback safeguard: if LLM changes header notation, capture raw bold notation
                if not steps:
                    steps = [line.replace('**', '').strip() for line in blueprint.split('\n') if line.strip().startswith('**Step')]
                
                # Update Phase 1 Planning State Tracker Paths
                state["Phase_1_Planning"]["status"] = "COMPLETED"
                state["Phase_1_Planning"]["raw_data_path"] = dataset_path
                state["Phase_1_Planning"]["prompt"] = objective
                state["Phase_1_Planning"]["data_profile_path"] = "data_info/profiler_summary.txt"
                state["Phase_1_Planning"]["llm_summary_path"] = "data_info/llm_summary.txt"
                state["Phase_1_Planning"]["blueprint_path"] = "data_info/blueprint.md" # Swapped extension
                state["Phase_1_Planning"]["blueprint"] = steps
                
                # Initialize Phase 2 Execution State (The Dynamic Graph Engine)
                execution_plan = {}
                for i, step_title in enumerate(steps, start=1):
                    execution_plan[f"step_{i}"] = {
                        "title": step_title,
                        "status": "PENDING",
                        "script_path": f"generated_scripts/step_{i}.py",
                        "attempts": 0
                    }
                
                state["Phase_2_Execution"]["status"] = "PENDING"
                state["Phase_2_Execution"]["current_step_id"] = "step_1"
                state["Phase_2_Execution"]["execution_plan"] = execution_plan
                
                with open(state_file, "w", encoding="utf-8") as f:
                    json.dump(state, f, indent=2)
            
            console.print("  [bold green]└─ Phase 1 Completed! All artifacts and session state saved.[/bold green]\n")
            return True
            
        except Exception as e:
            console.print(f"  [bold red]└─ Phase 1 Failed:[/bold red] {str(e)}\n")
            return False