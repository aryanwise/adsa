import os
from groq import Groq
from engine.core.config import Architect, GROQ_API_KEY
from rich.console import Console
from rich.markdown import Markdown

console = Console()

class PipelinePlanner:
    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("❌ GROQ_API_KEY is missing. Please check your .env file.")
            
        self.client = Groq(api_key=GROQ_API_KEY)
        
    def build_plan(self, profile_data: dict, user_objective: str) -> str:
        """Takes data from the profiler and generates a structural Markdown pipeline contract."""
        console.print(f"\n[bold status]🧠 Thinking Model ({Architect.MODEL}) is building your pipeline contract...[/bold status]")
        
        full_user_prompt = Architect.get_user_prompt(profile_data, user_objective)
        
        messages = [
            {"role": "system", "content": Architect.SYSTEM_PROMPT},
            {"role": "user", "content": full_user_prompt}
        ]
        
        attempt = 1
        while True:
            console.print(f"⏳ Architecting markdown layout framework (Iteration {attempt})...")
            
            try:
                response = self.client.chat.completions.create(
                    model=Architect.MODEL, 
                    messages=messages,
                    temperature=0.2,  # Dropped slightly for tighter formatting compliance
                    max_tokens=4096  # requirements list sits at the END; 2000 truncated it mid-package
                )
                
                plan = response.choices[0].message.content
                
                console.print("\n[bold cyan]======================================================================[/bold cyan]")
                console.print("                 🏗️  PROPOSED MARKDOWN PIPELINE CONTRACT              ")
                console.print("[bold cyan]======================================================================[/bold cyan]")
                # Use Rich to render the markdown perfectly onto the terminal screen
                console.print(Markdown(plan))
                console.print("[bold cyan]======================================================================\n[/bold cyan]")
                
                while True:
                    approval = input("👉 Do you approve this pipeline contract? (Y/n): ").strip().lower()
                    if approval in ['y', 'yes', 'n', 'no']:
                        break
                    console.print("[bold yellow]⚠️ Invalid input. Type 'Y' to approve or 'n' to reject.[/bold yellow]")
                
                if approval in ['y', 'yes']:
                    console.print("\n[bold green]✅ Contract approved! Locking blueprint.md for execution.[/bold green]")
                    return plan
                
                else:
                    feedback = input("\n📝 Provide structural revisions / feedback: ").strip()
                    console.print("\n🔄 Re-architecting blueprint layout based on feedback...")
                    
                    messages.append({"role": "assistant", "content": plan})
                    messages.append({"role": "user", "content": f"I reject the previous blueprint. Revise the markdown document based on this feedback: {feedback}"})
                    attempt += 1
                    
            except Exception as e:
                console.print(f"[bold red]❌ Cloud API Error during planning phase:[/bold red] {e}")
                raise e