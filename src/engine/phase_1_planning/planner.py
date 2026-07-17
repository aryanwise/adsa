# Takes the data from the ingestion + profiler 
# Generates a step by step data science "pipeline" based on provided user input (problem prompt)
# Wait for the human approval (Y/n?) if Yes then proceeds further 
# If no: then ask for human feedback and rethinks the pipeline again

import os
from groq import Groq
from engine.core.config import Architect, GROQ_API_KEY

class PipelinePlanner:
    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("❌ GROQ_API_KEY is missing. Please check your .env file.")
            
        self.client = Groq(api_key=GROQ_API_KEY)
        
    def build_plan(self, profile_data: dict, user_objective: str) -> str:
        """Takes the data from the profiler and generates a step-by-step data science pipeline."""
        print(f"🧠 Thinking Model ({Architect.MODEL}) is analyzing your dataset and objective...")
        
        # Load prompts directly from our centralized config
        full_user_prompt = Architect.get_user_prompt(profile_data, user_objective)
        
        messages = [
            {"role": "system", "content": Architect.SYSTEM_PROMPT},
            {"role": "user", "content": full_user_prompt}
        ]
        
        attempt = 1
        while True:
            print(f"\n⏳ Generating step-by-step pipeline (Iteration {attempt})...")
            
            try:
                response = self.client.chat.completions.create(
                    model=Architect.MODEL, 
                    messages=messages,
                    temperature=0.3,  
                    max_tokens=1500
                )
                
                plan = response.choices[0].message.content
                
                print("\n======================================================================")
                print("                 🏗️ PROPOSED PIPELINE PLAN                            ")
                print("======================================================================")
                print(plan)
                print("======================================================================\n")
                
                # Human-in-the-loop approval gateway
                while True:
                    approval = input("👉 Do you approve this pipeline plan? (Y/n): ").strip().lower()
                    if approval in ['y', 'yes', 'n', 'no']:
                        break
                    print("⚠️ Invalid input. Please type 'Y' for yes or 'n' for no.")
                
                if approval in ['y', 'yes']:
                    print("\n✅ Plan approved! Locking blueprint for Phase 2 execution.")
                    return plan
                
                else:
                    feedback = input("\n📝 What needs to be changed? Provide your feedback: ").strip()
                    print("\n🔄 Rethinking the pipeline based on your feedback...")
                    
                    messages.append({"role": "assistant", "content": plan})
                    messages.append({"role": "user", "content": f"I reject the previous plan. Revise it strictly based on this feedback: {feedback}"})
                    attempt += 1
                    
            except Exception as e:
                print(f"❌ Cloud API Error during planning: {e}")
                raise e