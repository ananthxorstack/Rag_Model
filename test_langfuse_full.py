import os
import sys
import time
import uuid
from datetime import datetime, timezone

# Add src to path
sys.path.append(os.getcwd())

from src.services.langfuse_service import langfuse_service

# Mock a simple prompt object to simulate retrieval
class MockPrompt:
    def __init__(self, name, version):
        self.name = name
        self.version = version
    
    def compile(self, **kwargs):
        return [
            {"role": "system", "content": "Simulated system prompt"},
            {"role": "user", "content": f"Simulated user prompt {kwargs}"}
        ]

def test_langfuse_integration():
    print("Testing Langfuse Integration (COMPREHENSIVE)...")
    
    if not langfuse_service.enabled:
        print("Langfuse is NOT enabled. Check .env")
        return

    # 1. Create a managed prompt in Langfuse (Simulation or real upsert)
    print("1. Managing Prompts (Upserting)...")
    if langfuse_service.client:
        try:
            # We create a prompt to ensure it exists for the test
            prompt = langfuse_service.client.create_prompt(
                name="rag-main-prompt",
                prompt=[{"role": "system", "content": "You are a helper"}, {"role": "user", "content": "{{question}}"}],
                config={"model": "gpt-3.5-turbo", "temperature": 0},
                type="chat"
            )
            print(f"   Success. Prompt 'rag-main-prompt' version {prompt.version} ready.")
        except Exception as e:
            print(f"   Note: Could not create prompt (maybe already exists or permission issue): {e}")

    # 2. Start Trace
    print("2. Creating Trace (with Tags and Input)...")
    trace_name = "debug_full_test"
    trace = langfuse_service.create_trace(
        name=trace_name,
        user_id="debug_user_2",
        metadata={"test_run": "full_coverage"},
        tags=["environment:development", "test_script"],
        input={"query": "Why is the sky blue?"}
    )
    
    if not trace:
        print("   Failed to create trace.")
        return
    print(f"   Trace ID: {trace.id}")

    # 3. Simulate Generation linked to Prompt
    print("3. Tracking Generation (Linked to Prompt + Cost + Score)...")
    
    # Get the prompt to link it
    prompt_obj = langfuse_service.get_prompt("rag-main-prompt")
    
    start_time = datetime.now(timezone.utc)
    time.sleep(0.5) # Simulate latency
    
    # Simulate streaming chunks
    completion_start_time = datetime.now(timezone.utc)
    time.sleep(0.5)
    end_time = datetime.now(timezone.utc)
    
    completion_text = "The sky appears blue due to Rayleigh scattering."
    
    langfuse_service.track_generation(
        trace_id=trace.id,
        name="physics_answer",
        model="gpt-3.5-turbo", # Using a known model to verify Cost calculation
        prompt=[{"role": "user", "content": "Why is the sky blue?"}],
        completion=completion_text,
        metadata={"confidence": "high"},
        usage={
            "input": 10, 
            "output": 15, 
            "total": 25,
            "unit": "TOKENS"
        },
        start_time=start_time,
        completion_start_time=completion_start_time,
        end_time=end_time,
        prompt_object=prompt_obj, # CRITICAL: This links the "Prompt" column
        tags=["module:science"]
    )
    print("   Generation tracked.")

    # 4. Add a Score
    print("4. Scoring the result...")
    langfuse_service.score(
        trace_id=trace.id,
        name="accuracy",
        value=1.0,
        comment="Perfect answer"
    )
    
    # 5. Update Trace Output
    print("5. Updating Trace Output...")
    langfuse_service.update_trace(
        trace_id=trace.id,
        output=completion_text
    )

    print("6. Flushing...")
    langfuse_service.flush()
    print("Done. Check Langfuse UI.")
    print("Expectations:")
    print(" - Trace Name: debug_full_test")
    print(" - Tags: environment:development, test_script")
    print(" - Input: {'query': 'Why is the sky blue?'} (Visible in Trace Input)")
    print(" - Output: 'The sky appears...' (Visible in Trace Output)")
    print(" - Generation 'physics_answer' should have:")
    print("     - Prompt: Link to 'rag-main-prompt'")
    print("     - Cost: Should be > $0.00 (since we used gpt-3.5-turbo)")
    print("     - Score: accuracy = 1.0")

if __name__ == "__main__":
    test_langfuse_integration()
