import os
import sys
import time
from datetime import datetime, timezone

# Add src to path
sys.path.append(os.getcwd())

from src.services.langfuse_service import langfuse_service

def test_langfuse_integration():
    print("Testing Langfuse Integration...")
    
    if not langfuse_service.enabled:
        print("Langfuse is NOT enabled. Check .env")
        return

    print("1. Creating Trace...")
    # Test new input arg
    try:
        trace = langfuse_service.create_trace(
            name="debug_test_trace",
            user_id="debug_user",
            metadata={"test": "true"},
            input={"test_input": "hello"}
        )
        if trace:
            print(f"   Success. Trace ID: {trace.id}")
        else:
            print("   Failed to create trace (returned None)")
            return
            
        print("2. Updating Trace...")
        # Test update_trace
        try:
             res = langfuse_service.update_trace(
                 trace_id=trace.id,
                 output={"test_output": "world"}
             )
             print(f"   Update result: {str(res)}")
        except Exception as e:
            print(f"   Failed to update trace: {e}")

        print("3. Tracking Generation...")
        # Test generation with datetime
        start_time = datetime.now(timezone.utc)
        time.sleep(0.1)
        completion_start_time = datetime.now(timezone.utc)
        time.sleep(0.1)
        end_time = datetime.now(timezone.utc)
        
        try:
            langfuse_service.track_generation(
                trace_id=trace.id,
                name="debug_generation_test",
                model="test-model",
                prompt=[{"role": "user", "content": "hello"}],
                completion="world",
                metadata={"foo": "bar"},
                usage={"input": 1, "output": 1, "total": 2},
                start_time=start_time,
                completion_start_time=completion_start_time,
                end_time=end_time
            )
            print("   Success.")
        except Exception as e:
            print(f"   Failed to track generation: {e}")

        print("4. Flushing...")
        langfuse_service.flush()
        print("Done.")

    except Exception as e:
        print(f"FATAL ERROR: {e}")

if __name__ == "__main__":
    test_langfuse_integration()
