import requests
import json
import time
from src.config.settings import settings
from src.config.constants import SYSTEM_PROMPT_TEMPLATE, GENERIC_ERROR_MESSAGE
from src.services.langfuse_service import langfuse_service
from src.utils.logger import print_trace

# Import litellm and configure it to drop unsupported parameters for Ollama
try:
    import litellm
    litellm.drop_params = True  # Drop unsupported parameters when calling Ollama
    print("LiteLLM configured: drop_params=True")
except ImportError:
    print("Warning: litellm not installed, using direct HTTP calls")

class LLMService:
    def __init__(self):
        self.model = settings.LLM_MODEL
        self.langfuse = langfuse_service
        self.base_url = settings.LITELLM_BASE_URL
        # Ensure base_url doesn't have trailing slash for clean appending
        if self.base_url.endswith("/"):
            self.base_url = self.base_url[:-1]

    def generate_response(self, context: str, question: str, trace_id: str = None) -> str:
        """
        Generates a response using the LiteLLM Proxy (OpenAI compatible).
        Tracks the interaction in Langfuse for analytics.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"}
        ]
        
        start_time = time.time()
        
        try:

            url = f"{self.base_url}/chat/completions"
            payload = {
                "model": self.model,
                "messages": messages
            }

            print_trace(
                "PREPARING REQUEST to LiteLLM", 
                f"URL: {url}\nModel: {self.model}\nPayload: {json.dumps(payload, indent=2)}"
            )
            
            # LiteLLM allows any key for local proxy usually, or check if one is set
            headers = {"Authorization": "Bearer any-key", "Content-Type": "application/json"}
            
            print_trace("SENDING HTTP POST", f"Destination: {self.base_url} (Local Proxy)\nWaiting for response...")
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            print_trace(
                "RECEIVED RESPONSE from LiteLLM", 
                f"Status: {response.status_code}\nRaw Data: {json.dumps(data, indent=2)}"
            )
            
            completion = data["choices"][0]["message"]["content"]
            
            elapsed_time = time.time() - start_time
            
            # Track in Langfuse
            if self.langfuse.enabled:
                print_trace(
                    "LOGGING TO LANGFUSE",
                    f"Trace ID: {trace_id}\nAction: track_generation\nProvider: litellm"
                )
                # Estimate Usage for Langfuse (Local models are free, but tracking volume is useful)
                usage = {
                    "input": len(json.dumps(messages)), 
                    "output": len(completion),
                    "total": len(json.dumps(messages)) + len(completion)
                }
                
                self.langfuse.track_generation(
                    trace_id=trace_id,
                    name="rag_generation",
                    model=self.model,
                    completion=completion,
                    metadata={
                        "question": question,
                        "context_length": len(context),
                        "response_time_seconds": elapsed_time,
                        "stream": False,
                        "provider": "litellm"
                    },
                    usage=usage,
                    prompt=messages
                )
            
            # LiteLLM 1.x+ sometimes needs explicit 'completion' model param or it may override return
            # But the error 'Invalid model name passed in model=string' suggests we are sending a malformed request
            # or the model alias isn't being picked up correctly by the proxy if we send 'model=llama-3.2-1b'
            
            # The error "Invalid model name passed in model=string" is weird. 'string'??
            # It usually happens if we pass "model": "string" literally or something.
            # But self.model IS "llama-3.2-1b".
            
            # Wait, look at the error log from user:
            # 400: {'error': '/chat/completions: Invalid model name passed in model=string. Call `/v1/models` ...'}
            # This suggests somewhere 'model' is being passed as the string 'string'?? 
            # OR the proxy thinks we are asking for a model named 'string'.
            
            # Actually, looking at the user log, the error happened LATER:
            # 17:05:40 ... Exception occured - 400: ... model=string ...
            
            # The earlier calls (17:05:05) SUCCEEDED with 200 OK despite the warning.
            # 17:05:05 - ... response model mismatch... Overriding...
            # INFO: ... "POST /chat/completions HTTP/1.1" 200 OK
            
            # So the chat worked? 
            # Techincally yes.
            
            # But wait, why did it fail later with 'model=string'? 
            # Maybe the user ran the test_litellm_connection.py script?
            # Or maybe the application code itself is somehow defaulting to 'string'??
            # Let's check settings.py again. LLM_MODEL is "llama-3.2-1b".
            
            # Ah, maybe the Swagger UI default example was executed?
            # Swagger UI often puts "string" as the default value for string fields.
            # If the user clicked "Execute" on Swagger without changing the model name from "string" to a real model, that would cause this exact error.
            
            # User request: "how to test whether it is working with litellms swagger"
            # User log shows: "17:05:40 ... model=string"
            # It is HIGHLY likely the user tried to execute the default Swagger payload.
            
            # I should explain this to the user.
            
            return completion
        except Exception as e:
            print(f"LLM Error (LiteLLM): {e}")
            if self.langfuse.enabled:
                self.langfuse.track_generation(
                    trace_id=trace_id,
                    name="rag_generation_error",
                    model=self.model,
                    prompt=messages,
                    completion=GENERIC_ERROR_MESSAGE,
                    metadata={"error": str(e)},
                    level="ERROR",
                    start_time=start_time,
                    end_time=time.time()
                )
            return GENERIC_ERROR_MESSAGE

    def generate_response_stream(self, context: str, question: str, trace_id: str = None):
        """
        Generates a streaming response using LiteLLM Proxy.
        """
        q_lower = question.strip().lower().replace("!", "").replace(".", "")
        if q_lower in ["hi", "hello", "hey", "greetings", "good morning", "good evening"]:
            user_content = f"QUESTION: {question}"
        else:
             user_content = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"

        # Try to use Managed Prompt from Langfuse
        langfuse_prompt = None
        messages = []
        
        if self.langfuse.enabled:
            langfuse_prompt = self.langfuse.get_prompt("rag-main-prompt")
            if not langfuse_prompt:
                 print_trace("LANGFUSE PROMPT", "WARNING: 'rag-main-prompt' not found in Langfuse. Prompt linking will be disabled.")
            
        if langfuse_prompt:
            # Compile messages from the managed template
            # compile() returns the list of messages with variables replaced
            try:
                messages = langfuse_prompt.compile(context=context, question=question)
                print_trace("LANGFUSE PROMPT", f"Loaded managed prompt 'rag-main-prompt' v{langfuse_prompt.version}")
            except Exception as e:
                print(f"Error compiling Langfuse prompt: {e}")
                # Fallback to manual construction
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE},
                    {"role": "user", "content": user_content}
                ]
        else:
            # Fallback
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE},
                {"role": "user", "content": user_content}
            ]
        
        # Use datetime for Langfuse compatibility
        from datetime import datetime, timezone
        start_time = datetime.now(timezone.utc)
        completion_start_time = None
        full_response = ""
        
        try:
            url = f"{self.base_url}/chat/completions"
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": True
            }
            
            # TRACE: Show the exact prompt being sent
            formatted_messages = ""
            for msg in messages:
                formatted_messages += f"\n--- ROLE: {msg['role'].upper()} ---\n{msg['content']}\n"
            
            print_trace(
                "FULL PROMPT CONSTRUCTION (What the LLM Sees)", 
                f"Model: {self.model}\n{formatted_messages}"
            )

            print_trace(
                "STARTING STREAM (LiteLLM)", 
                f"URL: {url}\nModel: {self.model}\nMode: Streaming"
            )
            
            headers = {"Authorization": "Bearer any-key", "Content-Type": "application/json"}
            
            with requests.post(url, json=payload, headers=headers, stream=True) as response:
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith('data: '):
                            data_str = line_str[6:] # strip 'data: '
                            if data_str.strip() == '[DONE]':
                                break
                            
                            try:
                                chunk_json = json.loads(data_str)
                                delta = chunk_json['choices'][0]['delta']
                                if 'content' in delta:
                                    if completion_start_time is None:
                                        completion_start_time = datetime.now(timezone.utc)
                                    content = delta['content']
                                    full_response += content
                                    yield content
                            except json.JSONDecodeError:
                                continue
            
            end_time = datetime.now(timezone.utc)
            elapsed_time = (end_time - start_time).total_seconds()
            
            # Track complete response
            if self.langfuse.enabled:
                # Estimate Usage
                input_tokens = len(json.dumps(messages)) // 4  # Roughly 4 chars per token
                output_tokens = len(full_response) // 4
                
                usage = {
                    "input": input_tokens, 
                    "output": output_tokens,
                    "total": input_tokens + output_tokens,
                    "unit": "TOKENS"  # Explicitly state unit
                }

                # Add explicit tags and metadata
                metadata = {
                    "question": question,
                    "response_time_seconds": elapsed_time,
                    "stream": True,
                    "provider": "litellm",
                    "environment": "development" # Tag environment
                }

                self.langfuse.track_generation(
                    trace_id=trace_id,
                    name="rag_streaming_generation",
                    model=self.model,
                    completion=full_response,
                    metadata=metadata,
                    usage=usage,
                    start_time=start_time,
                    completion_start_time=completion_start_time,
                    end_time=end_time,
                    level="DEFAULT",
                    prompt=messages,
                    prompt_object=langfuse_prompt
                )
                
                # Check if we should log a custom score for latency
                # < 2s = 1.0 (Excellent), < 5s = 0.7 (Good), > 5s = 0.5 (Slow)
                latency_score = 1.0 if elapsed_time < 2.0 else (0.7 if elapsed_time < 5.0 else 0.5)
                self.langfuse.score(
                    trace_id=trace_id,
                    name="response_speed",
                    value=latency_score,
                    comment=f"Response generated in {elapsed_time:.2f}s"
                )
                
        except Exception as e:
            print(f"LLM Stream Error (LiteLLM): {e}")
            if self.langfuse.enabled:
                self.langfuse.track_generation(
                    trace_id=trace_id,
                    name="rag_streaming_error",
                    model=self.model,
                    prompt=messages,
                    completion=GENERIC_ERROR_MESSAGE,
                    metadata={"error": str(e)},
                    start_time=start_time,
                    end_time=datetime.now(timezone.utc),
                    level="ERROR"
                )
            yield GENERIC_ERROR_MESSAGE

    def get_embedding(self, text: str, trace_id: str = None) -> list[float]:
        """
        Generates embeddings using LiteLLM (OpenAI compatible).
        """
        start_time = time.time()
        
        try:
            # LiteLLM/OpenAI Embeddings Endpoint
            url = f"{self.base_url}/embeddings"
            payload = {
                "model": settings.EMBEDDING_MODEL,
                "input": text
            }
            
            print_trace(
                "GENERATING EMBEDDINGS (LiteLLM)",
                f"Input Text: {text[:50]}...\nTarget URL: {url}\nModel: {settings.EMBEDDING_MODEL}"
            )
            
            headers = {"Authorization": "Bearer any-key", "Content-Type": "application/json"}
            
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            
            print_trace(
                "EMBEDDING RECEIVED",
                f"Status: {response.status_code}\nDimensions: {len(data['data'][0]['embedding'])}"
            )
            
            # OpenAI format: data[0].embedding
            embedding = data["data"][0]["embedding"]
            
            elapsed_time = time.time() - start_time
            
            if self.langfuse.enabled:
                self.langfuse.track_span(
                    trace_id=trace_id,
                    name="embedding_generation",
                    input_data={"text": text, "model": settings.EMBEDDING_MODEL},
                    output_data={"embedding_dimensions": len(embedding)},
                    metadata={
                        "processing_time_seconds": elapsed_time,
                        "provider": "litellm"
                    }
                )
            
            return embedding
        except Exception as e:
            if self.langfuse.enabled:
                self.langfuse.track_span(
                    trace_id=trace_id,
                    name="embedding_error",
                    input_data={"text": text},
                    output_data=None,
                    metadata={"error": str(e)}
                )
            raise Exception(f"Error generating embedding via LiteLLM: {str(e)}")
