import shutil
import os
import time
import json
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Header, Request
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from src.config.settings import settings
from src.utils.logger import logger

# Configure LiteLLM to drop unsupported params for Ollama
import litellm
litellm.drop_params = True

# Import New RAG Components
from src.rag.ingest import ingest_file, delete_document, clear_session_data
from src.rag.chain import get_rag_chain, get_langfuse_callback

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Server started (LangChain Mode). Model: {settings.LLM_MODEL}")
    yield

app = FastAPI(title=settings.APP_NAME, version=settings.VERSION, lifespan=lifespan)

# Mount static files
app.mount("/static", StaticFiles(directory="src/static"), name="static")

@app.get("/")
async def read_root():
    return FileResponse("src/static/index.html")

@app.get("/health")
async def health_check():
    return {
        "status": "ok", 
        "model": settings.LLM_MODEL, 
        "mode": "langchain_production",
        "stack": ["LangChain", "LiteLLM", "Langfuse"]
    }

@app.post("/ingest/")
async def ingest_document_endpoint(
    file: UploadFile = File(...),
    x_session_id: str = Header("default")
):
    """
    Ingest a document using the LangChain pipeline.
    """
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, f"{x_session_id}_{file.filename}")
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        logger.info(f"Ingesting file: {file.filename} for session: {x_session_id}")
        
        # Ingest using LangChain logic
        # This handles loading, splitting, and vector store addition
        num_chunks = ingest_file(file_path, x_session_id)
        
        return {
            "message": f"Successfully ingested {file.filename}", 
            "chunks_count": num_chunks,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Ingest failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/")
async def list_documents(x_session_id: str = Header("default")):
    """
    List uploaded documents for the session.
    """
    upload_dir = "uploads"
    files = []
    prefix = f"{x_session_id}_"
    
    if os.path.exists(upload_dir):
        for f in os.listdir(upload_dir):
            if f.startswith(prefix):
                # Return original filename
                files.append(f[len(prefix):])
                
    return {"files": files}

@app.delete("/documents/{filename}")
async def delete_document_endpoint(filename: str, x_session_id: str = Header("default")):
    try:
        logger.info(f"Deleting document: {filename}")
        
        # Delete from Vector Store (LangChain)
        delete_document(filename, x_session_id)
        
        # Delete Physical File
        upload_dir = "uploads"
        file_path = os.path.join(upload_dir, f"{x_session_id}_{filename}")
        if os.path.exists(file_path):
            os.remove(file_path)
            
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        return {"status": "error", "detail": str(e)}

@app.post("/documents/clear")
async def clear_all_endpoint(x_session_id: str = Header("default")):
    try:
        clear_session_data(x_session_id)
        
        # Clear uploads folder matches
        upload_dir = "uploads"
        prefix = f"{x_session_id}_"
        if os.path.exists(upload_dir):
            for f in os.listdir(upload_dir):
                if f.startswith(prefix):
                    os.remove(os.path.join(upload_dir, f))
                    
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.post("/set_model/")
async def set_model(model: str, x_session_id: str = Header("default")):
    # In this new architecture, model is set in config mainly, 
    # but we could theoretically update settings or inject it into get_llm
    # For now, we acknowledge it but warn it might default to .env
    return {"status": "success", "model": model, "note": "Model config is currently static in .env for LangChain stack"}

@app.get("/ask/")
async def ask_question_stream(query: str, x_session_id: str = Header("default")):
    """
    RAG Query Endpoint (Streaming)
    """
    from langchain_core.callbacks import StdOutCallbackHandler

    import uuid
    try:
        # Generate explicit trace ID
        trace_id = str(uuid.uuid4())
        
        # Pass trace_id so reranker can attach to it
        chain = get_rag_chain(x_session_id, trace_id=trace_id)
        
        # Define metadata for the trace
        trace_metadata = {
            "session_id": x_session_id,
            "endpoint": "/ask",
            "environment": "development"
        }
        trace_tags = ["rag-pipeline", "streaming", "beta"]
        
        # Pass tags and metadata to the callback generator
        # User requested matching trace name 'rag_streaming_generation' or similiar details
        langfuse_handler = get_langfuse_callback(
            x_session_id, 
            trace_name="rag_streaming_generation", 
            trace_id=trace_id,   # IMPORTANT: Match the ID used in the chain
            tags=trace_tags,
            metadata=trace_metadata
        )
        
        async def response_generator():
            config = {}
            callbacks = [StdOutCallbackHandler()]
            if langfuse_handler:
                callbacks.append(langfuse_handler)
            config["callbacks"] = callbacks
            
            # Use 'astream' to get events from the chain
            # The chain is: retrieval | stuff_documents
            sources_yielded = False
            
            start_time = time.time()
            full_response = ""
            
            try:
                async for chunk in chain.astream({"input": query}, config=config):
                    # 1. Handle Context (Sources)
                    # Depending on chain structure, 'context' usually appears in one of the first chunks
                    if "context" in chunk and not sources_yielded:
                        docs = chunk["context"]
                        # Extract filenames
                        sources = set()
                        prefix = f"{x_session_id}_"
                        for d in docs:
                            src = d.metadata.get("source", "unknown")
                            if src.startswith(prefix):
                                src = src[len(prefix):] # Clean prefix
                            sources.add(src)
                            
                        if sources:
                            yield f"Sources: {', '.join(sources)}\n\n"
                        sources_yielded = True
                    
                    # 2. Handle Answer
                    if "answer" in chunk:
                        content = chunk["answer"]
                        if content:
                            full_response += content
                            yield content
            finally:
                # Log final score manually if callback missed it
                elapsed = time.time() - start_time
                from src.services.langfuse_service import langfuse_service
                if langfuse_service.enabled:
                    # Latency Score
                    latency_score = 1.0 if elapsed < 2.0 else (0.7 if elapsed < 5.0 else 0.5)
                    langfuse_service.score(
                        trace_id=trace_id,
                        name="response_speed",
                        value=latency_score,
                        comment=f"Response generated in {elapsed:.2f}s"
                    )
                    
                    # Manual usage tracking backup
                    input_tokens = len(query) // 4 # Approximate
                    output_tokens = len(full_response) // 4
                    
                    usage = {
                        "input": input_tokens,
                        "output": output_tokens,
                        "total": input_tokens + output_tokens,
                        "unit": "TOKENS"
                    }
                    
                    # Explicitly track the generation to ensure it appears in dashboard
                    langfuse_service.track_generation(
                        trace_id=trace_id,
                        name="rag_streaming_generation", # The user specifically requested this name
                        model=settings.LLM_MODEL,
                        prompt=query,
                        completion=full_response,
                        metadata={
                            "stream": True, 
                            "response_time": elapsed,
                            "endpoint": "/ask"
                        },
                        usage=usage,
                        start_time=start_time,
                        end_time=time.time()
                    )
                    
        return StreamingResponse(response_generator(), media_type="text/plain")
        
    except Exception as e:
        logger.error(f"Ask failed: {e}")
        import traceback
        traceback.print_exc()
        error_msg = f"Error: {str(e)}"
        async def error_gen(): yield error_msg
        return StreamingResponse(error_gen(), media_type="text/plain")

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, x_session_id: str = Header("default")):
    """
    OpenAI-Compatible Endpoint for usage with generic clients
    """
    try:
        body = await request.json()
        messages = body.get("messages", [])
        if not messages: raise HTTPException(status_code=400)
        query = messages[-1]["content"] 
        
        chain = get_rag_chain(x_session_id)
        langfuse_handler = get_langfuse_callback(x_session_id, trace_name="openai_chat")
        
        async def openai_stream():
            config = {}
            if langfuse_handler: config["callbacks"] = [langfuse_handler]
            id = f"chatcmpl-{int(time.time())}"
            
            async for chunk in chain.astream({"input": query}, config=config):
                if "answer" in chunk:
                    data = {
                        "id": id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": settings.LLM_MODEL,
                        "choices": [{"index": 0, "delta": {"content": chunk["answer"]}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(data)}\n\n"
            
            yield "data: [DONE]\n\n"

        return StreamingResponse(openai_stream(), media_type="text/event-stream")
        
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.server:app", host="0.0.0.0", port=8000, reload=True)
