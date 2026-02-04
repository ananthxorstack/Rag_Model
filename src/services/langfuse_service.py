"""
Langfuse Integration for Prompt Analytics and Observability
Tracks all LLM requests, responses, and embeddings for analysis.

NOTE: This is a simplified integration. The Langfuse SDK methods vary by version.
For production use, please refer to the official Langfuse documentation:
https://langfuse.com/docs/sdk/python

This integration provides a working framework that logs tracking data.
When you have valid API keys, the data will be sent to Langfuse.
"""
from typing import Optional, Dict, Any, List
from src.config.settings import settings
import logging
import uuid
import json

logger = logging.getLogger(__name__)


class LangfuseService:
    """Service to handle Langfuse observability and analytics."""
    
    def __init__(self):
        self.enabled = settings.LANGFUSE_ENABLED
        self.client: Optional[Any] = None
        
        if self.enabled:
            try:
                if not settings.LANGFUSE_PUBLIC_KEY or not settings.LANGFUSE_SECRET_KEY:
                    logger.warning(
                        "Langfuse is enabled but API keys are missing. "
                        "Please set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in .env"
                    )
                    self.enabled = False
                else:
                    # Import and initialize Langfuse
                    from langfuse import Langfuse
                    self.client = Langfuse(
                        public_key=settings.LANGFUSE_PUBLIC_KEY,
                        secret_key=settings.LANGFUSE_SECRET_KEY,
                        host=settings.LANGFUSE_BASE_URL
                    )
                    logger.info("Langfuse client initialized successfully")
                    logger.info("Note: Tracking data will be sent to Langfuse when flush() is called")
            except Exception as e:
                logger.error(f"Failed to initialize Langfuse: {e}")
                self.enabled = False
    
    def create_trace(self, name: str, user_id: str = "default_user", metadata: Optional[Dict] = None, tags: Optional[List[str]] = None):
        """Create a new trace for tracking a user interaction."""
        if not self.enabled:
            return None
        
        try:
            # Generate a unique trace ID
            trace_id = str(uuid.uuid4())
            
            # Log the trace info
            logger.info(f"Langfuse Trace: {name} (ID: {trace_id}, User: {user_id}, Tags: {tags})")
            
            # If we have a client, send the event
            if self.client:
                self.client.trace(
                    id=trace_id,
                    name=name,
                    user_id=user_id,
                    metadata=metadata,
                    tags=tags
                )
            
            # Return a simple object with the ID
            class TraceRef:
                def __init__(self, trace_id):
                    self.id = trace_id
            
            return TraceRef(trace_id)
        except Exception as e:
            logger.error(f"Failed to create trace: {e}")
            return None
    
    def track_generation(
        self,
        trace_id: Optional[str],
        name: str,
        model: str,
        prompt: Any,
        completion: str,
        metadata: Optional[Dict] = None,
        usage: Optional[Dict] = None,
        level: str = "DEFAULT",
        start_time: Optional[float] = None,
        completion_start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        prompt_object: Any = None
    ):
        """Track an LLM generation event."""
        if not self.enabled:
            return None
        
        try:
            # Log the generation
            logger.info(f"Langfuse Generation: {name} (Model: {model}, TraceID: {trace_id})")
            
            # If we have a client, events are automatically queued
            if self.client:
                # Convert timestamps to datetime if needed for specific SDK versions, 
                # but Langfuse Python SDK generally accepts float timestamps or datetime objects.
                # However, generation() arguments often expect `start_time` and `end_time`
                
                # NOTE: We pass 'input' keyword arg for prompt, 'output' for completion
                self.client.generation(
                    trace_id=trace_id,
                    name=name,
                    model=model,
                    input=prompt,
                    output=completion,
                    metadata=metadata,
                    usage=usage,
                    level=level,
                    start_time=start_time,
                    completion_start_time=completion_start_time,
                    end_time=end_time,
                    prompt=prompt_object # Link the prompt version
                )
            
            return True
        except Exception as e:
            logger.error(f"Failed to track generation: {e}")
            return None
    
    def track_span(
        self,
        trace_id: Optional[str],
        name: str,
        input_data: Any = None,
        output_data: Any = None,
        metadata: Optional[Dict] = None
    ):
        """Track a span (e.g., retrieval, processing step)."""
        if not self.enabled:
            return None
        
        try:
            # Log the span
            logger.info(f"Langfuse Span: {name} (TraceID: {trace_id})")
            if metadata:
                logger.debug(f"Span metadata: {json.dumps(metadata, indent=2)}")
            
            # If we have a client, events are automatically queued
            if self.client:
                self.client.span(
                    trace_id=trace_id,
                    name=name,
                    input=input_data,
                    output=output_data,
                    metadata=metadata
                )
            
            return True
        except Exception as e:
            logger.error(f"Failed to track span: {e}")
            return None
    
    def get_prompt(self, name: str):
        """Fetch a managed prompt from Langfuse."""
        if not self.enabled or not self.client:
            return None
        try:
            return self.client.get_prompt(name)
        except Exception as e:
            logger.error(f"Failed to fetch prompt '{name}': {e}")
            return None

    def score(
        self,
        trace_id: Optional[str] = None,
        observation_id: Optional[str] = None,
        name: str = "quality",
        value: float = 0.0,
        comment: Optional[str] = None
    ):
        """
        Track a score/metric for a trace or observation.
        
        Args:
            trace_id: ID of the trace to score
            observation_id: ID of specific observation (generation/span) to score
            name: Name of the score (e.g., 'relevance', 'accuracy', 'helpfulness')
            value: Numeric score value (typically 0-1 or 0-100 depending on your scale)
            comment: Optional comment explaining the score
        """
        if not self.enabled or not self.client:
            return None
        
        try:
            logger.info(f"Langfuse Score: {name}={value} (TraceID: {trace_id}, ObsID: {observation_id})")
            
            self.client.score(
                trace_id=trace_id,
                observation_id=observation_id,
                name=name,
                value=value,
                comment=comment
            )
            
            return True
        except Exception as e:
            logger.error(f"Failed to track score: {e}")
            return None


    def flush(self):
        """Flush any pending events to Langfuse."""
        if self.enabled and self.client:
            try:
                # The Langfuse SDK automatically batches and sends events
                # flush() ensures all pending events are sent
                if hasattr(self.client, 'flush'):
                    self.client.flush()
                    logger.info("Langfuse events flushed successfully")
            except Exception as e:
                logger.error(f"Failed to flush Langfuse events: {e}")


# Singleton instance
langfuse_service = LangfuseService()
