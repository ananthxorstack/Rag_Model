from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document

from src.config.settings import settings
from src.config.constants import SYSTEM_PROMPT_TEMPLATE
# Use the service-layer VectorStore which has the custom hybrid search logic
from src.services.vector_store import VectorStore
from src.services.llm_service import LLMService

from src.rag.rewriter import QueryRewriter
from src.rag.reranker import Reranker
from langfuse.callback import CallbackHandler
import os

def get_llm():
    """
    Returns the ChatOpenAI configured for LiteLLM.
    """
    return ChatOpenAI(
        model=settings.LLM_MODEL,
        openai_api_base=settings.LITELLM_BASE_URL,
        openai_api_key="sk-any",
        temperature=0.3, # Precision for RAG
        streaming=True,
        # IMPORTANT for Langfuse Usage tracking in streams:
        model_kwargs={"stream_options": {"include_usage": True}}
    )

def get_langfuse_callback(session_id: str, trace_name="rag_chain", trace_id=None, tags=None, metadata=None):
    """
    Returns a configured Langfuse Callback Handler.
    """
    if settings.LANGFUSE_ENABLED:
        try:
            return CallbackHandler(
                public_key=settings.LANGFUSE_PUBLIC_KEY,
                secret_key=settings.LANGFUSE_SECRET_KEY,
                host=settings.LANGFUSE_BASE_URL,
                session_id=session_id,
                trace_name=trace_name,
                trace_id=trace_id,
                tags=tags,
                metadata=metadata
            )
        except Exception as e:
            print(f"Warning: Could not init Langfuse callback: {e}")
            return None
    return None

def get_advanced_retriever(session_id: str, llm, trace_id: str = None):
    """
    Custom retriever that performs:
    Rewriting -> Hybrid Search -> Reranking
    """
    # Instantiate the custom VectorStore service
    # We need an LLMService instance for embedding generation
    llm_service = LLMService()
    vector_store = VectorStore(llm_service=llm_service, session_id=session_id)
    
    rewriter = QueryRewriter(llm)
    # Pass trace_id implicitly or explicitly if possible, 
    # but Reranker needs to know the trace ID to score.
    # We will pass it to the rerank method.
    reranker = Reranker(llm)
    
    def retrieve_fn(input_data):
        # input_data can be a dict (from chain) or string (direct call)
        query = input_data.get("input") if isinstance(input_data, dict) else str(input_data)
        
        # 1. Rewrite Query
        # We start a new trace span ideally, but for now simple print
        print(f"Original Query: {query}")
        refined_query = rewriter.rewrite(query)
        print(f"Rewritten Query: {refined_query}")
        
        # 2. Hybrid Search (Vector + Keyword)
        # Fetch more candidates (e.g. 2x K) for reranking
        candidate_count = settings.RETRIEVAL_K * 2
        raw_results = vector_store.search(refined_query, k=candidate_count, trace_id=trace_id)
        
        # 3. Rerank Results
        # Pass trace_id for scoring
        reranked_results = reranker.rerank(refined_query, raw_results, top_k=settings.RETRIEVAL_K, trace_id=trace_id)
        
        # Convert to LangChain Documents
        docs = [
            Document(page_content=res.content, metadata={"source": res.source, "score": res.score}) 
            for res in reranked_results
        ]
        return docs

    return RunnableLambda(retrieve_fn)

def get_rag_chain(session_id: str, trace_id: str = None):
    """
    Builds the RAG chain for a specific session.
    """
    llm = get_llm()
    
    # Setup Advanced Retriever
    retriever = get_advanced_retriever(session_id, llm, trace_id)
    
    # Prompt
    # LangChain Stuff chain expects 'context' variable
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT_TEMPLATE),
        ("system", "Here is the context to answer the user query:\n\n{context}"),
        ("user", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

def get_retriever(session_id: str):
    """Helper to get just the retriever for listing sources etc."""
    # Return the advanced retriever for consistency
    llm = get_llm()
    return get_advanced_retriever(session_id, llm)
