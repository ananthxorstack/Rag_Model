from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.config.settings import settings
from src.config.constants import SYSTEM_PROMPT_TEMPLATE
from src.rag.vector_store import get_vector_store
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
        streaming=True
    )

def get_langfuse_callback(session_id: str, trace_name="rag_chain", trace_id=None):
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
                trace_id=trace_id
            )
        except Exception as e:
            print(f"Warning: Could not init Langfuse callback: {e}")
            return None
    return None

def get_rag_chain(session_id: str):
    """
    Builds the RAG chain for a specific session.
    """
    llm = get_llm()
    vector_store = get_vector_store(session_id)
    
    # Setup Retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.RETRIEVAL_K}
    )
    
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
    vector_store = get_vector_store(session_id)
    return vector_store.as_retriever(search_kwargs={"k": settings.RETRIEVAL_K})
