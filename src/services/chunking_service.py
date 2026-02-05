"""
Advanced Chunking Service
Provides multiple intelligent document splitting strategies
"""

import logging
from typing import List, Dict, Any, Optional
from enum import Enum

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    SpacyTextSplitter,
    CharacterTextSplitter
)

logger = logging.getLogger(__name__)


class ChunkStrategy(str, Enum):
    """Available chunking strategies"""
    RECURSIVE = "recursive"
    SENTENCE = "sentence"
    FIXED = "fixed"
    SEMANTIC = "semantic"  # Future: Use embeddings to detect topic changes


class ChunkingService:
    """
    Manages document chunking with multiple strategies
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        strategy: ChunkStrategy = ChunkStrategy.RECURSIVE
    ):
        """
        Initialize chunking service
        
        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Number of overlapping characters between chunks
            strategy: Chunking strategy to use
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        
        # Initialize splitters
        self._init_splitters()
        
        logger.info(
            f"ChunkingService initialized: strategy={strategy}, "
            f"size={chunk_size}, overlap={chunk_overlap}"
        )
    
    def _init_splitters(self):
        """Initialize all text splitters"""
        
        # Recursive Character Splitter (LangChain)
        # Tries to split on paragraphs, then sentences, then words
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Sentence-based Splitter (spaCy)
        # Linguistically aware, respects sentence boundaries
        try:
            self.sentence_splitter = SpacyTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                pipeline="en_core_web_sm"
            )
        except Exception as e:
            logger.warning(f"Could not load spaCy splitter: {e}. Falling back to recursive.")
            self.sentence_splitter = None
        
        # Fixed-size Splitter (simple character-based)
        self.fixed_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator="\n"
        )
    
    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        strategy: Optional[ChunkStrategy] = None
    ) -> List[Dict[str, Any]]:
        """
        Split text into chunks using the specified strategy
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to each chunk
            strategy: Override default strategy for this call
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []
        
        # Use provided strategy or fall back to default
        chunk_strategy = strategy or self.strategy
        
        # Select appropriate splitter
        if chunk_strategy == ChunkStrategy.RECURSIVE:
            splitter = self.recursive_splitter
        elif chunk_strategy == ChunkStrategy.SENTENCE:
            splitter = self.sentence_splitter or self.recursive_splitter
        elif chunk_strategy == ChunkStrategy.FIXED:
            splitter = self.fixed_splitter
        else:
            logger.warning(f"Unknown strategy {chunk_strategy}, using recursive")
            splitter = self.recursive_splitter
        
        # Split the text
        try:
            chunks = splitter.split_text(text)
        except Exception as e:
            logger.error(f"Chunking failed with {chunk_strategy}: {e}")
            # Fallback to simple splitting
            chunks = self._fallback_chunk(text)
        
        # Build chunk objects with metadata
        chunk_objects = []
        for i, chunk_text in enumerate(chunks):
            chunk_obj = {
                "text": chunk_text,
                "chunk_index": i,
                "chunk_strategy": chunk_strategy.value,
                "chunk_size": len(chunk_text),
                "metadata": metadata or {}
            }
            chunk_objects.append(chunk_obj)
        
        logger.info(
            f"Chunked text into {len(chunk_objects)} chunks "
            f"using {chunk_strategy.value} strategy"
        )
        
        return chunk_objects
    
    def _fallback_chunk(self, text: str) -> List[str]:
        """
        Simple fallback chunking if all else fails
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.chunk_overlap
        
        return chunks
    
    def validate_chunks(
        self,
        chunks: List[Dict[str, Any]],
        min_chunk_size: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Validate and filter chunks
        
        Args:
            chunks: List of chunk dictionaries
            min_chunk_size: Minimum acceptable chunk size
            
        Returns:
            Filtered list of valid chunks
        """
        valid_chunks = []
        
        for chunk in chunks:
            chunk_text = chunk.get("text", "")
            
            # Skip empty or too-small chunks
            if len(chunk_text.strip()) < min_chunk_size:
                logger.debug(f"Skipping small chunk: {len(chunk_text)} chars")
                continue
            
            # Skip chunks that are just whitespace
            if not chunk_text.strip():
                continue
            
            valid_chunks.append(chunk)
        
        logger.info(
            f"Validated {len(valid_chunks)}/{len(chunks)} chunks "
            f"(min_size={min_chunk_size})"
        )
        
        return valid_chunks
    
    def get_chunk_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about chunks
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Dictionary of statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "avg_chunk_size": 0,
                "min_chunk_size": 0,
                "max_chunk_size": 0
            }
        
        chunk_sizes = [len(chunk.get("text", "")) for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "strategy": chunks[0].get("chunk_strategy", "unknown")
        }
