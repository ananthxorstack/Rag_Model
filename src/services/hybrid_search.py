"""
Hybrid Search Service
Combines semantic vector search with BM25 keyword search
"""

import logging
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
import numpy as np

logger = logging.getLogger(__name__)


class HybridSearchService:
    """
    Manages hybrid search combining:
    1. Semantic search (vector similarity)
    2. BM25 keyword search
    """
    
    def __init__(
        self,
        semantic_weight: float = 0.7,
        bm25_weight: float = 0.3
    ):
        """
        Initialize hybrid search
        
        Args:
            semantic_weight: Weight for semantic scores (0-1)
            bm25_weight: Weight for BM25 scores (0-1)
        """
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight
        
        # BM25 index (will be built when documents are added)
        self.bm25_index = None
        self.documents = []
        self.doc_ids = []
        self.doc_metadata = []
        
        logger.info(
            f"HybridSearchService initialized: "
            f"semantic={semantic_weight}, bm25={bm25_weight}"
        )
    
    def build_bm25_index(
        self,
        documents: List[str],
        doc_ids: List[str],
        metadata: Optional[List[Dict]] = None
    ):
        """
        Build BM25 index from documents
        
        Args:
            documents: List of document texts
            doc_ids: List of document IDs
            metadata: Optional list of metadata dicts
        """
        if not documents:
            logger.warning("No documents provided for BM25 indexing")
            return
        
        self.documents = documents
        self.doc_ids = doc_ids
        self.doc_metadata = metadata or [{} for _ in documents]
        
        # Tokenize documents for BM25
        tokenized_docs = [doc.lower().split() for doc in documents]
        
        # Build BM25 index
        self.bm25_index = BM25Okapi(tokenized_docs)
        
        logger.info(f"BM25 index built with {len(documents)} documents")
    
    def search_bm25(
        self,
        query: str,
        k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search using BM25 keyword matching
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of results with scores
        """
        if not self.bm25_index:
            logger.warning("BM25 index not built yet")
            return []
        
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # Get top-k indices
        top_k_indices = np.argsort(scores)[::-1][:k]
        
        # Build results
        results = []
        for idx in top_k_indices:
            if scores[idx] > 0:  # Only include non-zero scores
                results.append({
                    "doc_id": self.doc_ids[idx],
                    "content": self.documents[idx],
                    "score": float(scores[idx]),
                    "metadata": self.doc_metadata[idx],
                    "search_type": "bm25"
                })
        
        logger.debug(f"BM25 search returned {len(results)} results")
        return results
    
    def combine_results(
        self,
        semantic_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Combine and rerank semantic and BM25 results
        
        Args:
            semantic_results: Results from vector search
            bm25_results: Results from BM25 search
            k: Number of final results to return
            
        Returns:
            Combined and reranked results
        """
        # Normalize scores to 0-1 range
        semantic_scores = self._normalize_scores(
            [r["score"] for r in semantic_results]
        )
        bm25_scores = self._normalize_scores(
            [r["score"] for r in bm25_results]
        )
        
        # Create score lookup by doc_id
        score_map = {}
        
        # Add semantic scores
        for i, result in enumerate(semantic_results):
            doc_id = result.get("doc_id", result.get("content", "")[:50])
            score_map[doc_id] = {
                "semantic": semantic_scores[i] if i < len(semantic_scores) else 0,
                "bm25": 0,
                "result": result
            }
        
        # Add BM25 scores
        for i, result in enumerate(bm25_results):
            doc_id = result.get("doc_id", result.get("content", "")[:50])
            if doc_id in score_map:
                score_map[doc_id]["bm25"] = bm25_scores[i] if i < len(bm25_scores) else 0
            else:
                score_map[doc_id] = {
                    "semantic": 0,
                    "bm25": bm25_scores[i] if i < len(bm25_scores) else 0,
                    "result": result
                }
        
        # Calculate hybrid scores
        hybrid_results = []
        for doc_id, scores in score_map.items():
            hybrid_score = (
                self.semantic_weight * scores["semantic"] +
                self.bm25_weight * scores["bm25"]
            )
            
            result = scores["result"].copy()
            result["hybrid_score"] = hybrid_score
            result["semantic_score"] = scores["semantic"]
            result["bm25_score"] = scores["bm25"]
            hybrid_results.append(result)
        
        # Sort by hybrid score
        hybrid_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        
        # Return top-k
        final_results = hybrid_results[:k]
        
        logger.info(
            f"Combined {len(semantic_results)} semantic + {len(bm25_results)} BM25 "
            f"results into {len(final_results)} hybrid results"
        )
        
        return final_results
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Normalize scores to 0-1 range
        
        Args:
            scores: List of scores
            
        Returns:
            Normalized scores
        """
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        return [
            (score - min_score) / (max_score - min_score)
            for score in scores
        ]
    
    def get_search_stats(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Get statistics about search results
        
        Args:
            results: List of search results
            
        Returns:
            Dictionary of statistics
        """
        if not results:
            return {
                "total_results": 0,
                "avg_hybrid_score": 0,
                "avg_semantic_score": 0,
                "avg_bm25_score": 0
            }
        
        return {
            "total_results": len(results),
            "avg_hybrid_score": np.mean([r.get("hybrid_score", 0) for r in results]),
            "avg_semantic_score": np.mean([r.get("semantic_score", 0) for r in results]),
            "avg_bm25_score": np.mean([r.get("bm25_score", 0) for r in results]),
            "top_score": results[0].get("hybrid_score", 0) if results else 0
        }
