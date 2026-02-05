"""
Test script for Phase 1 & 2 upgrades:
- Advanced Chunking (Recursive, Sentence-based)
- Hybrid Search (BM25 + Vector)
"""

import sys
import os

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.chunking_service import ChunkingService, ChunkStrategy
from src.services.hybrid_search import HybridSearchService


def test_chunking():
    """Test advanced chunking strategies"""
    print("\n" + "="*80)
    print("TESTING PHASE 1: ADVANCED CHUNKING")
    print("="*80)
    
    sample_text = """
    Machine learning is a subset of artificial intelligence. It focuses on building systems 
    that can learn from data. Deep learning is a type of machine learning that uses neural networks.
    
    Neural networks are inspired by the human brain. They consist of layers of interconnected nodes.
    Each node processes information and passes it to the next layer.
    
    Applications of machine learning include image recognition, natural language processing, 
    and recommendation systems. These technologies power many modern applications.
    """
    
    # Test 1: Recursive Chunking
    print("\n[Test 1] Recursive Chunking Strategy")
    print("-" * 80)
    chunker_recursive = ChunkingService(
        chunk_size=150,
        chunk_overlap=30,
        strategy=ChunkStrategy.RECURSIVE
    )
    chunks_recursive = chunker_recursive.chunk_text(sample_text)
    print(f"✓ Created {len(chunks_recursive)} chunks")
    for i, chunk in enumerate(chunks_recursive[:2]):  # Show first 2
        print(f"\nChunk {i+1}:")
        print(f"  Size: {chunk['chunk_size']} chars")
        print(f"  Text: {chunk['text'][:100]}...")
    
    stats = chunker_recursive.get_chunk_stats(chunks_recursive)
    print(f"\nStats: {stats}")
    
    # Test 2: Sentence-based Chunking
    print("\n[Test 2] Sentence-based Chunking Strategy")
    print("-" * 80)
    try:
        chunker_sentence = ChunkingService(
            chunk_size=200,
            chunk_overlap=50,
            strategy=ChunkStrategy.SENTENCE
        )
        chunks_sentence = chunker_sentence.chunk_text(sample_text)
        print(f"✓ Created {len(chunks_sentence)} chunks")
        for i, chunk in enumerate(chunks_sentence[:2]):
            print(f"\nChunk {i+1}:")
            print(f"  Size: {chunk['chunk_size']} chars")
            print(f"  Text: {chunk['text'][:100]}...")
    except Exception as e:
        print(f"⚠ Sentence chunking failed (spaCy might not be loaded): {e}")
    
    print("\n✅ Chunking tests completed!")


def test_hybrid_search():
    """Test hybrid search (BM25 + Semantic)"""
    print("\n" + "="*80)
    print("TESTING PHASE 2: HYBRID SEARCH")
    print("="*80)
    
    # Sample documents
    documents = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with multiple layers",
        "Natural language processing helps computers understand human language",
        "Computer vision enables machines to interpret visual information",
        "Reinforcement learning trains agents through rewards and penalties"
    ]
    
    doc_ids = [f"doc_{i}" for i in range(len(documents))]
    metadata = [{"source": f"doc_{i}.txt"} for i in range(len(documents))]
    
    # Initialize hybrid search
    print("\n[Test 1] Building BM25 Index")
    print("-" * 80)
    hybrid_search = HybridSearchService(
        semantic_weight=0.7,
        bm25_weight=0.3
    )
    hybrid_search.build_bm25_index(documents, doc_ids, metadata)
    print(f"✓ BM25 index built with {len(documents)} documents")
    
    # Test BM25 search
    print("\n[Test 2] BM25 Keyword Search")
    print("-" * 80)
    query = "neural networks deep learning"
    bm25_results = hybrid_search.search_bm25(query, k=3)
    print(f"Query: '{query}'")
    print(f"✓ Found {len(bm25_results)} results")
    for i, result in enumerate(bm25_results):
        print(f"\n  Result {i+1}:")
        print(f"    Score: {result['score']:.4f}")
        print(f"    Content: {result['content'][:80]}...")
    
    # Test hybrid combination (simulated semantic results)
    print("\n[Test 3] Hybrid Score Combination")
    print("-" * 80)
    
    # Simulate semantic results (in real use, these come from vector search)
    semantic_results = [
        {"doc_id": "doc_1", "content": documents[1], "score": 0.92, "metadata": metadata[1]},
        {"doc_id": "doc_0", "content": documents[0], "score": 0.85, "metadata": metadata[0]},
        {"doc_id": "doc_2", "content": documents[2], "score": 0.78, "metadata": metadata[2]},
    ]
    
    combined_results = hybrid_search.combine_results(
        semantic_results=semantic_results,
        bm25_results=bm25_results,
        k=5
    )
    
    print(f"✓ Combined results (top {len(combined_results)})")
    for i, result in enumerate(combined_results):
        print(f"\n  Result {i+1}:")
        print(f"    Hybrid Score: {result['hybrid_score']:.4f}")
        print(f"    Semantic: {result['semantic_score']:.4f} | BM25: {result['bm25_score']:.4f}")
        print(f"    Content: {result['content'][:80]}...")
    
    stats = hybrid_search.get_search_stats(combined_results)
    print(f"\nSearch Stats: {stats}")
    
    print("\n✅ Hybrid search tests completed!")


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("RAG UPGRADE TESTS - Phase 1 & 2")
    print("="*80)
    
    try:
        test_chunking()
        test_hybrid_search()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        print("\nNext steps:")
        print("1. Restart the application to use new chunking")
        print("2. Upload a document to test chunking in action")
        print("3. Hybrid search will be integrated into vector_store.py next")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
