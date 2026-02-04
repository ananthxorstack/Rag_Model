import chromadb
from concurrent.futures import ThreadPoolExecutor
from chromadb.config import Settings as ChromaSettings
from src.config.settings import settings
from src.core.types import DocumentChunk, SearchResult
from src.services.llm_service import LLMService
from src.services.langfuse_service import langfuse_service
import time

class VectorStore:
    def __init__(self, llm_service: LLMService, session_id: str = None):
        # Use PersistentClient for data persistence across restarts
        self.client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        
        # If session_id is provided, use a unique collection name
        self.collection_name = f"{settings.COLLECTION_NAME}_{session_id}" if session_id else settings.COLLECTION_NAME
        print(f"[VectorStore] Initializing with collection: {self.collection_name}")

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.llm_service = llm_service
        self.langfuse = langfuse_service

    def add_documents(self, chunks: list[DocumentChunk], trace_id: str = None):
        """
        Embeds and adds documents to the vector store.
        Tracks the embedding process in Langfuse.
        """
        if not chunks:
            return

        start_time = time.time()
        
        from src.utils.logger import logger
        from tqdm import tqdm

        # Log start
        logger.info(f"Starting ingestion of {len(chunks)} chunks...")
        
        ids = [f"{chunk.source_id}_{i}" for i, chunk in enumerate(chunks)]
        documents = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        # Generate embeddings in parallel with progress bar
        embeddings = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            # We use executor.map but wrap it in tqdm to visualize progress.
            # Since map returns an iterator, converting it to list under tqdm works.
            results = executor.map(
                lambda doc: self.llm_service.get_embedding(doc, trace_id=trace_id),
                documents
            )
            # Use tqdm to track progress of the iterator
            embeddings = list(tqdm(results, total=len(documents), desc="Generatng Embeddings", unit="chunk"))

        logger.info(f"Embeddings generated. Adding to VectorStore...")
        
        try:
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
        except Exception as e:
            # If collection was deleted externally or state is stale
            if "does not exist" in str(e):
                print(f"[VectorStore] Collection missing during add, recreating: {self.collection_name}")
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                # Retry add
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
            else:
                raise e
        
        elapsed_time = time.time() - start_time
        
        # Track document ingestion in Langfuse
        if self.langfuse.enabled:
            # Extract sources from chunk metadata
            sources = list(set([chunk.metadata.get('source', chunk.source_id) for chunk in chunks]))
            
            self.langfuse.track_span(
                trace_id=trace_id,
                name="document_ingestion",
                input_data={
                    "num_chunks": len(chunks),
                    "sources": sources
                },
                output_data={
                    "num_embeddings_created": len(embeddings),
                    "collection_name": self.collection_name
                },
                metadata={
                    "processing_time_seconds": elapsed_time,
                    "avg_chunk_length": sum(len(doc) for doc in documents) / len(documents) if documents else 0
                }
            )

    def search(self, query: str, k: int = 3, trace_id: str = None) -> list[SearchResult]:
        """
        Hybrid search combining semantic vector search with keyword matching.
        Tracks retrieval in Langfuse for analytics.
        """
        start_time = time.time()
        
        # Step 1: Semantic Vector Search
        query_embedding = self.llm_service.get_embedding(query, trace_id=trace_id)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        search_results = []
        retrieved_ids = set()
        
        from src.utils.logger import print_trace

        if results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                doc_id = results['ids'][0][i] if results.get('ids') else f"doc_{i}"
                retrieved_ids.add(doc_id)
                
                # Handle cases where metadata might be None
                meta = results['metadatas'][0][i] if results['metadatas'] else {}
                source = meta.get('source', 'unknown') if meta else 'unknown'
                
                search_results.append(SearchResult(
                    content=doc,
                    score=0.0,
                    source=source
                ))
            
            # Trace Retrieval Details for User
            trace_details = f"Query: '{query}'\nTop k: {k}\n\n"
            for idx, res in enumerate(search_results):
                preview = res.content[:200].replace('\n', ' ') + "..."
                trace_details += f"[{idx+1}] Source: {res.source} | Content: {preview}\n"
            
            print_trace("VECTOR RETRIEVAL RESULTS (Context Found)", trace_details)
        
        # Step 2: Keyword Matching Fallback
        # If query contains specific words, also do keyword search
        stop_words = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"}
        
        raw_query_words = set(query.lower().split())
        # Filter out stop words to find 'keywords' (e.g., "what is the glossary" -> {"glossary"})
        keywords = {w for w in raw_query_words if w not in stop_words}
        
        # Always run fallback if we have valid keywords, but limit to reasonable number of keywords
        # to avoid scanning for "the" in every document.
        if len(keywords) > 0 and len(keywords) <= 10:  
            try:
                all_docs = self.collection.get()
                if all_docs and all_docs.get('documents'):
                    keyword_matches = []
                    for i, doc in enumerate(all_docs['documents']):
                        doc_id = all_docs['ids'][i]
                        # Skip if already retrieved by vector search
                        if doc_id in retrieved_ids:
                            continue
                        
                        # Check if any keyword appears in the document
                        doc_lower = doc.lower()
                        if any(word in doc_lower for word in keywords):
                            meta = all_docs['metadatas'][i] if all_docs.get('metadatas') else {}
                            source = meta.get('source', 'unknown') if meta else 'unknown'
                            
                            # Calculate simple keyword score (number of matching words)
                            match_count = sum(1 for word in keywords if word in doc_lower)
                            keyword_matches.append({
                                'content': doc,
                                'source': source,
                                'match_count': match_count
                            })
                    
                    # Sort by match count and add top matches to results
                    keyword_matches.sort(key=lambda x: x['match_count'], reverse=True)
                    for match in keyword_matches[:k]:  # Add up to k keyword matches
                        search_results.append(SearchResult(
                            content=match['content'],
                            score=0.0,
                            source=match['source']
                        ))
                        print(f"[VectorStore] Added keyword match for query: {query}")
            except Exception as e:
                print(f"[VectorStore] Keyword search failed: {e}")
        
        elapsed_time = time.time() - start_time
        
        # Track retrieval in Langfuse
        if self.langfuse.enabled:
            self.langfuse.track_span(
                trace_id=trace_id,
                name="vector_retrieval",
                input_data={
                    "query": query,
                    "k": k
                },
                output_data={
                    "num_results": len(search_results),
                    "sources": [r.source for r in search_results]
                },
                metadata={
                    "retrieval_time_seconds": elapsed_time,
                    "collection_name": self.collection_name
                }
            )
                
        return search_results

    def delete_document(self, filename: str):
        """
        Deletes all chunks associated with a specific file source.
        """
        try:
            # Get current count before deletion
            initial_count = self.collection.count()
            print(f"[VectorStore] Deleting document: {filename}")
            print(f"[VectorStore] Collection count before deletion: {initial_count}")
            
            # Verify what we're about to delete by querying first
            all_data = self.collection.get()
            if all_data and all_data.get('metadatas'):
                matching_ids = []
                for i, meta in enumerate(all_data['metadatas']):
                    if meta and meta.get('source') == filename:
                        matching_ids.append(all_data['ids'][i])
                print(f"[VectorStore] Found {len(matching_ids)} chunks to delete with source: {filename}")
                
                if matching_ids:
                    # Delete by IDs instead of where clause for more reliability
                    self.collection.delete(ids=matching_ids)
                else:
                    print(f"[VectorStore] Warning: No chunks found with source: {filename}")
            
            # Verify deletion
            final_count = self.collection.count()
            deleted_count = initial_count - final_count
            print(f"[VectorStore] Collection count after deletion: {final_count}")
            print(f"[VectorStore] Deleted {deleted_count} chunks for document: {filename}")
            
            # For debugging: list remaining unique sources
            all_data = self.collection.get()
            if all_data and all_data.get('metadatas'):
                remaining_sources = set([meta.get('source', 'unknown') for meta in all_data['metadatas'] if meta])
                print(f"[VectorStore] Remaining sources in collection: {remaining_sources}")
            else:
                print(f"[VectorStore] Collection is now empty")
                
        except Exception as e:
            print(f"[VectorStore] Error deleting document {filename}: {e}")
            import traceback
            traceback.print_exc()
    
    def clear_all(self):
        """
        Completely clears all documents from the vector store.
        Useful for resetting the knowledge base.
        """
        try:
            # Safer way: delete all IDs instead of dropping collection
            # This avoids invalidating the collection object reference held by self.collection
            
            # Use try-catch for the get() in case collection is already gone
            try:
                all_data = self.collection.get()
                all_ids = all_data['ids'] if all_data else []
            except Exception as e:
                if "does not exist" in str(e):
                    print("[VectorStore] Collection does not exist. Recreating...")
                    self.collection = self.client.get_or_create_collection(
                        name=self.collection_name,
                        metadata={"hnsw:space": "cosine"}
                    )
                    return
                else:
                    raise e

            if all_ids:
                print(f"[VectorStore] Deleting {len(all_ids)} chunks...")
                self.collection.delete(ids=all_ids)
                print(f"[VectorStore] Successfully cleared all documents")
            else:
                 print("[VectorStore] Collection already empty.")

        except Exception as e:
            print(f"[VectorStore] Error clearing collection: {e}")
            # Fallback: Force recreate
            try:
                self.client.delete_collection(name=self.collection_name)
            except:
                pass 
            
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            print(f"[VectorStore] Error clearing collection: {e}")
            import traceback
            traceback.print_exc()
