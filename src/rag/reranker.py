from langchain_openai import ChatOpenAI
from src.core.types import SearchResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class Reranker:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a relevance scoring assistant. You will be given a query and a document chunk. You must rate the relevance of the document to the query on a scale of 0 to 10. Output ONLY the number."),
            ("user", "Query: {query}\nDocument: {document}\nRelevance Score (0-10):")
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()

    
    def rerank(self, query: str, results: list[SearchResult], top_k: int = 5, trace_id: str = None) -> list[SearchResult]:
        """
        Uses LLM to rerank the search results based on relevance to the query.
        """
        from concurrent.futures import ThreadPoolExecutor
        from src.services.langfuse_service import langfuse_service
        
        if not results:
            return []

        # Helper to process one item
        def score_doc(res: SearchResult) -> SearchResult:
            try:
                # We limit content to avoid huge context window usage for scoring
                preview = res.content[:1000] 
                output = self.chain.invoke({"query": query, "document": preview}).strip()
                # Parse number
                import re
                match = re.search(r'\d+(\.\d+)?', output)
                if match:
                    # Normalize to 0-1
                    score_val = float(match.group()) / 10.0
                    res.score = score_val
                else:
                    res.score = 0.0
            except Exception as e:
                print(f"Reranking failed for doc: {e}")
                res.score = 0.0
            return res

        # Parallelize reranking to reduce latency
        with ThreadPoolExecutor(max_workers=5) as executor:
             scored_results = list(executor.map(score_doc, results))

        # Sort by new score
        scored_results.sort(key=lambda x: x.score, reverse=True)
        
        filtered_results = scored_results[:top_k]
        
        # LOGGING SCORES TO LANGFUSE
        if langfuse_service.enabled and trace_id:
            # We log the average relevance score of the top K results as 'context_relevance'
            if filtered_results:
                avg_score = sum(r.score for r in filtered_results) / len(filtered_results)
                langfuse_service.score(
                    trace_id=trace_id,
                    name="context_relevance",
                    value=avg_score,
                    comment=f"Average relevance of top {len(filtered_results)} retrieved chunks after reranking"
                )
                
                # Also log individual scores in a span or observation if needed, 
                # but 'score' is usually a top-level metric.
        
        return filtered_results
