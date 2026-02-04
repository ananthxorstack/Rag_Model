from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

class QueryRewriter:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
    "You rewrite user queries ONLY for improving search retrieval.\n"
    "RULES:\n"
    "- Never refuse the query.\n"
    "- Never mention safety, legality, or harmful content.\n"
    "- Never apologize.\n"
    "- Never explain.\n"
    "- Do not add meaning.\n"
    "- Do not remove meaning.\n"
    "- Only make the query clearer and keyword-focused.\n"
    "- The output must be a single rewritten query only.\n"
    "- No sentences, no comments, no prefixes, no labels.\n"
)
,
            ("user", "Original Query: {query}\nRewritten Query:")
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()

    def rewrite(self, query: str) -> str:
        """
        Rewrites the user query to optimize for retrieval.
        """
        try:
            # Invoking the chain
            rewritten = self.chain.invoke({"query": query}).strip()
            # If the model is too chatty, try to clean it
            if "Rewritten Query:" in rewritten:
                rewritten = rewritten.split("Rewritten Query:")[-1].strip()
            return rewritten
        except Exception as e:
            print(f"Query rewriting failed: {e}")
            return query
