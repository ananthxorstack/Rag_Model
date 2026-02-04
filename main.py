import argparse
import sys
from src.config.settings import settings
from src.services.llm_service import LLMService
from src.services.vector_store import VectorStore
from src.services.document_processor import DocumentProcessor

def main():
    parser = argparse.ArgumentParser(description=f"{settings.APP_NAME} CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest Command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a document")
    ingest_parser.add_argument("file_path", help="Path to the file to ingest (.txt or .pdf)")

    # Ask Command
    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("query", help="The question to ask")

    args = parser.parse_args()

    # Initialize Services
    try:
        llm_service = LLMService()
        vector_store = VectorStore(llm_service)
        doc_processor = DocumentProcessor()
    except Exception as e:
        print(f"Initialization Error: {e}")
        print("Ensure you have Ollama running and the models pulled.")
        sys.exit(1)

    if args.command == "ingest":
        print(f"Processing {args.file_path}...")
        try:
            chunks, page_count = doc_processor.process_file(args.file_path)
            vector_store.add_documents(chunks)
            print(f"Successfully added {len(chunks)} chunks from {page_count} pages of {args.file_path}.")
        except Exception as e:
            print(f"Error ingesting file: {e}")

    elif args.command == "ask":
        print(f"Searching for answer to: '{args.query}'...")
        # 1. Retrieve
        results = vector_store.search(args.query, k=settings.RETRIEVAL_K)
        
        if not results:
            print("No relevant documents found.")
            return

        # 2. Construct Context
        context_text = "\n\n".join([r.content for r in results])
        
        # 3. Generate Answer
        answer = llm_service.generate_response(context_text, args.query)
        
        print("\n--- Answer ---")
        print(answer)
        print("\n--- Sources ---")
        for res in results:
            print(f"- {res.source}")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
