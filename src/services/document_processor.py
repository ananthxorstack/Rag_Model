import os
import fitz  # PyMuPDF
from src.core.types import DocumentChunk
from src.config.settings import settings

class DocumentProcessor:
    def process_file(self, file_path: str) -> list[DocumentChunk]:
        """
        Reads a file and returns a list of DocumentChunks.
        Supports .txt and .pdf
        """
        from src.utils.logger import logger
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Processing file: {file_path}")
        ext = os.path.splitext(file_path)[1].lower()
        
        content = ""
        if ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                page_count = 1
        elif ext == '.pdf':
            with fitz.open(file_path) as doc:
                page_count = len(doc)
                
                # Setup for optional vision
                from concurrent.futures import ThreadPoolExecutor
                from src.utils.logger import logger

                def process_image(img_info):
                    xref = img_info[0]
                    try:
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        # Strict Filter: Ignore tiny icons/lines (<200px)
                        if base_image["width"] < 200 or base_image["height"] < 200:
                            return None
                        
                        desc = self._analyze_image(image_bytes)
                        return f"[IMAGE DESCRIPTION: {desc}]" if desc else None
                    except Exception as e:
                        return None

                for page_index, page in enumerate(doc):
                    page_text = page.get_text()
                    page_content = ""
                    
                    page_content += f"--- Page {page_index + 1} ---\n"
                    page_content += page_text + "\n"
                    
                    # INTELLIGENT MODE: Only use Vision if text is missing/corrupted
                    # or very sparse (scanned pdfs often have junk chars or empty text)
                    clean_text = page_text.strip()
                    if len(clean_text) < 100:
                        logger.info(f"Page {page_index+1} has low text ({len(clean_text)} chars). Attempting Vision analysis...")
                        
                        image_descriptions = []
                        image_list = page.get_images(full=True)
                        
                        if image_list:
                            # Limit to top 2 images to stay fast
                            images_to_process = image_list[:2]
                            with ThreadPoolExecutor(max_workers=2) as executor:
                                results = list(executor.map(process_image, images_to_process))
                            image_descriptions = [r for r in results if r]
                        
                        if image_descriptions:
                            page_content += "\n[SCANNED CONTENT VIA VISION AI]:\n"
                            page_content += "\n".join(image_descriptions) + "\n"
                    
                    content += page_content
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        logger.info(f"Processed {page_count} pages from {file_path}")

        # Basic chunking (can be improved with specialized splitters)
        return self._chunk_text(content, source_id=os.path.basename(file_path)), page_count

    def _analyze_image(self, image_bytes: bytes) -> str:
        """
        Uses a Vision Language Model to describe the image content.
        """
        try:
            import ollama
            
            response = ollama.chat(
                model=settings.VISION_MODEL,
                messages=[
                    {
                        'role': 'user',
                        'content': 'Describe this image in detail. Extract any text, tables, signs, or signals visible.',
                        'images': [image_bytes]
                    }
                ]
            )
            return response['message']['content']
        except Exception as e:
            print(f"Vision analysis failed: {e}")
            return ""



    def _normalize_text(self, text: str) -> str:
        """
        Cleans and normalizes text from PDFs/documents.
        Removes excessive whitespace, fixes hyphenation, etc.
        """
        import re
        
        # 1. Fix hyphenated words at end of lines (e.g. "re-\nquire" -> "require")
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        
        # 2. Replace multiple newlines with a single newline (preserves paragraphs)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 3. Replace multiple spaces with single space
        text = re.sub(r'[ \t]+', ' ', text)
        
        # 4. Remove purely non-ascii garbage if necessary (optional, skipping for now to support utf8)
        
        return text.strip()

    def _chunk_text(self, text: str, source_id: str) -> list[DocumentChunk]:
        """
        Splits text into chunks using the advanced ChunkingService.
        Supports multiple strategies: recursive, sentence-based, fixed.
        """
        from src.services.chunking_service import ChunkingService, ChunkStrategy
        from src.utils.logger import logger
        
        # Normalize text first
        text = self._normalize_text(text)
        
        # Get chunking strategy from settings (default to recursive)
        strategy_name = getattr(settings, 'CHUNK_STRATEGY', 'recursive').lower()
        try:
            strategy = ChunkStrategy(strategy_name)
        except ValueError:
            logger.warning(f"Unknown chunk strategy '{strategy_name}', using recursive")
            strategy = ChunkStrategy.RECURSIVE
        
        # Initialize chunking service with settings
        chunking_service = ChunkingService(
            chunk_size=getattr(settings, 'CHUNK_SIZE', 500),
            chunk_overlap=getattr(settings, 'CHUNK_OVERLAP', 100),
            strategy=strategy
        )
        
        # Chunk the text
        chunk_dicts = chunking_service.chunk_text(
            text=text,
            metadata={"source": source_id}
        )
        
        # Validate chunks (remove too-small chunks)
        chunk_dicts = chunking_service.validate_chunks(chunk_dicts, min_chunk_size=50)
        
        # Log statistics
        stats = chunking_service.get_chunk_stats(chunk_dicts)
        logger.info(f"Chunking stats for {source_id}: {stats}")
        
        # Convert to DocumentChunk objects
        document_chunks = []
        for chunk_dict in chunk_dicts:
            document_chunks.append(DocumentChunk(
                content=chunk_dict["text"],
                source_id=source_id,
                metadata={
                    "source": source_id,
                    "chunk_index": chunk_dict["chunk_index"],
                    "chunk_strategy": chunk_dict["chunk_strategy"],
                    "chunk_size": chunk_dict["chunk_size"]
                }
            ))
        
        return document_chunks
