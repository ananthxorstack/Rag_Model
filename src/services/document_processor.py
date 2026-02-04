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
        Splits text into chunks safe for embeddings.
        Nomic Embed Text usually supports 2048 or 8192 tokens depending on version.
        However, the error 'input length exceeds context length' suggests we are sending massive chunks.
        We will use a character-based limit as a hard fallback to words.
        """
        # Normalize text first
        text = self._normalize_text(text)

        # Conservative limits
        # Assuming ~4 chars per token. 4000 chars ~= 1000 tokens.
        # This allows for larger context windows as requested.
        MAX_CHARS = 4000 
        
        chunk_size = settings.CHUNK_SIZE # This is currently treated as 'words' in the code.
        overlap = settings.CHUNK_OVERLAP # 'words'
        
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            # If a single word is massive (e.g. base64 string), hard split it
            if len(word) > MAX_CHARS:
                # Add current buffer if exists
                if current_chunk:
                    chunks.append(DocumentChunk(
                        content=" ".join(current_chunk),
                        source_id=source_id,
                        metadata={"source": source_id}
                    ))
                    current_chunk = []
                    current_length = 0
                
                # Split massive word
                for i in range(0, len(word), MAX_CHARS):
                    chunks.append(DocumentChunk(
                        content=word[i:i+MAX_CHARS],
                        source_id=source_id,
                        metadata={"source": source_id}
                    ))
                continue

            # Check if adding this word exceeds limits (either word count or char count)
            if (len(current_chunk) >= chunk_size) or (current_length + len(word) + 1 > MAX_CHARS):
                chunks.append(DocumentChunk(
                    content=" ".join(current_chunk),
                    source_id=source_id,
                    metadata={"source": source_id}
                ))
                
                # Overlap logic
                overlap_words = current_chunk[-overlap:] if overlap > 0 else []
                current_chunk = list(overlap_words)
                current_length = sum(len(w) + 1 for w in current_chunk)
            
            current_chunk.append(word)
            current_length += len(word) + 1
        
        if current_chunk:
            chunks.append(DocumentChunk(
                content=" ".join(current_chunk),
                source_id=source_id,
                metadata={"source": source_id}
            ))
            
        return chunks
