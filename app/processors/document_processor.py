from pathlib import Path
from PyPDF2 import PdfReader
from transformers import pipeline

class DocumentProcessor:
    """Handles document processing and analysis."""
    
    def __init__(self):
        """Initialize the document processor with necessary ML pipelines."""
        self.summarizer = pipeline('summarization')
        
    def _extract_text_from_pdf(self, file_path: Path) -> str:
        """Extract text content from a PDF file."""
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    
    def process_pdf(self, file_path: Path) -> dict:
        """Process a PDF document and extract key information."""
        # Extract text
        text = self._extract_text_from_pdf(file_path)
        
        # Generate summary (chunk text if too long)
        max_chunk_size = 1024
        chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
        
        summaries = []
        for chunk in chunks:
            if len(chunk.strip()) > 100:  # Only summarize substantial chunks
                summary = self.summarizer(chunk, max_length=130, min_length=30)
                summaries.append(summary[0]['summary_text'])
        
        # Basic metadata
        metadata = {
            'total_length': len(text),
            'num_pages': len(reader.pages),
            'file_type': 'PDF'
        }
        
        return {
            'summary': ' '.join(summaries),
            'metadata': metadata,
            'success': True
        }
