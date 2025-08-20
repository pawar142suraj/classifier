"""
Document loader for various file formats.
"""

import json
from pathlib import Path
from typing import Dict, Any, Union
import logging

logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    Utility class for loading documents in various formats.
    """
    
    def __init__(self):
        """Initialize document loader."""
        self.supported_formats = {'.txt', '.json', '.pdf', '.docx', '.md'}
        logger.info("Initialized DocumentLoader")
    
    def load(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load a document from file path.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary containing document data
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        suffix = file_path.suffix.lower()
        
        if suffix == '.txt':
            return self._load_text(file_path)
        elif suffix == '.json':
            return self._load_json(file_path)
        elif suffix == '.pdf':
            return self._load_pdf(file_path)
        elif suffix == '.docx':
            return self._load_docx(file_path)
        elif suffix == '.md':
            return self._load_markdown(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def _load_text(self, file_path: Path) -> Dict[str, Any]:
        """Load plain text file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return {
            "content": content,
            "file_path": str(file_path),
            "file_type": "text",
            "metadata": {
                "size": len(content),
                "lines": content.count('\n') + 1
            }
        }
    
    def _load_json(self, file_path: Path) -> Dict[str, Any]:
        """Load JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # If it's already a document structure, return it
        if isinstance(data, dict) and ("content" in data or "text" in data):
            return data
        
        # Otherwise, wrap it
        return {
            "content": json.dumps(data, indent=2),
            "file_path": str(file_path),
            "file_type": "json",
            "raw_data": data,
            "metadata": {
                "size": len(str(data))
            }
        }
    
    def _load_pdf(self, file_path: Path) -> Dict[str, Any]:
        """Load PDF file."""
        try:
            import PyPDF2
            
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                
                pages = []
                full_text = []
                
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    pages.append({
                        "page_number": i + 1,
                        "text": page_text
                    })
                    full_text.append(page_text)
                
                return {
                    "content": "\n\n".join(full_text),
                    "pages": pages,
                    "file_path": str(file_path),
                    "file_type": "pdf",
                    "metadata": {
                        "num_pages": len(pages),
                        "size": len("\n\n".join(full_text))
                    }
                }
                
        except ImportError:
            logger.error("PyPDF2 not installed. Cannot load PDF files.")
            raise
        except Exception as e:
            logger.error(f"Failed to load PDF: {e}")
            raise
    
    def _load_docx(self, file_path: Path) -> Dict[str, Any]:
        """Load DOCX file."""
        try:
            import python_docx
            
            doc = python_docx.Document(file_path)
            
            paragraphs = []
            full_text = []
            
            for i, para in enumerate(doc.paragraphs):
                if para.text.strip():
                    paragraphs.append({
                        "paragraph_number": i + 1,
                        "text": para.text
                    })
                    full_text.append(para.text)
            
            return {
                "content": "\n\n".join(full_text),
                "paragraphs": paragraphs,
                "file_path": str(file_path),
                "file_type": "docx",
                "metadata": {
                    "num_paragraphs": len(paragraphs),
                    "size": len("\n\n".join(full_text))
                }
            }
            
        except ImportError:
            logger.error("python-docx not installed. Cannot load DOCX files.")
            raise
        except Exception as e:
            logger.error(f"Failed to load DOCX: {e}")
            raise
    
    def _load_markdown(self, file_path: Path) -> Dict[str, Any]:
        """Load Markdown file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return {
            "content": content,
            "file_path": str(file_path),
            "file_type": "markdown",
            "metadata": {
                "size": len(content),
                "lines": content.count('\n') + 1,
                "headers": content.count('#')
            }
        }
