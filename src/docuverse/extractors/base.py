"""
Base extractor class that all extraction methods inherit from.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

from ..core.config import LLMConfig

logger = logging.getLogger(__name__)


class BaseExtractor(ABC):
    """
    Abstract base class for all extraction methods.
    """
    
    def __init__(self, llm_config: LLMConfig):
        """Initialize base extractor."""
        self.llm_config = llm_config
        self.last_token_usage = None
        self.last_chunks_processed = None
        self.last_confidence = None
        
        logger.info(f"Initialized {self.__class__.__name__}")
    
    @abstractmethod
    def extract(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract information from a document.
        
        Args:
            document: Document data containing text, metadata, etc.
            
        Returns:
            Extracted information as a dictionary
        """
        pass
    
    def _prepare_document_text(self, document: Dict[str, Any]) -> str:
        """Extract and prepare text content from document."""
        if isinstance(document, dict):
            # Handle different document formats
            if "content" in document:
                return document["content"]
            elif "text" in document:
                return document["text"]
            elif "pages" in document:
                # Multi-page document
                return "\n\n".join([page.get("text", "") for page in document["pages"]])
            else:
                # Try to concatenate all string values
                text_parts = []
                for value in document.values():
                    if isinstance(value, str):
                        text_parts.append(value)
                return "\n".join(text_parts)
        elif isinstance(document, str):
            return document
        else:
            raise ValueError(f"Unsupported document format: {type(document)}")
    
    def _get_llm_client(self):
        """Get appropriate LLM client based on configuration."""
        try:
            from ..utils.llm_client import LLMManager
            return LLMManager(self.llm_config)
        except ImportError as e:
            logger.error(f"Failed to import LLM client: {e}")
            # Fallback to simple client
            return self._get_simple_llm_client()
    
    def _get_simple_llm_client(self):
        """Simple fallback LLM client for basic OpenAI/Anthropic support."""
        try:
            if self.llm_config.provider.lower() == "openai":
                from openai import OpenAI
                return OpenAI(api_key=self.llm_config.api_key)
            elif self.llm_config.provider.lower() == "anthropic":
                from anthropic import Anthropic
                return Anthropic(api_key=self.llm_config.api_key)
            else:
                raise ValueError(f"Unsupported LLM provider: {self.llm_config.provider}")
        except ImportError as e:
            logger.error(f"Failed to import LLM client: {e}")
            raise
    
    def _call_llm(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Call the configured LLM with the given prompt."""
        client = self._get_llm_client()
        
        # Use new LLM manager if available
        if hasattr(client, 'generate'):
            try:
                response = client.generate(prompt, system_prompt)
                
                # Get statistics from the client
                stats = client.get_combined_stats()
                if stats and "totals" in stats:
                    self.last_token_usage = {
                        "total_tokens": stats["totals"]["tokens"],
                        "requests": stats["totals"]["requests"],
                        "cost": stats["totals"]["cost"]
                    }
                
                return response
                
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                raise
        
        # Fallback to simple client
        return self._call_simple_llm(client, prompt, system_prompt)
    
    def _call_simple_llm(self, client, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Fallback method for simple LLM clients."""
        try:
            if self.llm_config.provider.lower() == "openai":
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                response = client.chat.completions.create(
                    model=self.llm_config.model_name,
                    messages=messages,
                    temperature=self.llm_config.temperature,
                    max_tokens=self.llm_config.max_tokens,
                    timeout=self.llm_config.timeout
                )
                
                self.last_token_usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
                
                return response.choices[0].message.content
                
            elif self.llm_config.provider.lower() == "anthropic":
                response = client.messages.create(
                    model=self.llm_config.model_name,
                    max_tokens=self.llm_config.max_tokens,
                    temperature=self.llm_config.temperature,
                    system=system_prompt or "",
                    messages=[{"role": "user", "content": prompt}]
                )
                
                self.last_token_usage = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                }
                
                return response.content[0].text
                
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
