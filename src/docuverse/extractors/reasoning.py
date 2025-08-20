"""
Reasoning-enhanced extractor with CoT and ReAct.
"""

import json
from typing import Dict, Any
import logging

from .base import BaseExtractor
from ..core.config import LLMConfig, ReasoningConfig, ExtractionMethod

logger = logging.getLogger(__name__)


class ReasoningExtractor(BaseExtractor):
    """
    Reasoning-enhanced extraction with Chain of Thought or ReAct.
    """
    
    def __init__(self, llm_config: LLMConfig, reasoning_config: ReasoningConfig, method_type: ExtractionMethod):
        """Initialize reasoning extractor."""
        super().__init__(llm_config)
        self.reasoning_config = reasoning_config
        self.method_type = method_type
        
        logger.info(f"Initialized ReasoningExtractor with {method_type}")
    
    def extract(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Extract using reasoning approach."""
        document_text = self._prepare_document_text(document)
        
        if self.method_type == ExtractionMethod.REASONING_COT:
            return self._extract_with_cot(document_text)
        elif self.method_type == ExtractionMethod.REASONING_REACT:
            return self._extract_with_react(document_text)
        else:
            raise ValueError(f"Unsupported reasoning method: {self.method_type}")
    
    def _extract_with_cot(self, document_text: str) -> Dict[str, Any]:
        """Extract using Chain of Thought reasoning."""
        prompt = f"""Extract structured information from the document using step-by-step reasoning.

Document:
{document_text}

Think step by step:
1. First, identify the document type and key sections
2. Then, locate specific information fields
3. Finally, extract and structure the information

Reasoning:"""
        
        system_prompt = "You are an expert extractor using chain-of-thought reasoning."
        
        response = self._call_llm(prompt, system_prompt)
        
        # In a full implementation, this would parse the reasoning steps
        # and extract the final JSON
        try:
            # Simple extraction from response
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end]
            else:
                # Try to find JSON-like content
                json_str = response
            
            extracted_data = json.loads(json_str.strip())
            self.last_confidence = 0.82
            return extracted_data
        except json.JSONDecodeError:
            logger.error("Failed to parse CoT reasoning response")
            return {}
    
    def _extract_with_react(self, document_text: str) -> Dict[str, Any]:
        """Extract using ReAct (Reasoning + Acting) approach."""
        prompt = f"""Extract structured information using ReAct methodology (Reasoning + Acting).

Document:
{document_text}

Use this format:
Thought: [your reasoning about what to extract]
Action: [what specific extraction step to take]
Observation: [what you found]
... (repeat as needed)
Final Answer: [JSON with extracted information]

Begin:"""
        
        system_prompt = "You are an expert extractor using ReAct methodology."
        
        response = self._call_llm(prompt, system_prompt)
        
        # In a full implementation, this would parse the ReAct steps
        try:
            # Extract final answer
            if "Final Answer:" in response:
                final_start = response.find("Final Answer:") + 13
                json_str = response[final_start:].strip()
            else:
                json_str = response
            
            if "```json" in json_str:
                json_start = json_str.find("```json") + 7
                json_end = json_str.find("```", json_start)
                json_str = json_str[json_start:json_end]
            
            extracted_data = json.loads(json_str.strip())
            self.last_confidence = 0.80
            return extracted_data
        except json.JSONDecodeError:
            logger.error("Failed to parse ReAct response")
            return {}
