"""
Few-shot baseline extractor using examples for in-context learning.
Supports dynamic loading of ground truth from *.labels.json files.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import logging

from .base import BaseExtractor
from ..core.config import LLMConfig

logger = logging.getLogger(__name__)


class FewShotExtractor(BaseExtractor):
    """
    Baseline few-shot extraction method using examples for in-context learning.
    Supports both manual examples and dynamic loading from *.labels.json files.
    """
    
    def __init__(
        self, 
        llm_config: LLMConfig, 
        examples: Optional[List[Dict[str, Any]]] = None,
        data_path: Optional[Union[str, Path]] = None,
        max_examples: int = 3,
        schema_path: Optional[Union[str, Path]] = None
    ):
        """
        Initialize few-shot extractor.
        
        Args:
            llm_config: LLM configuration
            examples: List of few-shot examples (input-output pairs) - optional if using data_path
            data_path: Path to directory containing documents and *.labels.json files
            max_examples: Maximum number of examples to use in prompts
            schema_path: Path to JSON schema file for validation context
        """
        super().__init__(llm_config)
        self.max_examples = max_examples
        self.schema_path = schema_path
        
        # Load examples from provided list or data directory
        if examples:
            self.examples = examples[:max_examples]
            logger.info(f"Initialized FewShotExtractor with {len(self.examples)} manual examples")
        elif data_path:
            self.examples = self._load_examples_from_data_path(data_path)
            logger.info(f"Initialized FewShotExtractor with {len(self.examples)} examples from {data_path}")
        else:
            self.examples = []
            logger.warning("FewShotExtractor initialized without examples - will use zero-shot approach")
    
    def _load_examples_from_data_path(self, data_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Load few-shot examples from a directory containing documents and *.labels.json files.
        
        Args:
            data_path: Path to directory containing example files
            
        Returns:
            List of example dictionaries with 'input' and 'output' keys
        """
        data_path = Path(data_path)
        examples = []
        
        if not data_path.exists():
            logger.error(f"Data path does not exist: {data_path}")
            return examples
        
        # Find all .labels.json files
        label_files = list(data_path.glob("*.labels.json"))
        
        for label_file in label_files[:self.max_examples]:
            # Get corresponding document file
            base_name = label_file.stem.replace('.labels', '')
            
            # Try common document extensions
            doc_file = None
            for ext in ['.txt', '.md', '.pdf', '.docx']:
                potential_doc = data_path / f"{base_name}{ext}"
                if potential_doc.exists():
                    doc_file = potential_doc
                    break
            
            if doc_file is None:
                logger.warning(f"No document file found for labels: {label_file}")
                continue
            
            try:
                # Load ground truth labels
                with open(label_file, 'r', encoding='utf-8') as f:
                    labels = json.load(f)
                
                # Load document content
                with open(doc_file, 'r', encoding='utf-8') as f:
                    document_content = f.read()
                
                # Create example
                example = {
                    "input": document_content.strip(),
                    "output": labels,
                    "metadata": {
                        "document_file": str(doc_file),
                        "labels_file": str(label_file),
                        "document_type": labels.get("document_type", "unknown")
                    }
                }
                
                examples.append(example)
                logger.debug(f"Loaded example from {doc_file.name}")
                
            except Exception as e:
                logger.error(f"Failed to load example from {label_file}: {e}")
                continue
        
        if not examples:
            logger.warning(f"No valid examples found in {data_path}")
        else:
            logger.info(f"Loaded {len(examples)} examples from {data_path}")
            
        return examples
    
    def extract(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract information using few-shot prompting with hybrid extraction + classification.
        
        Args:
            document: Document to extract from
            
        Returns:
            Extracted information following hybrid schema format
        """
        document_text = self._prepare_document_text(document)
        
        # Build few-shot prompt with examples
        prompt = self._build_few_shot_prompt(document_text)
        
        # Enhanced system prompt for hybrid extraction + classification
        system_prompt = """You are an expert document information extractor specializing in contract analysis. 

Extract structured information from documents following these guidelines:

1. **Hybrid Fields**: For fields with both extracted_text and classification:
   - Extract the exact relevant text from the document
   - Classify the extracted content based on the provided enum definitions
   - Be precise and faithful to the source text

2. **Classification Fields**: For fields with only classification:
   - Analyze the document content and classify based on enum definitions
   - Consider context, urgency indicators, and document structure

3. **Structured Data**: Extract specific data types (amounts, dates, contacts) accurately

4. **Evidence-Based**: Base all classifications on actual document content
   - Use exact phrases and terms from the document
   - Maintain consistency with the provided examples

Return only valid JSON without any additional text or explanation.
Follow the exact schema structure shown in the examples."""
        
        # Call LLM
        response = self._call_llm(prompt, system_prompt)
        
        # Parse response
        try:
            extracted_data = json.loads(response.strip())
            
            # Calculate confidence based on example similarity and field completeness
            self.last_confidence = self._calculate_extraction_confidence(extracted_data)
            
            return extracted_data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Response was: {response}")
            return {}
    
    def _calculate_extraction_confidence(self, extracted_data: Dict[str, Any]) -> float:
        """Calculate confidence score based on extraction completeness and consistency."""
        if not extracted_data:
            return 0.0
        
        # Base confidence from having examples
        base_confidence = 0.9 if self.examples else 0.7
        
        # Check field completeness
        expected_fields = ["payment_terms", "contract_value", "document_type"]
        found_fields = sum(1 for field in expected_fields if field in extracted_data)
        completeness_score = found_fields / len(expected_fields)
        
        # Check hybrid field structure
        hybrid_fields = ["payment_terms", "contract_value", "delivery_terms", "compliance_requirements"]
        proper_structure_count = 0
        total_hybrid_count = 0
        
        for field in hybrid_fields:
            if field in extracted_data:
                total_hybrid_count += 1
                field_data = extracted_data[field]
                if isinstance(field_data, dict):
                    if "extracted_text" in field_data and "classification" in field_data:
                        proper_structure_count += 1
        
        structure_score = proper_structure_count / max(total_hybrid_count, 1)
        
        # Combine scores
        final_confidence = base_confidence * 0.5 + completeness_score * 0.3 + structure_score * 0.2
        return min(final_confidence, 1.0)
    
    def _build_few_shot_prompt(self, document_text: str) -> str:
        """Build prompt with few-shot examples for hybrid extraction + classification."""
        prompt_parts = []
        
        # Add schema context if available
        if self.schema_path:
            prompt_parts.append("EXTRACTION SCHEMA CONTEXT:")
            prompt_parts.append("This extraction follows a hybrid approach where fields can:")
            prompt_parts.append("1. Extract exact text content from documents")
            prompt_parts.append("2. Classify that content using predefined categories")
            prompt_parts.append("3. Parse structured data (amounts, dates, contacts)")
            prompt_parts.append("")
        
        # Add examples
        if self.examples:
            prompt_parts.append("Here are examples of contract information extraction:")
            prompt_parts.append("")
            
            for i, example in enumerate(self.examples, 1):
                if "input" in example and "output" in example:
                    prompt_parts.append(f"Example {i}:")
                    prompt_parts.append("Input Document:")
                    prompt_parts.append(example["input"])
                    prompt_parts.append("")
                    prompt_parts.append("Extracted Information:")
                    prompt_parts.append(json.dumps(example["output"], indent=2))
                    prompt_parts.append("")
                    prompt_parts.append("---")
                    prompt_parts.append("")
        else:
            # Zero-shot instructions
            prompt_parts.append("EXTRACTION INSTRUCTIONS:")
            prompt_parts.append("Extract contract information following this hybrid approach:")
            prompt_parts.append("")
            prompt_parts.append("For HYBRID fields (like payment_terms, contract_value):")
            prompt_parts.append("- extracted_text: Exact text from the document")
            prompt_parts.append("- classification: Category based on the content")
            prompt_parts.append("- Additional fields: Parsed structured data")
            prompt_parts.append("")
            prompt_parts.append("For CLASSIFICATION fields (like document_type, priority_level):")
            prompt_parts.append("- Single classification value based on document analysis")
            prompt_parts.append("")
            prompt_parts.append("For CONTACT fields:")
            prompt_parts.append("- Extract structured contact information")
            prompt_parts.append("- Classify completeness of contact data")
            prompt_parts.append("")
        
        # Add the actual document to extract from
        prompt_parts.append("Now extract information from this contract document:")
        prompt_parts.append("")
        prompt_parts.append("Input Document:")
        prompt_parts.append(document_text)
        prompt_parts.append("")
        prompt_parts.append("Extracted Information:")
        
        return "\n".join(prompt_parts)
    
    def update_examples(self, new_examples: List[Dict[str, Any]]):
        """
        Update the examples used for few-shot learning.
        
        Args:
            new_examples: New list of examples to use
        """
        self.examples = new_examples[:self.max_examples]
        logger.info(f"Updated few-shot examples: {len(self.examples)} examples")
    
    def add_example_from_files(self, document_path: Union[str, Path], labels_path: Union[str, Path]):
        """
        Add a new example from document and labels files.
        
        Args:
            document_path: Path to document file
            labels_path: Path to corresponding labels JSON file
        """
        try:
            # Load document
            with open(document_path, 'r', encoding='utf-8') as f:
                document_content = f.read()
            
            # Load labels
            with open(labels_path, 'r', encoding='utf-8') as f:
                labels = json.load(f)
            
            # Create example
            example = {
                "input": document_content.strip(),
                "output": labels,
                "metadata": {
                    "document_file": str(document_path),
                    "labels_file": str(labels_path),
                    "document_type": labels.get("document_type", "unknown")
                }
            }
            
            # Add to examples (keeping within max_examples limit)
            self.examples.append(example)
            if len(self.examples) > self.max_examples:
                self.examples = self.examples[-self.max_examples:]
            
            logger.info(f"Added example from {Path(document_path).name}, total examples: {len(self.examples)}")
            
        except Exception as e:
            logger.error(f"Failed to add example from files: {e}")
    
    def get_example_summary(self) -> Dict[str, Any]:
        """Get summary information about loaded examples."""
        if not self.examples:
            return {"count": 0, "document_types": [], "fields_covered": []}
        
        # Analyze examples
        document_types = []
        all_fields = set()
        
        for example in self.examples:
            metadata = example.get("metadata", {})
            doc_type = metadata.get("document_type", "unknown")
            if doc_type not in document_types:
                document_types.append(doc_type)
            
            # Collect fields from output
            output = example.get("output", {})
            all_fields.update(output.keys())
        
        return {
            "count": len(self.examples),
            "document_types": document_types,
            "fields_covered": sorted(list(all_fields)),
            "example_sources": [ex.get("metadata", {}).get("document_file", "manual") for ex in self.examples]
        }
