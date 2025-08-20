"""
Classification-enhanced extractor that handles enums with definitions.
"""

import json
from typing import Dict, Any, List, Optional
import logging

from .base import BaseExtractor
from ..core.config import LLMConfig
from ..utils.schema_validator import SchemaValidator

logger = logging.getLogger(__name__)


class ClassificationExtractor(BaseExtractor):
    """
    Enhanced extractor for classification tasks using enum definitions.
    """
    
    def __init__(self, llm_config: LLMConfig, schema_path: Optional[str] = None):
        """
        Initialize classification extractor.
        
        Args:
            llm_config: LLM configuration
            schema_path: Path to JSON schema with enum definitions
        """
        super().__init__(llm_config)
        self.schema_validator = SchemaValidator(schema_path) if schema_path else None
        self.schema = None
        
        if schema_path:
            with open(schema_path, 'r') as f:
                self.schema = json.load(f)
        
        logger.info("Initialized ClassificationExtractor")
    
    def extract(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract information with classification based on enum definitions.
        
        Args:
            document: Document to extract from
            
        Returns:
            Extracted information with classifications
        """
        document_text = self._prepare_document_text(document)
        
        if not self.schema:
            # Fallback to basic extraction if no schema
            return self._basic_extract(document_text)
        
        # Extract enum fields with definitions
        enum_fields = self._extract_enum_fields()
        
        # Build classification prompt
        prompt = self._build_classification_prompt(document_text, enum_fields)
        
        # System prompt for classification
        system_prompt = """You are an expert document classifier and information extractor.
        Your task is to:
        1. Extract relevant content from the document
        2. Classify fields based on the provided enum definitions
        3. Provide evidence for your classifications
        4. Return structured JSON output
        
        Be precise and base classifications on the specific definitions provided."""
        
        # Call LLM
        response = self._call_llm(prompt, system_prompt)
        
        # Parse and validate response
        try:
            extracted_data = json.loads(response.strip())
            
            # Validate against schema if available
            if self.schema_validator:
                validation_result = self.schema_validator.validate(extracted_data)
                if not validation_result.is_valid:
                    logger.warning(f"Classification validation failed: {validation_result.errors}")
                    # Try to repair common issues
                    extracted_data = self._repair_classification(extracted_data, validation_result.errors)
            
            self.last_confidence = self._calculate_classification_confidence(extracted_data)
            return extracted_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse classification response: {e}")
            logger.error(f"Response was: {response}")
            return {}
    
    def _extract_enum_fields(self) -> Dict[str, Dict]:
        """Extract enum fields and their definitions from schema."""
        enum_fields = {}
        
        if not self.schema or "properties" not in self.schema:
            return enum_fields
        
        for field_name, field_def in self.schema["properties"].items():
            # Handle simple enum fields (classification only)
            if "enum" in field_def and "enumDescriptions" in field_def:
                enum_fields[field_name] = {
                    "description": field_def.get("description", ""),
                    "enum_values": field_def["enum"],
                    "enum_descriptions": field_def["enumDescriptions"],
                    "extraction_type": field_def.get("extractionType", "classification")
                }
            
            # Handle hybrid fields (extraction + classification)
            elif field_def.get("type") == "object" and "properties" in field_def:
                properties = field_def["properties"]
                if "classification" in properties and "enum" in properties["classification"]:
                    enum_fields[field_name] = {
                        "description": field_def.get("description", ""),
                        "enum_values": properties["classification"]["enum"],
                        "enum_descriptions": properties["classification"].get("enumDescriptions", {}),
                        "extraction_type": field_def.get("extractionType", "hybrid"),
                        "has_extraction": "extracted_text" in properties or "extracted_contacts" in properties,
                        "properties": properties
                    }
        
        return enum_fields
    
    def _build_classification_prompt(self, document_text: str, enum_fields: Dict[str, Dict]) -> str:
        """Build prompt for classification with enum definitions."""
        prompt_parts = []
        
        prompt_parts.append("Analyze the following document and perform both extraction and classification tasks:")
        prompt_parts.append("")
        prompt_parts.append("DOCUMENT:")
        prompt_parts.append(document_text)
        prompt_parts.append("")
        
        if enum_fields:
            prompt_parts.append("EXTRACTION AND CLASSIFICATION TASKS:")
            prompt_parts.append("")
            
            for field_name, field_info in enum_fields.items():
                extraction_type = field_info.get('extraction_type', 'classification')
                
                prompt_parts.append(f"**{field_name}**: {field_info['description']}")
                
                if extraction_type == "classification":
                    # Simple classification field
                    prompt_parts.append("Task: Classify the document into one of these categories:")
                    for enum_value in field_info['enum_values']:
                        description = field_info['enum_descriptions'].get(enum_value, "No description")
                        prompt_parts.append(f"  - '{enum_value}': {description}")
                
                elif extraction_type == "hybrid":
                    # Hybrid extraction + classification field
                    prompt_parts.append("Task: Extract relevant content AND classify it:")
                    prompt_parts.append("1. Extract the actual text/content related to this field")
                    prompt_parts.append("2. Classify the extracted content into one of these categories:")
                    
                    for enum_value in field_info['enum_values']:
                        description = field_info['enum_descriptions'].get(enum_value, "No description")
                        prompt_parts.append(f"     - '{enum_value}': {description}")
                    
                    if 'properties' in field_info:
                        prompt_parts.append("3. Additional fields to extract:")
                        for prop_name, prop_def in field_info['properties'].items():
                            if prop_name not in ['classification', 'extracted_text', 'extracted_contacts']:
                                prop_desc = prop_def.get('description', '')
                                prompt_parts.append(f"     - {prop_name}: {prop_desc}")
                
                prompt_parts.append("")
        
        prompt_parts.append("INSTRUCTIONS:")
        prompt_parts.append("1. Read the document carefully")
        prompt_parts.append("2. For CLASSIFICATION fields: Select the most appropriate category based on definitions")
        prompt_parts.append("3. For HYBRID fields:")
        prompt_parts.append("   a. Extract the exact text/content from the document")
        prompt_parts.append("   b. Classify that content according to the enum definitions")
        prompt_parts.append("   c. Extract any additional required fields")
        prompt_parts.append("4. Provide evidence snippets to justify your decisions")
        prompt_parts.append("5. If content is not found, use appropriate null/unknown values")
        prompt_parts.append("")
        
        prompt_parts.append("EXPECTED JSON STRUCTURE:")
        
        # Build example JSON structure
        example_structure = {}
        for field_name, field_info in enum_fields.items():
            extraction_type = field_info.get('extraction_type', 'classification')
            
            if extraction_type == "classification":
                example_structure[field_name] = f"<one of: {', '.join(field_info['enum_values'])}>"
            elif extraction_type == "hybrid":
                # Create hybrid structure
                hybrid_example = {
                    "extracted_text": "<actual text found in document>",
                    "classification": f"<one of: {', '.join(field_info['enum_values'])}>"
                }
                
                # Add additional properties if they exist
                if 'properties' in field_info:
                    for prop_name, prop_def in field_info['properties'].items():
                        if prop_name not in ['classification', 'extracted_text']:
                            if prop_def.get('type') == 'array':
                                hybrid_example[prop_name] = ["<extracted items>"]
                            elif prop_def.get('type') == 'number':
                                hybrid_example[prop_name] = "<extracted number>"
                            else:
                                hybrid_example[prop_name] = f"<extracted {prop_name}>"
                
                example_structure[field_name] = hybrid_example
        
        # Add evidence structure
        evidence_structure = {}
        for field_name in enum_fields.keys():
            evidence_structure[f"{field_name}_evidence"] = ["<text snippets supporting extraction/classification>"]
        
        example_structure["extracted_content"] = {
            "key_entities": ["<important entities from document>"],
            "classification_evidence": evidence_structure
        }
        
        prompt_parts.append(json.dumps(example_structure, indent=2))
        prompt_parts.append("")
        prompt_parts.append("JSON Response:")
        
        return "\n".join(prompt_parts)
    
    def _basic_extract(self, document_text: str) -> Dict[str, Any]:
        """Fallback extraction when no schema is available."""
        prompt = f"""Extract and classify information from this document:

{document_text}

Return structured JSON with any classifications you can determine."""
        
        system_prompt = "You are a document classifier. Extract and classify information appropriately."
        
        response = self._call_llm(prompt, system_prompt)
        
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            logger.error("Failed to parse basic extraction response")
            return {}
    
    def _repair_classification(self, data: Dict[str, Any], errors: List[str]) -> Dict[str, Any]:
        """Attempt to repair common classification errors."""
        repaired_data = data.copy()
        
        # Get enum fields for validation
        enum_fields = self._extract_enum_fields()
        
        for field_name, field_info in enum_fields.items():
            if field_name in repaired_data:
                current_value = repaired_data[field_name]
                valid_values = field_info['enum_values']
                
                # If value is not in enum, try to map it
                if current_value not in valid_values:
                    # Try case-insensitive match
                    for valid_value in valid_values:
                        if current_value.lower() == valid_value.lower():
                            repaired_data[field_name] = valid_value
                            break
                    else:
                        # If no match, try semantic similarity (simplified)
                        mapped_value = self._map_to_enum(current_value, valid_values, field_info['enum_descriptions'])
                        if mapped_value:
                            repaired_data[field_name] = mapped_value
                            logger.info(f"Mapped '{current_value}' to '{mapped_value}' for field '{field_name}'")
        
        return repaired_data
    
    def _map_to_enum(self, value: str, valid_values: List[str], descriptions: Dict[str, str]) -> Optional[str]:
        """Map an invalid value to a valid enum value using descriptions."""
        value_lower = value.lower()
        
        # Check if value appears in any description
        for enum_value, description in descriptions.items():
            if value_lower in description.lower() or any(word in description.lower() for word in value_lower.split()):
                return enum_value
        
        # Default to first value if no match
        return valid_values[0] if valid_values else None
    
    def _calculate_classification_confidence(self, extracted_data: Dict[str, Any]) -> float:
        """Calculate confidence score for classification results."""
        if not extracted_data:
            return 0.0
        
        confidence_factors = []
        
        # Check if evidence is provided
        if "extracted_content" in extracted_data and "classification_evidence" in extracted_data["extracted_content"]:
            evidence = extracted_data["extracted_content"]["classification_evidence"]
            
            # Higher confidence if evidence is provided for classifications
            for field, indicators in evidence.items():
                if isinstance(indicators, list) and len(indicators) > 0:
                    confidence_factors.append(0.9)  # High confidence with evidence
                else:
                    confidence_factors.append(0.6)  # Lower confidence without evidence
        
        # Check if all required classifications are present
        enum_fields = self._extract_enum_fields()
        classification_completeness = 0
        if enum_fields:
            present_classifications = sum(1 for field in enum_fields.keys() if field in extracted_data)
            classification_completeness = present_classifications / len(enum_fields)
            confidence_factors.append(classification_completeness)
        
        # Calculate overall confidence
        if confidence_factors:
            return sum(confidence_factors) / len(confidence_factors)
        else:
            return 0.7  # Default confidence
    
    def classify_batch(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Classify a batch of documents efficiently.
        
        Args:
            documents: List of documents to classify
            
        Returns:
            List of classification results
        """
        results = []
        
        for doc in documents:
            try:
                result = self.extract(doc)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to classify document: {e}")
                results.append({"error": str(e)})
        
        return results
    
    def get_classification_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate statistics from classification results.
        
        Args:
            results: List of classification results
            
        Returns:
            Classification statistics
        """
        stats = {
            "total_documents": len(results),
            "successful_classifications": 0,
            "field_distributions": {},
            "confidence_stats": {
                "mean": 0.0,
                "min": 1.0,
                "max": 0.0
            }
        }
        
        enum_fields = self._extract_enum_fields()
        
        # Initialize field distributions
        for field_name, field_info in enum_fields.items():
            stats["field_distributions"][field_name] = {
                value: 0 for value in field_info['enum_values']
            }
        
        confidences = []
        
        for result in results:
            if "error" not in result:
                stats["successful_classifications"] += 1
                
                # Count field distributions
                for field_name in enum_fields.keys():
                    if field_name in result:
                        value = result[field_name]
                        if value in stats["field_distributions"][field_name]:
                            stats["field_distributions"][field_name][value] += 1
                
                # Collect confidence if available
                if hasattr(self, 'last_confidence') and self.last_confidence:
                    confidences.append(self.last_confidence)
        
        # Calculate confidence statistics
        if confidences:
            stats["confidence_stats"]["mean"] = sum(confidences) / len(confidences)
            stats["confidence_stats"]["min"] = min(confidences)
            stats["confidence_stats"]["max"] = max(confidences)
        
        return stats
