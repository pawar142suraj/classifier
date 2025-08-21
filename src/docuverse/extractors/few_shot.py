"""
Unified Few-shot extractor for hybrid extraction and classification.
Supports dynamic loading of ground truth from data/labels folder.
Handles both field extraction and enum-based classification in a single pipeline.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import logging
import glob

from .base import BaseExtractor
from ..core.config import LLMConfig

logger = logging.getLogger(__name__)


class FewShotExtractor(BaseExtractor):
    """
    Unified few-shot extraction and classification method.
    
    Features:
    - Hybrid extraction + classification for fields with enums
    - Pure extraction for fields without enums
    - Dynamic loading from data/labels folder
    - Schema-aware processing
    - Unified pipeline for both extraction and classification
    """
    
    def __init__(
        self, 
        llm_config: LLMConfig, 
        schema: Optional[Dict[str, Any]] = None,
        schema_path: Optional[Union[str, Path]] = None,
        examples: Optional[List[Dict[str, Any]]] = None,
        data_labels_path: Optional[Union[str, Path]] = None,
        max_examples: int = 5,
        auto_load_labels: bool = True
    ):
        """
        Initialize unified few-shot extractor with hybrid extraction + classification.
        
        Args:
            llm_config: LLM configuration
            schema: JSON schema dict defining fields and their types
            schema_path: Path to JSON schema file
            examples: List of manual few-shot examples (optional)
            data_labels_path: Path to data/labels directory for auto-loading examples
            max_examples: Maximum number of examples to use in prompts
            auto_load_labels: Whether to automatically load examples from data/labels
        """
        super().__init__(llm_config)
        self.max_examples = max_examples
        self.schema = schema
        self.schema_path = schema_path
        
        # Load schema if path provided
        if schema_path and not schema:
            self.schema = self._load_schema(schema_path)
        
        # Initialize examples
        self.examples = []
        
        # Load manual examples first
        if examples:
            self.examples.extend(examples[:max_examples])
            logger.info(f"Loaded {len(examples)} manual examples")
        
        # Auto-load from data/labels if enabled
        if auto_load_labels:
            if data_labels_path:
                labels_path = Path(data_labels_path)
            else:
                # Default to data/labels relative to project root
                current_dir = Path(__file__).parent
                project_root = current_dir.parent.parent.parent  # Go up to project root
                labels_path = project_root / "data" / "labels"
            
            if labels_path.exists():
                auto_examples = self._load_examples_from_labels_folder(labels_path)
                remaining_slots = max_examples - len(self.examples)
                if remaining_slots > 0:
                    self.examples.extend(auto_examples[:remaining_slots])
                    logger.info(f"Auto-loaded {len(auto_examples[:remaining_slots])} examples from {labels_path}")
            else:
                logger.warning(f"Labels path not found: {labels_path}")
        
        logger.info(f"Initialized FewShotExtractor with {len(self.examples)} total examples")
    
    def _load_schema(self, schema_path: Union[str, Path]) -> Dict[str, Any]:
        """Load schema from JSON file."""
        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema = json.load(f)
            logger.info(f"Loaded schema from {schema_path}")
            return schema
        except Exception as e:
            logger.error(f"Failed to load schema from {schema_path}: {e}")
            return {}
    
    def _load_examples_from_labels_folder(self, labels_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Load few-shot examples from data/labels folder.
        
        Args:
            labels_path: Path to labels directory
            
        Returns:
            List of example dictionaries with 'input' and 'output' keys
        """
        labels_path = Path(labels_path)
        examples = []
        
        if not labels_path.exists():
            logger.error(f"Labels path does not exist: {labels_path}")
            return examples
        
        # Find all JSON files in labels folder
        label_files = list(labels_path.glob("*.json"))
        
        for label_file in label_files:
            try:
                # Load labels
                with open(label_file, 'r', encoding='utf-8') as f:
                    labels_data = json.load(f)
                
                # Look for corresponding document file in parent data directory
                base_name = label_file.stem.replace('_label', '').replace('.label', '')
                data_dir = labels_path.parent
                
                # Try to find document file
                doc_file = None
                for ext in ['.txt', '.md', '.pdf', '.docx']:
                    potential_doc = data_dir / f"{base_name}{ext}"
                    if potential_doc.exists():
                        doc_file = potential_doc
                        break
                
                # If no document file found, create synthetic input from labels
                if doc_file is None:
                    # Generate synthetic document content from labels for demo purposes
                    document_content = self._generate_synthetic_document_from_labels(labels_data, base_name)
                    logger.debug(f"Generated synthetic document for {label_file.name}")
                else:
                    # Load actual document content
                    with open(doc_file, 'r', encoding='utf-8') as f:
                        document_content = f.read()
                    logger.debug(f"Loaded document from {doc_file.name}")
                
                # Create example
                example = {
                    "input": document_content.strip(),
                    "output": labels_data,
                    "metadata": {
                        "labels_file": str(label_file),
                        "document_file": str(doc_file) if doc_file else "synthetic",
                        "base_name": base_name
                    }
                }
                
                examples.append(example)
                
            except Exception as e:
                logger.error(f"Failed to load example from {label_file}: {e}")
                continue
        
        logger.info(f"Loaded {len(examples)} examples from {labels_path}")
        return examples
    
    def _generate_synthetic_document_from_labels(self, labels_data: Dict[str, Any], base_name: str) -> str:
        """Generate synthetic document content from labels for demonstration."""
        doc_lines = [f"CONTRACT DOCUMENT - {base_name.upper()}", "=" * 50, ""]
        
        # Extract fields from labels
        fields = labels_data.get("fields", labels_data)
        
        for field_name, field_data in fields.items():
            if isinstance(field_data, dict):
                extracted_content = field_data.get("extracted_content", "")
                if extracted_content:
                    doc_lines.append(f"Regarding {field_name.replace('_', ' ').title()}:")
                    doc_lines.append(extracted_content.strip())
                    doc_lines.append("")
            elif isinstance(field_data, str):
                doc_lines.append(f"Regarding {field_name.replace('_', ' ').title()}:")
                doc_lines.append(field_data)
                doc_lines.append("")
        
        doc_lines.extend(["", "END OF CONTRACT", "=" * 50])
        return "\n".join(doc_lines)
    
    
    def extract(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and classify information using unified hybrid approach.
        
        Args:
            document: Document to extract from
            
        Returns:
            Extracted information with both extraction and classification
        """
        document_text = self._prepare_document_text(document)
        
        # Build hybrid prompt with schema context
        prompt = self._build_hybrid_prompt(document_text)
        
        # Enhanced system prompt for unified extraction + classification
        system_prompt = self._build_system_prompt()
        
        # Call LLM
        response = self._call_llm(prompt, system_prompt)
        
        # Parse and validate response
        try:
            extracted_data = json.loads(response.strip())
            
            # Post-process to ensure proper hybrid structure
            processed_data = self._post_process_extraction(extracted_data)
            
            # Calculate confidence
            self.last_confidence = self._calculate_hybrid_confidence(processed_data)
            
            return processed_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Response was: {response}")
            return {}
    
    def _build_system_prompt(self) -> str:
        """Build comprehensive system prompt for hybrid extraction + classification."""
        system_parts = [
            "You are an expert document information extractor specializing in hybrid extraction and classification.",
            "",
            "EXTRACTION GUIDELINES:",
            "",
            "1. **HYBRID FIELDS** (fields with enum definitions):",
            "   - Extract the exact relevant text from the document",
            "   - Classify the extracted content based on provided enum categories",
            "   - Structure: {\"extracted_content\": \"exact text\", \"classification\": \"enum_value\"}",
            "",
            "2. **PURE EXTRACTION FIELDS** (fields without enums):",
            "   - Extract the exact relevant content as a string",
            "   - Structure: {\"extracted_content\": \"exact text\"}",
            "",
            "3. **CLASSIFICATION PRINCIPLES**:",
            "   - Use only the enum values provided in the schema",
            "   - Base classification on the extracted content and context",
            "   - Consider enum descriptions for accurate classification",
            "",
            "4. **ACCURACY REQUIREMENTS**:",
            "   - Extract exact phrases and terms from the document",
            "   - Maintain consistency with provided examples",
            "   - If information is not found, use empty string for extracted_content",
            "",
            "5. **OUTPUT FORMAT**:",
            "   - Return valid JSON following the exact structure shown in examples",
            "   - Use \"fields\" as the top-level key",
            "   - No additional text or explanations",
            ""
        ]
        
        # Add schema context if available
        if self.schema:
            system_parts.extend([
                "SCHEMA CONTEXT:",
                "The following schema defines the expected fields and their types:",
                json.dumps(self.schema, indent=2),
                ""
            ])
        
        return "\n".join(system_parts)
    
    def _build_hybrid_prompt(self, document_text: str) -> str:
        """Build prompt with schema-aware few-shot examples."""
        prompt_parts = []
        
        # Add schema field definitions if available
        if self.schema and "field" in self.schema:
            prompt_parts.append("FIELD DEFINITIONS:")
            for field_name, field_def in self.schema["field"].items():
                field_type = field_def.get("type", "string")
                description = field_def.get("description", "")
                enums = field_def.get("enum", [])
                enum_descriptions = field_def.get("enumDescriptions", {})
                
                prompt_parts.append(f"- {field_name} ({field_type}): {description}")
                
                if enums:
                    prompt_parts.append(f"  CLASSIFICATION OPTIONS:")
                    for enum_val in enums:
                        enum_desc = enum_descriptions.get(enum_val, "")
                        prompt_parts.append(f"    * {enum_val}: {enum_desc}")
                    prompt_parts.append(f"  FORMAT: {{\"extracted_content\": \"text\", \"classification\": \"{enums[0]}\"}}")
                else:
                    prompt_parts.append(f"  FORMAT: {{\"extracted_content\": \"text\"}}")
                prompt_parts.append("")
            
            prompt_parts.append("---")
            prompt_parts.append("")
        
        # Add few-shot examples
        if self.examples:
            prompt_parts.append("EXAMPLES OF PROPER EXTRACTION AND CLASSIFICATION:")
            prompt_parts.append("")
            
            for i, example in enumerate(self.examples, 1):
                if "input" in example and "output" in example:
                    prompt_parts.append(f"Example {i}:")
                    prompt_parts.append("Input Document:")
                    prompt_parts.append(example["input"][:500] + "..." if len(example["input"]) > 500 else example["input"])
                    prompt_parts.append("")
                    prompt_parts.append("Extracted Information:")
                    prompt_parts.append(json.dumps(example["output"], indent=2))
                    prompt_parts.append("")
                    prompt_parts.append("---")
                    prompt_parts.append("")
        else:
            # Zero-shot instructions
            prompt_parts.extend([
                "EXTRACTION INSTRUCTIONS:",
                "Extract information following the hybrid approach:",
                "- For fields with enum options: provide both extracted_content and classification",
                "- For fields without enums: provide only extracted_content",
                "- Be precise and faithful to the source document",
                ""
            ])
        
        # Add the actual document to process
        prompt_parts.extend([
            "Now extract information from this document:",
            "",
            "Input Document:",
            document_text,
            "",
            "Extracted Information (JSON format):"
        ])
        
        return "\n".join(prompt_parts)
    
    def _post_process_extraction(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process extraction to ensure proper hybrid structure."""
        if not self.schema or "field" not in self.schema:
            return extracted_data
        
        # Ensure proper structure for each field
        processed_fields = {}
        fields_data = extracted_data.get("fields", extracted_data)
        
        for field_name, field_def in self.schema["field"].items():
            if field_name in fields_data:
                field_value = fields_data[field_name]
                has_enum = "enum" in field_def
                
                if has_enum:
                    # Should be hybrid structure
                    if isinstance(field_value, dict):
                        processed_fields[field_name] = field_value
                    elif isinstance(field_value, str):
                        # Convert simple string to hybrid structure
                        processed_fields[field_name] = {
                            "extracted_content": field_value,
                            "classification": self._classify_extracted_content(field_value, field_def)
                        }
                else:
                    # Should be simple extraction
                    if isinstance(field_value, dict) and "extracted_content" in field_value:
                        processed_fields[field_name] = field_value
                    elif isinstance(field_value, str):
                        processed_fields[field_name] = {"extracted_content": field_value}
                    else:
                        processed_fields[field_name] = field_value
            else:
                # Field not found in extraction
                if "enum" in field_def:
                    processed_fields[field_name] = {
                        "extracted_content": "",
                        "classification": field_def["enum"][0] if field_def["enum"] else ""
                    }
                else:
                    processed_fields[field_name] = {"extracted_content": ""}
        
        return {"fields": processed_fields}
    
    def _classify_extracted_content(self, content: str, field_def: Dict[str, Any]) -> str:
        """Classify extracted content based on field definition."""
        enums = field_def.get("enum", [])
        if not enums:
            return ""
        
        content_lower = content.lower()
        
        # Simple keyword-based classification
        for enum_val in enums:
            if enum_val.lower() in content_lower:
                return enum_val
        
        # Default to first enum if no match
        return enums[0]
    
    def _calculate_hybrid_confidence(self, extracted_data: Dict[str, Any]) -> float:
        """Calculate confidence for hybrid extraction + classification."""
        if not extracted_data or "fields" not in extracted_data:
            return 0.0
        
        fields = extracted_data["fields"]
        total_fields = len(self.schema.get("field", {})) if self.schema else len(fields)
        
        if total_fields == 0:
            return 0.8 if self.examples else 0.6
        
        # Score based on field completeness and structure
        correct_structure_count = 0
        non_empty_content_count = 0
        
        for field_name, field_value in fields.items():
            if isinstance(field_value, dict):
                # Check if has proper structure
                if "extracted_content" in field_value:
                    correct_structure_count += 1
                    if field_value["extracted_content"].strip():
                        non_empty_content_count += 1
        
        structure_score = correct_structure_count / total_fields
        content_score = non_empty_content_count / total_fields
        example_bonus = 0.1 if self.examples else 0.0
        
        return min(0.7 * structure_score + 0.2 * content_score + example_bonus, 1.0)
    
    
    def update_examples(self, new_examples: List[Dict[str, Any]]):
        """
        Update the examples used for few-shot learning.
        
        Args:
            new_examples: New list of examples to use
        """
        self.examples = new_examples[:self.max_examples]
        logger.info(f"Updated few-shot examples: {len(self.examples)} examples")
    
    def add_example_from_labels(self, labels_path: Union[str, Path], document_content: Optional[str] = None):
        """
        Add a new example from a labels file.
        
        Args:
            labels_path: Path to labels JSON file
            document_content: Optional document content (will generate synthetic if not provided)
        """
        try:
            labels_path = Path(labels_path)
            
            # Load labels
            with open(labels_path, 'r', encoding='utf-8') as f:
                labels = json.load(f)
            
            # Use provided content or generate synthetic
            if document_content is None:
                base_name = labels_path.stem.replace('_label', '').replace('.label', '')
                document_content = self._generate_synthetic_document_from_labels(labels, base_name)
            
            # Create example
            example = {
                "input": document_content.strip(),
                "output": labels,
                "metadata": {
                    "labels_file": str(labels_path),
                    "document_file": "provided" if document_content else "synthetic",
                    "base_name": labels_path.stem
                }
            }
            
            # Add to examples (keeping within max_examples limit)
            self.examples.append(example)
            if len(self.examples) > self.max_examples:
                self.examples = self.examples[-self.max_examples:]
            
            logger.info(f"Added example from {labels_path.name}, total examples: {len(self.examples)}")
            
        except Exception as e:
            logger.error(f"Failed to add example from {labels_path}: {e}")
    
    def reload_examples_from_labels(self, labels_path: Optional[Union[str, Path]] = None):
        """
        Reload all examples from the labels directory.
        
        Args:
            labels_path: Path to labels directory (uses default if not provided)
        """
        if labels_path is None:
            # Use default path
            current_dir = Path(__file__).parent
            project_root = current_dir.parent.parent.parent
            labels_path = project_root / "data" / "labels"
        
        labels_path = Path(labels_path)
        
        if not labels_path.exists():
            logger.error(f"Labels path does not exist: {labels_path}")
            return
        
        # Clear existing examples and reload
        old_count = len(self.examples)
        self.examples = self._load_examples_from_labels_folder(labels_path)
        
        logger.info(f"Reloaded examples: {old_count} -> {len(self.examples)}")
    
    def get_field_analysis(self) -> Dict[str, Any]:
        """Get analysis of fields based on schema and examples."""
        analysis = {
            "schema_fields": [],
            "hybrid_fields": [],
            "extraction_fields": [],
            "example_coverage": {},
            "total_examples": len(self.examples)
        }
        
        if self.schema and "field" in self.schema:
            for field_name, field_def in self.schema["field"].items():
                analysis["schema_fields"].append(field_name)
                
                if "enum" in field_def:
                    analysis["hybrid_fields"].append({
                        "name": field_name,
                        "type": field_def.get("type", "string"),
                        "description": field_def.get("description", ""),
                        "enum_options": field_def["enum"],
                        "enum_descriptions": field_def.get("enumDescriptions", {})
                    })
                else:
                    analysis["extraction_fields"].append({
                        "name": field_name,
                        "type": field_def.get("type", "string"),
                        "description": field_def.get("description", "")
                    })
        
        # Analyze example coverage
        if self.examples:
            for field_name in analysis["schema_fields"]:
                covered_examples = 0
                for example in self.examples:
                    output = example.get("output", {})
                    fields = output.get("fields", output)
                    if field_name in fields:
                        covered_examples += 1
                
                analysis["example_coverage"][field_name] = {
                    "covered_examples": covered_examples,
                    "coverage_percentage": (covered_examples / len(self.examples)) * 100
                }
        
        return analysis
    
    def validate_schema_compliance(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate extracted data against schema requirements.
        
        Args:
            extracted_data: Extracted data to validate
            
        Returns:
            Validation report with compliance details
        """
        validation_report = {
            "is_valid": True,
            "missing_fields": [],
            "invalid_enums": [],
            "structure_errors": [],
            "warnings": []
        }
        
        if not self.schema or "field" not in self.schema:
            validation_report["warnings"].append("No schema available for validation")
            return validation_report
        
        fields = extracted_data.get("fields", extracted_data)
        
        for field_name, field_def in self.schema["field"].items():
            # Check if field exists
            if field_name not in fields:
                validation_report["missing_fields"].append(field_name)
                validation_report["is_valid"] = False
                continue
            
            field_value = fields[field_name]
            has_enum = "enum" in field_def
            
            # Check structure
            if has_enum:
                if not isinstance(field_value, dict):
                    validation_report["structure_errors"].append(
                        f"{field_name}: Expected dict with extracted_content and classification"
                    )
                    validation_report["is_valid"] = False
                elif "classification" in field_value:
                    classification = field_value["classification"]
                    if classification not in field_def["enum"]:
                        validation_report["invalid_enums"].append(
                            f"{field_name}: '{classification}' not in {field_def['enum']}"
                        )
                        validation_report["is_valid"] = False
            else:
                if isinstance(field_value, dict) and "extracted_content" not in field_value:
                    validation_report["structure_errors"].append(
                        f"{field_name}: Expected extracted_content field"
                    )
                    validation_report["is_valid"] = False
        
        return validation_report
    
    def get_example_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of loaded examples and capabilities."""
        summary = {
            "total_examples": len(self.examples),
            "example_sources": [],
            "field_coverage": {},
            "schema_info": {},
            "capabilities": {
                "hybrid_extraction": bool(self.schema),
                "classification": bool(self.schema and any("enum" in field for field in self.schema.get("field", {}).values())),
                "few_shot_learning": len(self.examples) > 0,
                "auto_loading": True
            }
        }
        
        # Analyze examples
        for i, example in enumerate(self.examples):
            metadata = example.get("metadata", {})
            summary["example_sources"].append({
                "index": i + 1,
                "labels_file": metadata.get("labels_file", "unknown"),
                "document_file": metadata.get("document_file", "unknown"),
                "base_name": metadata.get("base_name", f"example_{i+1}")
            })
        
        # Field coverage analysis
        if self.examples:
            all_fields = set()
            for example in self.examples:
                output = example.get("output", {})
                fields = output.get("fields", output)
                all_fields.update(fields.keys())
            
            for field in all_fields:
                examples_with_field = sum(
                    1 for ex in self.examples 
                    if field in ex.get("output", {}).get("fields", ex.get("output", {}))
                )
                summary["field_coverage"][field] = {
                    "count": examples_with_field,
                    "percentage": (examples_with_field / len(self.examples)) * 100
                }
        
        # Schema info
        if self.schema:
            summary["schema_info"] = {
                "total_fields": len(self.schema.get("field", {})),
                "hybrid_fields": len([f for f in self.schema.get("field", {}).values() if "enum" in f]),
                "extraction_fields": len([f for f in self.schema.get("field", {}).values() if "enum" not in f])
            }
        
        return summary
