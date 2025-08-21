"""
Reasoning-enhanced extractor with CoT and ReAct, building on Vector RAG capabilities.
Implements advanced reasoning strategies for contract information extraction including:
- Chain of Thought (CoT) reasoning with retrieval-augmented context
- ReAct (Reasoning + Acting) methodology with iterative refinement
- Multi-step verification and self-correction
- Uncertainty-aware extraction with confidence scoring
- Schema-guided reasoning for structured outputs
"""

import json
import re
import time
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass
import logging

from .base import BaseExtractor
from .vector_rag import VectorRAGExtractor, HybridRetriever, AdvancedChunker, RetrievalResult
from ..core.config import LLMConfig, ReasoningConfig, VectorRAGConfig, ExtractionMethod

logger = logging.getLogger(__name__)


@dataclass
class ReasoningStep:
    """Represents a step in the reasoning process."""
    step_type: str  # "thought", "action", "observation", "verification"
    content: str
    confidence: float = 0.0
    step_number: int = 0
    extracted_info: Optional[Dict[str, Any]] = None
    uncertainty_flags: List[str] = None
    
    def __post_init__(self):
        if self.uncertainty_flags is None:
            self.uncertainty_flags = []


@dataclass
class ExtractionEvidence:
    """Evidence supporting an extracted field."""
    field_name: str
    extracted_value: Any
    evidence_text: str
    confidence: float
    reasoning: str
    source_chunks: List[str] = None
    
    def __post_init__(self):
        if self.source_chunks is None:
            self.source_chunks = []


class ReasoningExtractor(BaseExtractor):
    """
    Advanced reasoning-enhanced extraction building on Vector RAG capabilities.
    
    Features:
    - Chain of Thought reasoning with retrieval context
    - ReAct methodology for iterative problem solving
    - Multi-step verification and self-correction
    - Uncertainty detection and handling
    - Schema-guided structured reasoning
    - Evidence tracking and confidence scoring
    """
    
    def __init__(
        self, 
        llm_config: LLMConfig, 
        reasoning_config: ReasoningConfig,
        method_type: ExtractionMethod,
        schema: Optional[Dict[str, Any]] = None,
        schema_path: Optional[Union[str, Path]] = None,
        use_vector_rag: bool = True
    ):
        """Initialize reasoning extractor with optional Vector RAG integration."""
        super().__init__(llm_config)
        self.reasoning_config = reasoning_config
        self.method_type = method_type
        self.schema = schema
        self.use_vector_rag = use_vector_rag
        
        # Load schema if path provided
        if schema_path and not schema:
            self.schema = self._load_schema(schema_path)
        
        # Initialize Vector RAG components if enabled
        self.vector_rag_extractor = None
        self.retriever = None
        self.chunker = None
        
        if use_vector_rag:
            # Create a Vector RAG config optimized for reasoning
            rag_config = VectorRAGConfig(
                chunk_size=384,  # Smaller chunks for better reasoning
                chunk_overlap=50,
                chunking_strategy="semantic",
                retrieval_k=8,  # More chunks for comprehensive reasoning
                rerank_top_k=5,
                use_hybrid_search=True,
                bm25_weight=0.4  # Higher weight for keyword matching
            )
            
            try:
                self.vector_rag_extractor = VectorRAGExtractor(
                    llm_config=llm_config,
                    rag_config=rag_config,
                    schema=self.schema
                )
                self.retriever = self.vector_rag_extractor.retriever
                self.chunker = self.vector_rag_extractor.chunker
                logger.info("Initialized ReasoningExtractor with Vector RAG integration")
            except Exception as e:
                logger.warning(f"Failed to initialize Vector RAG integration: {e}")
                self.use_vector_rag = False
        
        # Reasoning state
        self.reasoning_steps: List[ReasoningStep] = []
        self.extraction_evidence: List[ExtractionEvidence] = []
        self.uncertainty_threshold = reasoning_config.uncertainty_threshold
        
        logger.info(f"Initialized ReasoningExtractor with {method_type.value}")
    
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
    
    def extract(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Extract using reasoning approach with optional retrieval augmentation."""
        start_time = time.time()
        
        # Reset reasoning state
        self.reasoning_steps = []
        self.extraction_evidence = []
        
        document_text = self._prepare_document_text(document)
        document_metadata = document.get("metadata", {})
        
        # Prepare retrieval context if Vector RAG is enabled
        retrieval_context = None
        if self.use_vector_rag and self.vector_rag_extractor:
            try:
                retrieval_context = self._prepare_retrieval_context(document)
            except Exception as e:
                logger.warning(f"Failed to prepare retrieval context: {e}")
                retrieval_context = None
        
        # Extract using the specified reasoning method
        if self.method_type == ExtractionMethod.REASONING_COT:
            extracted_data = self._extract_with_cot(document_text, retrieval_context)
        elif self.method_type == ExtractionMethod.REASONING_REACT:
            extracted_data = self._extract_with_react(document_text, retrieval_context)
        else:
            raise ValueError(f"Unsupported reasoning method: {self.method_type}")
        
        # Perform verification if enabled
        if self.reasoning_config.verification_enabled:
            extracted_data = self._verify_and_correct(extracted_data, document_text, retrieval_context)
        
        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence()
        
        # Construct comprehensive result
        result = {
            "fields": extracted_data,
            "metadata": {
                "extraction_method": f"reasoning_{self.method_type.value.split('_')[1]}",
                "reasoning_steps": len(self.reasoning_steps),
                "evidence_pieces": len(self.extraction_evidence),
                "overall_confidence": overall_confidence,
                "uncertainty_threshold": self.uncertainty_threshold,
                "verification_enabled": self.reasoning_config.verification_enabled,
                "auto_repair_enabled": self.reasoning_config.auto_repair_enabled,
                "vector_rag_enabled": self.use_vector_rag,
                "extraction_time": time.time() - start_time,
                "schema_guided": self.schema is not None
            }
        }
        
        # Add reasoning details if requested
        if self.reasoning_config.verification_enabled:
            result["metadata"]["reasoning_trace"] = [
                {
                    "step": step.step_number,
                    "type": step.step_type,
                    "content": step.content[:200] + "..." if len(step.content) > 200 else step.content,
                    "confidence": step.confidence,
                    "uncertainty_flags": step.uncertainty_flags
                }
                for step in self.reasoning_steps
            ]
            
            result["metadata"]["evidence_summary"] = [
                {
                    "field": evidence.field_name,
                    "value": str(evidence.extracted_value)[:100] + "..." if len(str(evidence.extracted_value)) > 100 else str(evidence.extracted_value),
                    "confidence": evidence.confidence,
                    "reasoning": evidence.reasoning[:150] + "..." if len(evidence.reasoning) > 150 else evidence.reasoning
                }
                for evidence in self.extraction_evidence
            ]
        
        self.last_confidence = overall_confidence
        return result
    
    def _prepare_retrieval_context(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare retrieval context using Vector RAG components."""
        document_text = self._prepare_document_text(document)
        document_metadata = document.get("metadata", {})
        
        # Chunk the document
        chunks = self.chunker.chunk_document(document_text, document_metadata)
        
        # Index chunks for retrieval
        self.retriever.index_chunks(chunks)
        
        # Generate reasoning-focused queries
        reasoning_queries = self._generate_reasoning_queries()
        
        # Retrieve relevant context for each query
        context_by_query = {}
        for query_name, query_text in reasoning_queries.items():
            results = self.retriever.retrieve(query_text, k=5)
            context_by_query[query_name] = {
                "query": query_text,
                "results": results,
                "context": self._format_retrieval_context(results)
            }
        
        return {
            "chunks": chunks,
            "reasoning_contexts": context_by_query,
            "total_chunks": len(chunks)
        }
    
    def _generate_reasoning_queries(self) -> Dict[str, str]:
        """Generate queries for reasoning-focused retrieval."""
        base_queries = {
            "contract_identification": "contract number, title, subject, agreement name",
            "parties_information": "parties, contractor, client, vendor, supplier, names, addresses",
            "dates_and_timeline": "effective date, expiration date, start date, end date, term, duration",
            "financial_terms": "contract value, amount, cost, price, payment, billing, currency",
            "key_terms_conditions": "terms, conditions, obligations, responsibilities, requirements",
            "legal_provisions": "governing law, jurisdiction, termination, cancellation, breach"
        }
        
        # Enhance queries with schema information
        if self.schema and "properties" in self.schema:
            schema_enhanced_queries = {}
            for field_name, field_def in self.schema["properties"].items():
                field_description = field_def.get("description", "")
                field_keywords = field_name.replace("_", " ")
                
                # Find matching base query or create new one
                enhanced_query = f"{field_keywords} {field_description}".strip()
                
                # Add enum context for classification fields
                if "enum" in field_def:
                    enum_context = " ".join(field_def["enum"])
                    enhanced_query += f" categories: {enum_context}"
                
                schema_enhanced_queries[field_name] = enhanced_query
            
            return {**base_queries, **schema_enhanced_queries}
        
        return base_queries
    
    def _format_retrieval_context(self, results: List[RetrievalResult]) -> str:
        """Format retrieval results for reasoning context."""
        if not results:
            return "No relevant context found."
        
        context_parts = []
        for i, result in enumerate(results):
            chunk_text = result.chunk.content
            score = result.score
            context_parts.append(f"[Context {i+1}, Relevance: {score:.3f}]\n{chunk_text}")
        
        return "\n\n".join(context_parts)
    
    def _extract_with_cot(self, document_text: str, retrieval_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract using Chain of Thought reasoning with retrieval augmentation."""
        # Step 1: Document Analysis
        analysis_step = self._reasoning_step_document_analysis(document_text, retrieval_context)
        
        # Step 2: Schema-guided field identification
        identification_step = self._reasoning_step_field_identification(document_text, retrieval_context)
        
        # Step 3: Information extraction with evidence
        extraction_step = self._reasoning_step_information_extraction(document_text, retrieval_context)
        
        # Step 4: Structure and validate
        structuring_step = self._reasoning_step_structure_validation(extraction_step)
        
        # Parse final extraction
        extracted_data = self._parse_extraction_result(structuring_step)
        
        return extracted_data
    
    def _extract_with_react(self, document_text: str, retrieval_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract using ReAct (Reasoning + Acting) methodology."""
        extracted_fields = {}
        
        # Get target fields from schema
        target_fields = self._get_target_fields()
        
        for field_name in target_fields:
            field_result = self._react_extract_field(field_name, document_text, retrieval_context)
            extracted_fields[field_name] = field_result
        
        return extracted_fields
    
    def _reasoning_step_document_analysis(self, document_text: str, retrieval_context: Optional[Dict[str, Any]]) -> str:
        """Step 1: Analyze document type and structure."""
        step_num = len(self.reasoning_steps) + 1
        
        # Prepare context
        context_info = ""
        if retrieval_context and "reasoning_contexts" in retrieval_context:
            context_info = f"""
RELEVANT CONTEXT FROM RETRIEVAL:
{retrieval_context['reasoning_contexts'].get('contract_identification', {}).get('context', '')}
"""
        
        prompt = f"""Analyze this document step by step to understand its type and structure.

DOCUMENT TEXT:
{document_text[:2000]}{"..." if len(document_text) > 2000 else ""}

{context_info}

ANALYSIS TASK:
1. Identify the document type (contract, agreement, etc.)
2. Identify key sections and their purposes
3. Note any structural patterns or formatting
4. Identify language style and complexity

Think step by step:

Step {step_num} - Document Analysis:"""
        
        system_prompt = """You are an expert document analyst. Analyze documents systematically to understand their structure and content organization."""
        
        response = self._call_llm(prompt, system_prompt)
        
        # Record reasoning step
        step = ReasoningStep(
            step_type="thought",
            content=response,
            confidence=0.8,
            step_number=step_num
        )
        self.reasoning_steps.append(step)
        
        return response
    
    def _reasoning_step_field_identification(self, document_text: str, retrieval_context: Optional[Dict[str, Any]]) -> str:
        """Step 2: Identify target fields and their likely locations."""
        step_num = len(self.reasoning_steps) + 1
        
        # Prepare schema information
        schema_info = ""
        if self.schema:
            schema_info = f"""
TARGET SCHEMA FIELDS:
{json.dumps(self.schema.get('properties', {}), indent=2)}
"""
        
        # Prepare context from previous step
        previous_analysis = self.reasoning_steps[-1].content if self.reasoning_steps else ""
        
        prompt = f"""Based on the previous analysis, identify where each target field might be located in the document.

PREVIOUS ANALYSIS:
{previous_analysis}

{schema_info}

FIELD IDENTIFICATION TASK:
1. For each target field, identify likely sections or patterns in the document
2. Note any challenges or ambiguities in extraction
3. Prioritize fields by extraction confidence
4. Identify any missing or unclear information

Step {step_num} - Field Identification:"""
        
        system_prompt = """You are an expert at identifying information locations in documents. Map target fields to document sections systematically."""
        
        response = self._call_llm(prompt, system_prompt)
        
        # Record reasoning step
        step = ReasoningStep(
            step_type="action",
            content=response,
            confidence=0.75,
            step_number=step_num
        )
        self.reasoning_steps.append(step)
        
        return response
    
    def _reasoning_step_information_extraction(self, document_text: str, retrieval_context: Optional[Dict[str, Any]]) -> str:
        """Step 3: Extract specific information with evidence."""
        step_num = len(self.reasoning_steps) + 1
        
        # Prepare context from retrieval
        extraction_contexts = ""
        if retrieval_context and "reasoning_contexts" in retrieval_context:
            for context_name, context_data in retrieval_context["reasoning_contexts"].items():
                extraction_contexts += f"""
{context_name.upper()}:
{context_data.get('context', '')}

"""
        
        # Prepare previous reasoning
        previous_steps = "\n".join([f"Step {step.step_number}: {step.content[:200]}..." 
                                   for step in self.reasoning_steps[-2:]])
        
        prompt = f"""Extract specific information for each target field, providing evidence and reasoning.

PREVIOUS REASONING:
{previous_steps}

RELEVANT CONTEXTS:
{extraction_contexts}

EXTRACTION TASK:
For each target field:
1. Extract the specific value from the document
2. Provide the exact text evidence supporting the extraction
3. Rate your confidence in the extraction (0.0-1.0)
4. Note any uncertainties or alternative interpretations

Format your response as:
FIELD_NAME:
- Extracted Value: [value]
- Evidence: "[exact text from document]"
- Confidence: [0.0-1.0]
- Reasoning: [why this value is correct]
- Uncertainties: [any doubts or alternatives]

Step {step_num} - Information Extraction:"""
        
        system_prompt = """You are an expert information extractor. Extract precise information with supporting evidence and confidence assessments."""
        
        response = self._call_llm(prompt, system_prompt)
        
        # Parse and record evidence
        self._parse_extraction_evidence(response)
        
        # Record reasoning step
        step = ReasoningStep(
            step_type="observation",
            content=response,
            confidence=0.85,
            step_number=step_num
        )
        self.reasoning_steps.append(step)
        
        return response
    
    def _reasoning_step_structure_validation(self, extraction_result: str) -> str:
        """Step 4: Structure and validate the extracted information."""
        step_num = len(self.reasoning_steps) + 1
        
        # Prepare schema validation context
        schema_requirements = ""
        if self.schema:
            required_fields = self.schema.get("required", [])
            schema_requirements = f"""
SCHEMA REQUIREMENTS:
- Required fields: {required_fields}
- Field types and constraints: {json.dumps(self.schema.get('properties', {}), indent=2)}
"""
        
        prompt = f"""Structure the extracted information into valid JSON format and validate against requirements.

EXTRACTED INFORMATION:
{extraction_result}

{schema_requirements}

STRUCTURING TASK:
1. Convert extracted information to proper JSON format
2. Validate all required fields are present
3. Ensure data types match schema requirements
4. Handle any missing or invalid information
5. Apply reasonable defaults where appropriate

Provide the final structured JSON:

Step {step_num} - Structure and Validation:"""
        
        system_prompt = """You are an expert at structuring extracted information into valid JSON formats according to schemas."""
        
        response = self._call_llm(prompt, system_prompt)
        
        # Record reasoning step
        step = ReasoningStep(
            step_type="verification",
            content=response,
            confidence=0.9,
            step_number=step_num
        )
        self.reasoning_steps.append(step)
        
        return response
    
    def _react_extract_field(self, field_name: str, document_text: str, retrieval_context: Optional[Dict[str, Any]]) -> Any:
        """Extract a single field using ReAct methodology."""
        max_iterations = self.reasoning_config.max_reasoning_steps
        
        # Get field definition
        field_def = self.schema.get("properties", {}).get(field_name, {}) if self.schema else {}
        field_description = field_def.get("description", "")
        field_type = field_def.get("type", "string")
        
        # Prepare retrieval context for this field
        field_context = ""
        if retrieval_context and field_name in retrieval_context.get("reasoning_contexts", {}):
            field_context = retrieval_context["reasoning_contexts"][field_name].get("context", "")
        
        extracted_value = None
        confidence = 0.0
        
        for iteration in range(max_iterations):
            step_num = len(self.reasoning_steps) + 1
            
            # ReAct iteration
            if iteration == 0:
                # Initial thought
                thought_prompt = f"""Extract '{field_name}' from the document using ReAct methodology.

FIELD: {field_name}
DESCRIPTION: {field_description}
TYPE: {field_type}

RELEVANT CONTEXT:
{field_context}

Thought {iteration + 1}: What information do I need to extract for '{field_name}' and where might it be located?"""
            else:
                # Continue reasoning
                thought_prompt = f"""Continue ReAct extraction for '{field_name}'.

Previous attempts: {iteration}
Current extracted value: {extracted_value}
Current confidence: {confidence}

Thought {iteration + 1}: How can I improve or verify the extraction for '{field_name}'?"""
            
            system_prompt = "You are using ReAct methodology. Provide clear thoughts, actions, and observations."
            
            thought_response = self._call_llm(thought_prompt, system_prompt)
            
            # Record thought step
            thought_step = ReasoningStep(
                step_type="thought",
                content=thought_response,
                confidence=0.7,
                step_number=step_num
            )
            self.reasoning_steps.append(thought_step)
            
            # Action step
            action_prompt = f"""Based on your thought, what specific action will you take to extract '{field_name}'?

Thought: {thought_response}

Action {iteration + 1}:"""
            
            action_response = self._call_llm(action_prompt, system_prompt)
            
            # Record action step
            action_step = ReasoningStep(
                step_type="action",
                content=action_response,
                confidence=0.8,
                step_number=step_num + 1
            )
            self.reasoning_steps.append(action_step)
            
            # Observation step - perform the extraction
            observation_prompt = f"""Perform the action and observe the results for extracting '{field_name}'.

Action: {action_response}

DOCUMENT EXCERPT (relevant to {field_name}):
{self._get_relevant_excerpt(document_text, field_name)}

{f"RETRIEVAL CONTEXT: {field_context}" if field_context else ""}

Observation {iteration + 1}: What did you extract for '{field_name}'? Provide the value and your confidence level.

Format:
- Extracted Value: [value]
- Confidence: [0.0-1.0]
- Evidence: "[supporting text]"
- Reasoning: [explanation]"""
            
            observation_response = self._call_llm(observation_prompt, system_prompt)
            
            # Parse observation
            new_value, new_confidence = self._parse_react_observation(observation_response, field_name)
            
            if new_value is not None:
                extracted_value = new_value
                confidence = new_confidence
            
            # Record observation step
            observation_step = ReasoningStep(
                step_type="observation",
                content=observation_response,
                confidence=confidence,
                step_number=step_num + 2,
                extracted_info={field_name: extracted_value}
            )
            self.reasoning_steps.append(observation_step)
            
            # Check if we have sufficient confidence
            if confidence >= self.uncertainty_threshold:
                break
        
        # Record evidence
        if extracted_value is not None:
            evidence = ExtractionEvidence(
                field_name=field_name,
                extracted_value=extracted_value,
                evidence_text=self._get_relevant_excerpt(document_text, field_name)[:500],
                confidence=confidence,
                reasoning=f"ReAct extraction with {len(self.reasoning_steps)} steps"
            )
            self.extraction_evidence.append(evidence)
        
        return extracted_value
    
    def _get_target_fields(self) -> List[str]:
        """Get target fields from schema or use defaults."""
        if self.schema and "properties" in self.schema:
            return list(self.schema["properties"].keys())
        
        # Default contract fields
        return [
            "contract_number", "contract_title", "parties", "effective_date", 
            "expiration_date", "contract_value", "key_terms", "governing_law"
        ]
    
    def _get_relevant_excerpt(self, document_text: str, field_name: str, max_length: int = 1000) -> str:
        """Get relevant document excerpt for a specific field."""
        # Simple keyword-based extraction
        field_keywords = {
            "contract_number": ["contract number", "agreement number", "contract id", "contract #"],
            "contract_title": ["title", "subject", "agreement", "contract for"],
            "parties": ["party", "parties", "contractor", "client", "vendor", "between"],
            "effective_date": ["effective", "start date", "commence", "begin"],
            "expiration_date": ["expire", "end date", "termination", "until"],
            "contract_value": ["amount", "value", "cost", "price", "payment"],
            "key_terms": ["terms", "conditions", "obligations", "requirements"],
            "governing_law": ["governing law", "jurisdiction", "applicable law"]
        }
        
        keywords = field_keywords.get(field_name, [field_name.replace("_", " ")])
        
        # Find text sections containing keywords
        relevant_sections = []
        text_lower = document_text.lower()
        
        for keyword in keywords:
            start_pos = text_lower.find(keyword.lower())
            if start_pos != -1:
                # Extract context around the keyword
                context_start = max(0, start_pos - 200)
                context_end = min(len(document_text), start_pos + 300)
                relevant_sections.append(document_text[context_start:context_end])
        
        if not relevant_sections:
            # Return beginning of document as fallback
            return document_text[:max_length]
        
        # Combine and deduplicate sections
        combined_text = " ... ".join(relevant_sections)
        return combined_text[:max_length]
    
    def _parse_react_observation(self, observation_text: str, field_name: str) -> Tuple[Any, float]:
        """Parse ReAct observation to extract value and confidence."""
        # Look for extracted value
        value_match = re.search(r"Extracted Value:\s*(.+)", observation_text, re.IGNORECASE)
        confidence_match = re.search(r"Confidence:\s*([\d.]+)", observation_text, re.IGNORECASE)
        
        extracted_value = None
        confidence = 0.0
        
        if value_match:
            value_text = value_match.group(1).strip()
            # Clean up the value
            value_text = value_text.strip('"[](){}').strip()
            if value_text.lower() not in ['none', 'null', 'not found', 'n/a', '']:
                extracted_value = value_text
        
        if confidence_match:
            try:
                confidence = float(confidence_match.group(1))
                confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
            except ValueError:
                confidence = 0.5
        
        return extracted_value, confidence
    
    def _parse_extraction_evidence(self, extraction_text: str):
        """Parse extraction text to record evidence for each field."""
        # Look for field extraction patterns
        field_pattern = r"(\w+):\s*-\s*Extracted Value:\s*(.+?)\s*-\s*Evidence:\s*\"(.+?)\"\s*-\s*Confidence:\s*([\d.]+)\s*-\s*Reasoning:\s*(.+?)(?=\n\w+:|$)"
        
        matches = re.findall(field_pattern, extraction_text, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            field_name, value, evidence_text, confidence_str, reasoning = match
            
            try:
                confidence = float(confidence_str)
            except ValueError:
                confidence = 0.5
            
            # Clean up extracted value
            cleaned_value = value.strip().strip('"[](){}').strip()
            if cleaned_value.lower() not in ['none', 'null', 'not found', 'n/a', '']:
                evidence = ExtractionEvidence(
                    field_name=field_name.lower(),
                    extracted_value=cleaned_value,
                    evidence_text=evidence_text.strip(),
                    confidence=confidence,
                    reasoning=reasoning.strip()
                )
                self.extraction_evidence.append(evidence)
    
    def _parse_extraction_result(self, extraction_text: str) -> Dict[str, Any]:
        """Parse final extraction result into structured data."""
        # Try to find JSON in the response
        json_patterns = [
            r"```json\s*(\{.*?\})\s*```",
            r"```\s*(\{.*?\})\s*```",
            r"(\{.*?\})"
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, extraction_text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
        
        # Fallback: use evidence to construct result
        result = {}
        for evidence in self.extraction_evidence:
            if evidence.confidence >= 0.3:  # Minimum confidence threshold
                result[evidence.field_name] = evidence.extracted_value
        
        return result
    
    def _verify_and_correct(self, extracted_data: Dict[str, Any], document_text: str, retrieval_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Verify extracted data and apply corrections if needed."""
        if not self.reasoning_config.verification_enabled:
            return extracted_data
        
        verification_prompt = f"""Verify and correct the extracted information against the original document.

EXTRACTED DATA:
{json.dumps(extracted_data, indent=2)}

ORIGINAL DOCUMENT (excerpt):
{document_text[:1500]}{"..." if len(document_text) > 1500 else ""}

VERIFICATION TASK:
1. Check each extracted field against the document
2. Identify any errors or inconsistencies
3. Correct any mistakes found
4. Flag any uncertain or missing information

Provide corrected JSON with explanations for any changes:"""
        
        system_prompt = "You are an expert verifier. Carefully check extracted information for accuracy and completeness."
        
        verification_response = self._call_llm(verification_prompt, system_prompt)
        
        # Record verification step
        verification_step = ReasoningStep(
            step_type="verification",
            content=verification_response,
            confidence=0.9,
            step_number=len(self.reasoning_steps) + 1
        )
        self.reasoning_steps.append(verification_step)
        
        # Try to parse corrected data
        corrected_data = self._parse_extraction_result(verification_response)
        
        if corrected_data and self.reasoning_config.auto_repair_enabled:
            return corrected_data
        
        return extracted_data
    
    def _calculate_overall_confidence(self) -> float:
        """Calculate overall confidence score for the extraction."""
        if not self.extraction_evidence:
            return 0.5
        
        # Weighted average of evidence confidence
        total_confidence = sum(evidence.confidence for evidence in self.extraction_evidence)
        avg_confidence = total_confidence / len(self.extraction_evidence)
        
        # Boost for multiple pieces of evidence
        evidence_boost = min(0.1, len(self.extraction_evidence) * 0.02)
        
        # Boost for successful reasoning steps
        reasoning_boost = min(0.1, len(self.reasoning_steps) * 0.01)
        
        # Penalize if many uncertainty flags
        uncertainty_penalty = sum(len(step.uncertainty_flags) for step in self.reasoning_steps) * 0.02
        
        final_confidence = avg_confidence + evidence_boost + reasoning_boost - uncertainty_penalty
        
        return max(0.0, min(1.0, final_confidence))
    
    def get_reasoning_analysis(self) -> Dict[str, Any]:
        """Get detailed analysis of the reasoning process."""
        return {
            "method": self.method_type.value,
            "total_steps": len(self.reasoning_steps),
            "evidence_pieces": len(self.extraction_evidence),
            "overall_confidence": self._calculate_overall_confidence(),
            "uncertainty_threshold": self.uncertainty_threshold,
            "vector_rag_enabled": self.use_vector_rag,
            "verification_enabled": self.reasoning_config.verification_enabled,
            "step_breakdown": {
                "thoughts": len([s for s in self.reasoning_steps if s.step_type == "thought"]),
                "actions": len([s for s in self.reasoning_steps if s.step_type == "action"]),
                "observations": len([s for s in self.reasoning_steps if s.step_type == "observation"]),
                "verifications": len([s for s in self.reasoning_steps if s.step_type == "verification"])
            },
            "evidence_by_field": {
                evidence.field_name: {
                    "value": evidence.extracted_value,
                    "confidence": evidence.confidence,
                    "reasoning": evidence.reasoning[:100] + "..." if len(evidence.reasoning) > 100 else evidence.reasoning
                }
                for evidence in self.extraction_evidence
            },
            "average_step_confidence": sum(step.confidence for step in self.reasoning_steps) / len(self.reasoning_steps) if self.reasoning_steps else 0.0,
            "schema_guided": self.schema is not None
        }
