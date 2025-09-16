"""
Dynamic Graph RAG extractor - Novel approach with adaptive graph expansion and uncertainty-based refinement.

This implementation introduces several novel concepts:
1. Uncertainty-based graph expansion
2. Adaptive chunk sizing based on extraction complexity
3. Multi-stage verification and auto-repair
4. Context-aware entity relationship discovery
5. Iterative refinement with confidence tracking
6. Page-aware evidence extraction
"""

import json
import re
import math
from typing import Dict, Any, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import logging

from .base import BaseExtractor
from ..core.config import LLMConfig, DynamicGraphRAGConfig
from ..utils.graph_utils import (
    KnowledgeGraph, 
    Entity, 
    Relation, 
    EntityExtractor, 
    RelationExtractor,
    GraphRetriever
)
from ..utils.graph_db import GraphDatabaseManager
from ..utils.page_extractor import page_extractor, get_page_info, extract_text_with_pages

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyScore:
    """Represents uncertainty in extraction for a specific field."""
    field_name: str
    confidence: float
    uncertainty_reasons: List[str]
    suggested_expansion: List[str]


@dataclass
class ExtractionEvidence:
    """Represents evidence supporting an extraction decision."""
    field_name: str
    extracted_value: Any
    evidence_text: str
    confidence: float
    reasoning: str
    source_location: Dict[str, Any]  # start, end positions, page info
    supporting_entities: List[str]
    related_clauses: List[str]
    page_number: Optional[int] = None
    page_context: Optional[str] = None
    total_pages: Optional[int] = None


@dataclass
class ExtractionResult:
    """Enhanced extraction result with evidence tracking."""
    extracted_data: Dict[str, Any]
    evidence: Dict[str, ExtractionEvidence]
    metadata: Dict[str, Any]


@dataclass
@dataclass
class ExtractionState:
    """Tracks the state of extraction through multiple iterations."""
    iteration: int
    extracted_data: Dict[str, Any]
    confidence_scores: Dict[str, float]
    uncertainty_scores: List[UncertaintyScore]
    graph_stats: Dict[str, Any]
    refinement_history: List[str]
    evidence: Dict[str, ExtractionEvidence]


class DynamicUncertaintyEstimator:
    """Estimates uncertainty in extractions and suggests refinement strategies."""
    
    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema
        self.uncertainty_patterns = {
            "missing_value": r"(?:not\s+(?:found|available|specified|mentioned)|unknown|unclear|n/?a)",
            "hedge_words": r"(?:possibly|maybe|might|could|seems|appears|likely|probably)",
            "contradictory": r"(?:however|but|although|despite|contradicts)",
            "incomplete": r"(?:partial|incomplete|fragment|portion)"
        }
    
    def estimate_field_uncertainty(self, field_name: str, extracted_value: Any, 
                                 context: str, graph_evidence: List[Entity]) -> UncertaintyScore:
        """Estimate uncertainty for a specific field."""
        confidence = 1.0
        uncertainty_reasons = []
        suggested_expansion = []
        
        # Check if value is missing or null
        if extracted_value is None or extracted_value == "":
            confidence = 0.0
            uncertainty_reasons.append("Missing value")
            suggested_expansion.append(f"expand_entities_for_{field_name}")
        
        # Check for uncertainty patterns in the extracted value
        if isinstance(extracted_value, str):
            for pattern_name, pattern in self.uncertainty_patterns.items():
                if re.search(pattern, extracted_value, re.IGNORECASE):
                    confidence *= 0.7
                    uncertainty_reasons.append(f"Contains {pattern_name} indicators")
        
        # Check schema compliance
        if "field" in self.schema and field_name in self.schema["field"]:
            field_def = self.schema["field"][field_name]
            
            # Check enum compliance
            if "enum" in field_def and extracted_value not in field_def["enum"]:
                confidence *= 0.5
                uncertainty_reasons.append("Value not in allowed enum")
                suggested_expansion.append("retrieve_enum_context")
            
            # Check type compliance
            expected_type = field_def.get("type", "string")
            if not self._check_type_compliance(extracted_value, expected_type):
                confidence *= 0.6
                uncertainty_reasons.append("Type mismatch")
        
        # Check graph evidence strength
        evidence_strength = len(graph_evidence) / max(len(self.schema.get("field", {})), 1)
        if evidence_strength < 0.3:
            confidence *= 0.8
            uncertainty_reasons.append("Weak graph evidence")
            suggested_expansion.append("expand_graph_context")
        
        return UncertaintyScore(
            field_name=field_name,
            confidence=confidence,
            uncertainty_reasons=uncertainty_reasons,
            suggested_expansion=suggested_expansion
        )
    
    def _check_type_compliance(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type."""
        if expected_type == "string":
            return isinstance(value, str)
        elif expected_type == "number":
            return isinstance(value, (int, float))
        elif expected_type == "boolean":
            return isinstance(value, bool)
        elif expected_type == "array":
            return isinstance(value, list)
        elif expected_type == "object":
            return isinstance(value, dict)
        return True  # Unknown type, assume valid


class AdaptiveGraphExpander:
    """Dynamically expands graph context based on uncertainty and extraction needs."""
    
    def __init__(self, base_graph: KnowledgeGraph, schema: Dict[str, Any]):
        self.base_graph = base_graph
        self.schema = schema
        self.expansion_cache = {}
    
    def expand_for_uncertainty(self, uncertainty_score: UncertaintyScore, 
                             max_expansion_depth: int = 3) -> KnowledgeGraph:
        """Expand graph based on specific uncertainty patterns."""
        expansion_key = f"{uncertainty_score.field_name}_{hash(str(uncertainty_score.suggested_expansion))}"
        
        if expansion_key in self.expansion_cache:
            return self.expansion_cache[expansion_key]
        
        expanded_graph = KnowledgeGraph()
        
        # Start with field-specific entities
        field_entities = self.base_graph.get_entities_by_type(uncertainty_score.field_name)
        seed_entity_ids = [e.id for e in field_entities]
        
        # Add entities based on suggested expansions
        for expansion_type in uncertainty_score.suggested_expansion:
            if expansion_type.startswith("expand_entities_for_"):
                # Find related entities through relations
                related_ids = self._find_related_entities(seed_entity_ids, max_expansion_depth)
                seed_entity_ids.extend(related_ids)
            
            elif expansion_type == "retrieve_enum_context":
                # Add entities that might provide enum context
                enum_context_ids = self._find_enum_context_entities(uncertainty_score.field_name)
                seed_entity_ids.extend(enum_context_ids)
            
            elif expansion_type == "expand_graph_context":
                # Add high-confidence entities for broader context
                context_ids = self._find_high_confidence_entities()
                seed_entity_ids.extend(context_ids)
        
        # Build expanded subgraph
        expanded_graph = self.base_graph.get_subgraph(seed_entity_ids, max_expansion_depth)
        
        self.expansion_cache[expansion_key] = expanded_graph
        return expanded_graph
    
    def _find_related_entities(self, seed_ids: List[str], max_depth: int) -> List[str]:
        """Find entities related to seed entities through graph traversal."""
        related_ids = set()
        visited = set(seed_ids)
        queue = [(eid, 0) for eid in seed_ids]
        
        while queue:
            entity_id, depth = queue.pop(0)
            if depth >= max_depth:
                continue
            
            # Find neighbors
            neighbors = self.base_graph.get_neighbors(entity_id)
            for neighbor_entity, relation in neighbors:
                if neighbor_entity.id not in visited:
                    related_ids.add(neighbor_entity.id)
                    visited.add(neighbor_entity.id)
                    queue.append((neighbor_entity.id, depth + 1))
        
        return list(related_ids)
    
    def _find_enum_context_entities(self, field_name: str) -> List[str]:
        """Find entities that might provide context for enum values."""
        if "field" not in self.schema or field_name not in self.schema["field"]:
            return []
        
        field_def = self.schema["field"][field_name]
        if "enum" not in field_def:
            return []
        
        context_entities = []
        
        # Look for entities that contain enum values or related terms
        for enum_value in field_def["enum"]:
            # Find entities with text similar to enum values
            for entity in self.base_graph.entities.values():
                if enum_value.lower() in entity.text.lower() or entity.text.lower() in enum_value.lower():
                    context_entities.append(entity.id)
        
        return context_entities
    
    def _find_high_confidence_entities(self) -> List[str]:
        """Find entities with high confidence scores for general context."""
        high_conf_entities = []
        
        for entity in self.base_graph.entities.values():
            if entity.confidence > 0.8:
                high_conf_entities.append(entity.id)
        
        return high_conf_entities[:10]  # Limit to top 10


class EvidenceExtractor:
    """Extracts evidence and supporting clauses for extraction decisions."""
    
    def __init__(self, document_text: str, schema: Dict[str, Any]):
        self.document_text = document_text
        self.schema = schema
        self.clause_patterns = self._build_clause_patterns()
    
    def _build_clause_patterns(self) -> Dict[str, List[str]]:
        """Build patterns to identify relevant clauses for each field."""
        patterns = {}
        
        if "field" in self.schema:
            for field_name, field_def in self.schema["field"].items():
                field_patterns = []
                
                # Add field-specific patterns
                if field_name == "payment_terms":
                    field_patterns.extend([
                        r"payment\s+terms?:?\s*([^.]+)",
                        r"payments?\s+are\s+due\s+([^.]+)",
                        r"billing\s+frequency:?\s*([^.]+)",
                        r"payment\s+schedule:?\s*([^.]+)",
                        r"invoicing:?\s*([^.]+)"
                    ])
                
                elif field_name == "warranty":
                    field_patterns.extend([
                        r"warranty\s+(?:information|period|terms?):?\s*([^.]+)",
                        r"(?:standard|non-standard)\s+warranty\s+([^.]+)",
                        r"warranty\s+is\s+provided\s+([^.]+)",
                        r"guarantee:?\s*([^.]+)"
                    ])
                
                elif field_name == "customer_name":
                    field_patterns.extend([
                        r"customer[\"\']*:?\s*([^\"\',.()]+)",
                        r"between\s+[^\"\']+[\"\']\s*and\s+([^\"\',.()]+)",
                        r"entered\s+into\s+.*?and\s+([^\"\',.()]+)",
                        r"client:?\s*([^\"\',.()]+)"
                    ])
                
                # Add enum-specific patterns
                if "enum" in field_def:
                    for enum_val in field_def["enum"]:
                        field_patterns.append(rf"\b{re.escape(enum_val)}\b")
                
                patterns[field_name] = field_patterns
        
        return patterns
    
    def extract_evidence_for_field(self, field_name: str, extracted_value: Any, 
                                  supporting_entities: List[Entity]) -> ExtractionEvidence:
        """Extract evidence supporting a field's extraction."""
        
        # Find relevant text segments
        evidence_segments = []
        source_locations = []
        related_clauses = []
        
        # Use field-specific patterns to find evidence
        if field_name in self.clause_patterns:
            for pattern in self.clause_patterns[field_name]:
                matches = re.finditer(pattern, self.document_text, re.IGNORECASE)
                for match in matches:
                    evidence_segments.append(match.group(0))
                    source_locations.append({
                        "start": match.start(),
                        "end": match.end(),
                        "pattern": pattern
                    })
        
        # Extract evidence from supporting entities
        entity_evidence = []
        for entity in supporting_entities:
            if hasattr(entity, 'properties') and 'start' in entity.properties:
                # Get surrounding context
                start = max(0, entity.properties['start'] - 50)
                end = min(len(self.document_text), entity.properties['end'] + 50)
                context = self.document_text[start:end]
                entity_evidence.append(context)
        
        # Find complete clauses/sentences containing the evidence
        related_clauses = self._extract_related_clauses(evidence_segments + entity_evidence)
        
        # Build comprehensive evidence text
        all_evidence = evidence_segments + entity_evidence
        evidence_text = " | ".join(set(all_evidence)) if all_evidence else "No direct evidence found"
        
        # Generate reasoning explanation
        reasoning = self._generate_reasoning(field_name, extracted_value, evidence_segments, supporting_entities)
        
        # Calculate confidence based on evidence quality
        confidence = self._calculate_evidence_confidence(evidence_segments, supporting_entities, extracted_value)
        
        # Extract page information for the primary evidence location
        page_number = None
        page_context = None
        total_pages = None
        
        if source_locations:
            primary_location = source_locations[0]
            start_pos = primary_location.get("start", 0)
            end_pos = primary_location.get("end", start_pos)
            
            # Get page information
            page_info = get_page_info(self.document_text, start_pos, end_pos)
            page_number = page_info.get('page_number')
            total_pages = page_info.get('total_pages')
            
            # Get page context
            text_with_pages = extract_text_with_pages(self.document_text, start_pos, end_pos, context_chars=150)
            page_context = text_with_pages.get('context', '')
            
            # Enhance source location with page info
            primary_location.update(page_info)
        
        return ExtractionEvidence(
            field_name=field_name,
            extracted_value=extracted_value,
            evidence_text=evidence_text,
            confidence=confidence,
            reasoning=reasoning,
            source_location=source_locations[0] if source_locations else {},
            supporting_entities=[e.text for e in supporting_entities],
            related_clauses=related_clauses,
            page_number=page_number,
            page_context=page_context,
            total_pages=total_pages
        )
    
    def _extract_related_clauses(self, evidence_segments: List[str]) -> List[str]:
        """Extract complete clauses/sentences containing the evidence."""
        clauses = []
        
        # Split document into sentences/clauses
        sentences = re.split(r'[.!?]+', self.document_text)
        
        for evidence in evidence_segments:
            evidence_clean = evidence.strip().lower()
            for sentence in sentences:
                sentence_clean = sentence.strip().lower()
                if evidence_clean in sentence_clean and len(sentence.strip()) > 10:
                    clauses.append(sentence.strip())
        
        return list(set(clauses))
    
    def _generate_reasoning(self, field_name: str, extracted_value: Any, 
                          evidence_segments: List[str], supporting_entities: List[Entity]) -> str:
        """Generate human-readable reasoning for the extraction decision."""
        
        reasoning_parts = []
        
        # Field-specific reasoning
        if field_name == "payment_terms":
            if "monthly" in str(extracted_value).lower():
                reasoning_parts.append("Found 'monthly' payment pattern in contract text")
            elif "yearly" in str(extracted_value).lower():
                reasoning_parts.append("Found 'yearly' payment pattern in contract text")
            elif "one-time" in str(extracted_value).lower():
                reasoning_parts.append("Found 'one-time' payment pattern in contract text")
        
        elif field_name == "warranty":
            if "standard" in str(extracted_value).lower():
                reasoning_parts.append("Contract specifies standard warranty terms")
            elif "non_standard" in str(extracted_value).lower():
                reasoning_parts.append("Contract indicates non-standard warranty provisions")
        
        elif field_name == "customer_name":
            reasoning_parts.append(f"Identified '{extracted_value}' as customer name from contract parties")
        
        # Evidence-based reasoning
        if evidence_segments:
            reasoning_parts.append(f"Based on {len(evidence_segments)} evidence segments found in document")
        
        if supporting_entities:
            reasoning_parts.append(f"Supported by {len(supporting_entities)} related entities in knowledge graph")
        
        return "; ".join(reasoning_parts) if reasoning_parts else "Extracted based on document analysis"
    
    def _calculate_evidence_confidence(self, evidence_segments: List[str], 
                                     supporting_entities: List[Entity], extracted_value: Any) -> float:
        """Calculate confidence score based on evidence quality."""
        confidence = 0.5  # Base confidence
        
        # Evidence segment bonus
        if evidence_segments:
            confidence += min(0.3, len(evidence_segments) * 0.1)
        
        # Supporting entities bonus
        if supporting_entities:
            avg_entity_confidence = sum(e.confidence for e in supporting_entities) / len(supporting_entities)
            confidence += avg_entity_confidence * 0.2
        
        # Value quality bonus
        if extracted_value and str(extracted_value).strip():
            confidence += 0.1
        
        return min(confidence, 0.95)


class DynamicGraphRAGExtractor(BaseExtractor):
    """Dynamic Graph RAG extractor with adaptive refinement and evidence tracking."""
    
    def __init__(self, llm, schema: Dict[str, Any], config: Dict[str, Any] = None):
        super().__init__(llm, schema, config)
        self.uncertainty_estimator = DynamicUncertaintyEstimator()
        self.graph_expander = AdaptiveGraphExpander(llm)
        self.graph_db = GraphDatabaseManager()
        self.evidence_extractor = None
        
        # Configuration
        self.max_iterations = config.get('max_iterations', 3) if config else 3
        self.confidence_threshold = config.get('confidence_threshold', 0.8) if config else 0.8
        self.expansion_threshold = config.get('expansion_threshold', 0.6) if config else 0.6
        
        # Field mapping and auto-repair
        self.field_mappings = self._build_field_mappings()
        self.auto_repair = config.get('auto_repair', True) if config else True
        
        logger.info("Initialized DynamicGraphRAGExtractor with novel adaptive capabilities and evidence tracking")
    
    def extract(self, document_text: str) -> ExtractionResult:
        """
        Extract information using dynamic Graph RAG with evidence tracking.
        
        Returns:
            ExtractionResult: Complete extraction result with evidence
        """
        # Initialize evidence extractor
        self.evidence_extractor = EvidenceExtractor(document_text, self.schema)
        
        logger.info("Starting Dynamic Graph RAG extraction with evidence tracking")
        
        # Initial extraction and graph construction
        initial_state = self._initial_extraction(document_text)
        
        # Initialize current state
        current_state = initial_state
        iteration = 0
        
        # Iterative refinement process
        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"Starting iteration {iteration}")
            
            # Assess uncertainty and identify areas for improvement
            uncertainty_analysis = self._assess_extraction_uncertainty(current_state)
            
            # Check if we've reached satisfactory confidence
            if self._is_extraction_satisfactory(uncertainty_analysis):
                logger.info(f"Extraction converged after {iteration-1} iterations")
                break
            
            # Expand knowledge graph for uncertain areas
            expanded_state = self._expand_knowledge_graph(current_state, uncertainty_analysis)
            
            # Refine extraction based on expanded graph
            refined_state = self._refine_extraction(expanded_state, uncertainty_analysis)
            
            current_state = refined_state
        
        # Store final graph in database
        self._store_graph(current_state.knowledge_graph, document_text)
        
        # Prepare final result with evidence
        final_extraction = current_state.extracted_fields
        
        # Generate evidence for each extracted field
        evidence_list = []
        for field_name, value in final_extraction.items():
            # Find supporting entities for this field
            supporting_entities = self._find_supporting_entities(field_name, current_state.knowledge_graph)
            
            # Extract evidence
            evidence = self.evidence_extractor.extract_evidence_for_field(
                field_name, value, supporting_entities
            )
            evidence_list.append(evidence)
        
        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(evidence_list)
        
        logger.info(f"Dynamic Graph RAG extraction completed. Confidence: {overall_confidence:.2f}")
        
        return ExtractionResult(
            extracted_fields=final_extraction,
            overall_confidence=overall_confidence,
            evidence=evidence_list,
            graph_size=len(current_state.knowledge_graph.entities),
            iterations_used=iteration
        )
    """
    Novel Dynamic Graph RAG extraction method with:
    - Uncertainty-based adaptive graph expansion
    - Multi-stage iterative refinement
    - Context-aware entity relationship discovery
    - Adaptive chunk sizing and retrieval
    - Self-repair mechanisms
    """
    
    def __init__(self, llm_config: LLMConfig, dynamic_config: DynamicGraphRAGConfig, 
                 schema: Optional[Dict[str, Any]] = None):
        """Initialize Dynamic Graph RAG extractor."""
        super().__init__(llm_config)
        self.dynamic_config = dynamic_config
        self.schema = schema
        
        # Initialize components
        self.knowledge_graph: Optional[KnowledgeGraph] = None
        self.uncertainty_estimator: Optional[DynamicUncertaintyEstimator] = None
        self.graph_expander: Optional[AdaptiveGraphExpander] = None
        self.extraction_states: List[ExtractionState] = []
        
        # Initialize graph database manager
        if not dynamic_config.base_graph_config.use_lightweight_kg:
            # Only access credentials when not using lightweight mode
            graph_db_config = {
                "graph_db_uri": dynamic_config.base_graph_config.graph_db_uri,
                "graph_db_user": dynamic_config.base_graph_config.graph_db_user,
                "graph_db_password": dynamic_config.base_graph_config.graph_db_password,
                "use_lightweight_kg": dynamic_config.base_graph_config.use_lightweight_kg
            }
            self.graph_db_manager = GraphDatabaseManager(graph_db_config)
        else:
            self.graph_db_manager = None
        
        logger.info("Initialized DynamicGraphRAGExtractor with novel adaptive capabilities")
        if self.graph_db_manager:
            logger.info("Graph database integration enabled")
        else:
            logger.info("Using lightweight in-memory knowledge graphs")
    
    def set_schema(self, schema: Dict[str, Any]) -> None:
        """Set the extraction schema."""
        self.schema = schema
        logger.info(f"Dynamic Graph RAG schema set with fields: {list(schema.get('field', {}).keys())}")
    
    def extract(self, document: Dict[str, Any], schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract using Dynamic Graph RAG with adaptive expansion."""
        # Use provided schema or fall back to instance schema
        extraction_schema = schema or self.schema
        if not extraction_schema:
            raise ValueError("No schema provided for dynamic extraction")
        
        document_text = self._prepare_document_text(document)
        document_id = document.get("metadata", {}).get("source", "unknown_document")
        
        # Try to retrieve existing knowledge graph from database
        if self.graph_db_manager:
            existing_kg = self.graph_db_manager.retrieve_knowledge_graph(document_id)
            if existing_kg and len(existing_kg.entities) > 0:
                logger.info(f"Retrieved existing knowledge graph for {document_id}")
                self.knowledge_graph = existing_kg
                # Still need to initialize other components even with existing graph
                self.uncertainty_estimator = DynamicUncertaintyEstimator(extraction_schema)
                self.graph_expander = AdaptiveGraphExpander(self.knowledge_graph, extraction_schema)
                self.extraction_states = []
            else:
                # No existing graph or empty graph - create new one
                self._initialize_extraction(document_text, extraction_schema)
                # Store the newly created knowledge graph
                self.graph_db_manager.store_knowledge_graph(self.knowledge_graph, document_id)
        else:
            # Initialize components for this extraction
            self._initialize_extraction(document_text, extraction_schema)
        
        # Perform iterative extraction with adaptive refinement
        final_result = self._iterative_extraction(document_text, extraction_schema)
        
        return final_result
    
    def _initialize_extraction(self, document_text: str, schema: Dict[str, Any]) -> None:
        """Initialize components for the extraction process."""
        # Build initial knowledge graph
        self.knowledge_graph = self._build_adaptive_knowledge_graph(document_text, schema)
        
        # Initialize uncertainty estimator
        self.uncertainty_estimator = DynamicUncertaintyEstimator(schema)
        
        # Initialize graph expander
        self.graph_expander = AdaptiveGraphExpander(self.knowledge_graph, schema)
        
        # Reset extraction states
        self.extraction_states = []
    
    def _build_adaptive_knowledge_graph(self, text: str, schema: Dict[str, Any]) -> KnowledgeGraph:
        """Build knowledge graph with adaptive chunking based on complexity."""
        kg = KnowledgeGraph()
        
        # Determine adaptive chunk size based on document complexity
        adaptive_chunk_size = self._calculate_adaptive_chunk_size(text, schema)
        
        # Extract entities with enhanced patterns
        entity_extractor = EntityExtractor(schema)
        entities = entity_extractor.extract_entities(text)
        
        # Add entities to graph
        for entity in entities:
            kg.add_entity(entity)
        
        # Extract relations with context awareness
        relation_extractor = RelationExtractor()
        relations = relation_extractor.extract_relations(text, entities)
        
        # Add relations to graph
        for relation in relations:
            kg.add_relation(relation)
        
        # Enhance with schema-specific entity linking
        self._enhance_entity_linking(kg, schema)
        
        logger.info(f"Built adaptive knowledge graph: {len(entities)} entities, {len(relations)} relations")
        
        return kg
    
    def _calculate_adaptive_chunk_size(self, text: str, schema: Dict[str, Any]) -> int:
        """Calculate optimal chunk size based on document and schema complexity."""
        base_size = 512
        
        # Adjust based on document length
        doc_length = len(text)
        if doc_length > 5000:
            base_size = 1024
        elif doc_length < 1000:
            base_size = 256
        
        # Adjust based on schema complexity
        if "field" in schema:
            field_count = len(schema["field"])
            enum_fields = sum(1 for field_def in schema["field"].values() if "enum" in field_def)
            
            # More complex schemas need larger chunks
            complexity_factor = 1.0 + (field_count * 0.1) + (enum_fields * 0.2)
            base_size = int(base_size * complexity_factor)
        
        return min(base_size, 2048)  # Cap at 2048
    
    def _enhance_entity_linking(self, kg: KnowledgeGraph, schema: Dict[str, Any]) -> None:
        """Enhance entity linking with schema-specific patterns."""
        if "field" not in schema:
            return
        
        # Create cross-references between related entities
        for field_name, field_def in schema["field"].items():
            field_entities = kg.get_entities_by_type(field_name)
            
            # Link entities that might be related based on proximity and context
            for i, entity1 in enumerate(field_entities):
                for entity2 in field_entities[i+1:]:
                    # Check if entities are contextually related
                    if self._are_entities_related(entity1, entity2, field_def):
                        relation = Relation(
                            source_id=entity1.id,
                            target_id=entity2.id,
                            relation_type="SCHEMA_RELATED",
                            properties={"field": field_name},
                            confidence=0.7
                        )
                        kg.add_relation(relation)
    
    def _are_entities_related(self, entity1: Entity, entity2: Entity, field_def: Dict[str, Any]) -> bool:
        """Check if two entities are contextually related based on field definition."""
        # Simple heuristic: entities are related if they're close in position
        pos1 = entity1.properties.get("start", 0)
        pos2 = entity2.properties.get("start", 0)
        
        return abs(pos1 - pos2) < 200  # Within 200 characters
    
    def _iterative_extraction(self, document_text: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Perform iterative extraction with adaptive refinement."""
        max_iterations = self.dynamic_config.max_expansion_depth
        
        # Initial extraction
        current_result = self._initial_extraction(document_text, schema)
        
        for iteration in range(max_iterations):
            # Estimate uncertainty for current result
            uncertainty_scores = self._estimate_extraction_uncertainty(current_result, schema)
            
            # Calculate overall uncertainty
            overall_uncertainty = sum(us.confidence for us in uncertainty_scores) / len(uncertainty_scores) if uncertainty_scores else 1.0
            overall_uncertainty = 1.0 - overall_uncertainty  # Convert confidence to uncertainty
            
            # Track extraction state
            extraction_state = ExtractionState(
                iteration=iteration,
                extracted_data=current_result.copy(),
                confidence_scores={us.field_name: us.confidence for us in uncertainty_scores},
                uncertainty_scores=uncertainty_scores,
                graph_stats=self._get_graph_stats(),
                refinement_history=[],
                evidence={}  # Initialize empty evidence dict
            )
            self.extraction_states.append(extraction_state)
            
            # Stop if uncertainty is below threshold
            if overall_uncertainty < (1.0 - self.dynamic_config.uncertainty_threshold):
                logger.info(f"Converged after {iteration + 1} iterations with uncertainty {overall_uncertainty:.3f}")
                break
            
            # Identify fields that need refinement
            uncertain_fields = [us for us in uncertainty_scores if us.confidence < self.dynamic_config.uncertainty_threshold]
            
            if not uncertain_fields:
                break
            
            # Adaptive graph expansion and refinement
            refined_result = self._adaptive_refinement(document_text, current_result, uncertain_fields, schema)
            
            # Apply auto-repair if enabled
            if self.dynamic_config.base_graph_config.use_lightweight_kg:
                refined_result = self._auto_repair_extraction(refined_result, schema)
            
            current_result = refined_result
        
        self.last_confidence = self._calculate_final_confidence(current_result, schema)
        return current_result
    
    def _initial_extraction(self, document_text: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Perform initial extraction with basic graph context."""
        # Use standard graph retrieval for initial attempt
        graph_retriever = GraphRetriever(self.knowledge_graph)
        relevant_subgraph = graph_retriever.retrieve_for_schema(schema)
        
        # Build initial context
        graph_context = self._build_graph_context(relevant_subgraph, schema)
        
        # Create extraction prompt
        prompt = self._build_dynamic_extraction_prompt(document_text, schema, graph_context, iteration=0)
        
        system_prompt = """You are an expert extraction system using dynamic graph-based analysis.
        
Your task is to extract structured information from documents using knowledge graph context.
Pay attention to entity relationships and provide confident, accurate extractions.
If you're uncertain about any field, indicate this in your response."""
        
        response = self._call_llm(prompt, system_prompt)
        
        # Parse response with enhanced error handling
        return self._parse_llm_response(response, schema)
    
    def _estimate_extraction_uncertainty(self, extracted_data: Dict[str, Any], 
                                       schema: Dict[str, Any]) -> List[UncertaintyScore]:
        """Estimate uncertainty for each extracted field."""
        uncertainty_scores = []
        
        if "field" not in schema:
            return uncertainty_scores
        
        for field_name, field_def in schema["field"].items():
            extracted_value = extracted_data.get(field_name)
            
            # Get graph evidence for this field
            field_entities = self.knowledge_graph.get_entities_by_type(field_name)
            
            # Estimate uncertainty
            uncertainty_score = self.uncertainty_estimator.estimate_field_uncertainty(
                field_name, extracted_value, "", field_entities
            )
            
            uncertainty_scores.append(uncertainty_score)
        
        return uncertainty_scores
    
    def _adaptive_refinement(self, document_text: str, current_result: Dict[str, Any],
                           uncertain_fields: List[UncertaintyScore], schema: Dict[str, Any]) -> Dict[str, Any]:
        """Perform adaptive refinement based on uncertainty analysis."""
        refinement_contexts = []
        
        # Expand graph for each uncertain field
        for uncertainty_score in uncertain_fields:
            expanded_graph = self.graph_expander.expand_for_uncertainty(
                uncertainty_score, 
                self.dynamic_config.max_expansion_depth
            )
            
            # Build enhanced context for this field
            field_context = self._build_field_specific_context(
                uncertainty_score.field_name, expanded_graph, schema
            )
            refinement_contexts.append(field_context)
        
        # Create refinement prompt
        refinement_prompt = self._build_refinement_prompt(
            document_text, current_result, uncertain_fields, refinement_contexts, schema
        )
        
        system_prompt = """You are refining an extraction with expanded graph context and uncertainty analysis.
        
Focus on the uncertain fields identified and use the additional context to provide more confident extractions.
Maintain the confidence of fields that were already well-extracted."""
        
        response = self._call_llm(refinement_prompt, system_prompt)
        
        # Parse and merge with current result
        refined_data = self._parse_llm_response(response, schema)
        
        # Merge results, keeping high-confidence fields from previous iteration
        merged_result = current_result.copy()
        for field_name, new_value in refined_data.items():
            # Only update if the field was uncertain or the new value is more confident
            if any(us.field_name == field_name and us.confidence < 0.7 for us in uncertain_fields):
                merged_result[field_name] = new_value
        
        return merged_result
    
    def _build_field_specific_context(self, field_name: str, expanded_graph: KnowledgeGraph,
                                    schema: Dict[str, Any]) -> str:
        """Build context specific to a field using expanded graph."""
        context_parts = []
        
        context_parts.append(f"## Enhanced Context for '{field_name}'")
        
        # Field definition
        if "field" in schema and field_name in schema["field"]:
            field_def = schema["field"][field_name]
            context_parts.append(f"**Definition**: {field_def.get('description', 'N/A')}")
            
            if "enum" in field_def:
                context_parts.append(f"**Allowed Values**: {field_def['enum']}")
                if "enumDescriptions" in field_def:
                    context_parts.append("**Value Descriptions**:")
                    for enum_val, desc in field_def["enumDescriptions"].items():
                        context_parts.append(f"  - {enum_val}: {desc}")
        
        # Field-specific entities
        field_entities = expanded_graph.get_entities_by_type(field_name)
        if field_entities:
            context_parts.append(f"**Related Entities ({len(field_entities)})**:")
            for entity in field_entities[:5]:  # Top 5
                context_parts.append(f"  - '{entity.text}' (confidence: {entity.confidence:.2f})")
        
        # Related entities through graph expansion
        related_entities = []
        for entity_type in ["CAPITALIZED_WORD", "NUMBER", "DATE", "MONEY"]:
            type_entities = expanded_graph.get_entities_by_type(entity_type)
            related_entities.extend(type_entities[:2])  # Top 2 per type
        
        if related_entities:
            context_parts.append(f"**Related Context Entities**:")
            for entity in related_entities:
                context_parts.append(f"  - {entity.type}: '{entity.text}'")
        
        return "\n".join(context_parts)
    
    def _build_refinement_prompt(self, document_text: str, current_result: Dict[str, Any],
                               uncertain_fields: List[UncertaintyScore], 
                               refinement_contexts: List[str], schema: Dict[str, Any]) -> str:
        """Build prompt for refinement iteration."""
        prompt_parts = [
            "# Dynamic Graph RAG Refinement",
            "",
            "## Current Extraction Status",
            f"```json\n{json.dumps(current_result, indent=2)}\n```",
            "",
            "## Uncertainty Analysis"
        ]
        
        for us in uncertain_fields:
            prompt_parts.append(f"**{us.field_name}** (confidence: {us.confidence:.2f})")
            prompt_parts.append(f"  Issues: {', '.join(us.uncertainty_reasons)}")
            prompt_parts.append("")
        
        prompt_parts.append("## Enhanced Context from Graph Expansion")
        for context in refinement_contexts:
            prompt_parts.append(context)
            prompt_parts.append("")
        
        prompt_parts.extend([
            "## Original Document",
            f"```\n{document_text}\n```",
            "",
            "## Instructions",
            "1. Focus on improving the uncertain fields identified above",
            "2. Use the enhanced graph context to resolve ambiguities",
            "3. Maintain high-confidence extractions from the current result",
            "4. Ensure all values comply with the schema definitions",
            "",
            "Provide the refined extraction as JSON:"
        ])
        
        return "\n".join(prompt_parts)
    
    def _auto_repair_extraction(self, extracted_data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """Apply auto-repair mechanisms to fix common extraction issues."""
        repaired_data = extracted_data.copy()
        
        if "field" not in schema:
            return repaired_data
        
        for field_name, field_def in schema["field"].items():
            if field_name in repaired_data:
                value = repaired_data[field_name]
                
                # Repair enum violations
                if "enum" in field_def and value not in field_def["enum"]:
                    # Try to map to closest enum value
                    repaired_value = self._repair_enum_value(value, field_def["enum"])
                    if repaired_value:
                        repaired_data[field_name] = repaired_value
                        logger.info(f"Auto-repaired {field_name}: '{value}' -> '{repaired_value}'")
                
                # Repair type mismatches
                expected_type = field_def.get("type", "string")
                repaired_value = self._repair_type_mismatch(value, expected_type)
                if repaired_value != value:
                    repaired_data[field_name] = repaired_value
                    logger.info(f"Auto-repaired type for {field_name}: {type(value)} -> {type(repaired_value)}")
        
        return repaired_data
    
    def _repair_enum_value(self, value: Any, allowed_enum: List[str]) -> Optional[str]:
        """Attempt to repair enum value by finding closest match."""
        if not isinstance(value, str):
            return None
        
        value_lower = value.lower()
        
        # Exact match (case insensitive)
        for enum_val in allowed_enum:
            if enum_val.lower() == value_lower:
                return enum_val
        
        # Substring match
        for enum_val in allowed_enum:
            if value_lower in enum_val.lower() or enum_val.lower() in value_lower:
                return enum_val
        
        # Keyword matching for common patterns
        if "month" in value_lower or "monthly" in value_lower:
            if "monthly" in allowed_enum:
                return "monthly"
        elif "year" in value_lower or "annual" in value_lower:
            if "yearly" in allowed_enum:
                return "yearly"
        elif "once" in value_lower or "single" in value_lower:
            if "one-time" in allowed_enum:
                return "one-time"
        
        return None
    
    def _repair_type_mismatch(self, value: Any, expected_type: str) -> Any:
        """Attempt to repair type mismatches."""
        if expected_type == "string" and not isinstance(value, str):
            return str(value) if value is not None else ""
        elif expected_type == "number":
            if isinstance(value, str):
                # Try to extract number from string
                number_match = re.search(r'-?\d+\.?\d*', value)
                if number_match:
                    try:
                        return float(number_match.group()) if '.' in number_match.group() else int(number_match.group())
                    except ValueError:
                        pass
        
        return value
    
    def _build_graph_context(self, subgraph: KnowledgeGraph, schema: Dict[str, Any]) -> str:
        """Build textual representation of graph context."""
        context_parts = []
        
        # Entity summary
        context_parts.append("## Knowledge Graph Context")
        context_parts.append(f"Entities: {len(subgraph.entities)}, Relations: {len(subgraph.relations)}")
        
        # Entities by type
        entity_types = set(e.type for e in subgraph.entities.values())
        for entity_type in sorted(entity_types):
            entities = subgraph.get_entities_by_type(entity_type)
            if entities and entity_type in schema.get("field", {}):
                context_parts.append(f"\n### {entity_type}:")
                for entity in entities[:3]:
                    context_parts.append(f"  - '{entity.text}' ({entity.confidence:.2f})")
        
        return "\n".join(context_parts)
    
    def _build_dynamic_extraction_prompt(self, document_text: str, schema: Dict[str, Any],
                                       graph_context: str, iteration: int = 0) -> str:
        """Build extraction prompt with dynamic graph context."""
        prompt_parts = [
            f"# Dynamic Graph RAG Extraction (Iteration {iteration})",
            "",
            "## Task",
            "Extract structured information using dynamic graph-based analysis.",
            "IMPORTANT: Extract ONLY the fields specified in the schema below. Do NOT add extra fields.",
            "",
            "## Graph Context",
            graph_context,
            "",
            "## Schema (Extract EXACTLY these fields)",
            self._format_schema_for_prompt(schema),
            "",
            "## Document",
            f"```\n{document_text}\n```",
            "",
            "## Instructions",
            "1. Use the knowledge graph context to understand entity relationships",
            "2. Extract ONLY the fields specified in the schema",
            "3. For enum fields, select EXACTLY one value from the allowed options",
            "4. For string fields, provide simple string values, not complex objects",
            "5. Follow the exact field names and types specified in the schema",
            "",
            "## Required Output Format",
            self._build_schema_output_example(schema),
            "",
            "Return extraction as JSON with ONLY the schema fields:"
        ]
        
        return "\n".join(prompt_parts)
    
    def _build_schema_output_example(self, schema: Dict[str, Any]) -> str:
        """Build an example output format based on the schema."""
        if "field" not in schema:
            return "{}"
        
        example = {}
        for field_name, field_def in schema["field"].items():
            if "enum" in field_def:
                example[field_name] = f"<one of: {field_def['enum']}>"
            elif field_def.get("type") == "string":
                example[field_name] = f"<string value for {field_name}>"
            else:
                example[field_name] = f"<{field_def.get('type', 'string')} value>"
        
        return f"```json\n{json.dumps(example, indent=2)}\n```"
    
    def _format_schema_for_prompt(self, schema: Dict[str, Any]) -> str:
        """Format schema for inclusion in prompts."""
        if "field" not in schema:
            return json.dumps(schema, indent=2)
        
        formatted_parts = []
        for field_name, field_def in schema["field"].items():
            parts = [f"**{field_name}**:"]
            parts.append(f"  - Description: {field_def.get('description', 'N/A')}")
            parts.append(f"  - Type: {field_def.get('type', 'string')}")
            
            if "enum" in field_def:
                parts.append(f"  - Values: {field_def['enum']}")
                if "enumDescriptions" in field_def:
                    parts.append("  - Descriptions:")
                    for enum_val, desc in field_def["enumDescriptions"].items():
                        parts.append(f"    - {enum_val}: {desc}")
            
            formatted_parts.append("\n".join(parts))
        
        return "\n\n".join(formatted_parts)
    
    def _parse_llm_response(self, response: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Parse LLM response with enhanced error handling and schema alignment."""
        logger.info(f"Raw LLM response: {response}")
        
        try:
            # Try direct JSON parsing
            result = json.loads(response.strip())
            # Apply schema-based field extraction
            return self._align_with_schema(result, schema)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            
            # Try to extract JSON from markdown blocks
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(1))
                    return self._align_with_schema(result, schema)
                except json.JSONDecodeError:
                    pass
            
            # Try to find any JSON object in the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    return self._align_with_schema(result, schema)
                except json.JSONDecodeError:
                    pass
            
            # Return empty result structure
            logger.error(f"Could not parse response: {repr(response)}")
            return self._build_empty_result(schema)
    
    def _align_with_schema(self, extracted_data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """Align extracted data with the expected schema format."""
        if "field" not in schema:
            return extracted_data
        
        aligned_result = {}
        
        for field_name, field_def in schema["field"].items():
            field_value = None
            
            # Direct field match
            if field_name in extracted_data:
                field_value = extracted_data[field_name]
            
            # Handle specific field mappings based on the contracts schema
            elif field_name == "payment_terms":
                # Look for payment terms in various formats
                if "payment_terms" in extracted_data:
                    pt = extracted_data["payment_terms"]
                    if isinstance(pt, dict):
                        # Extract frequency from complex object
                        if "frequency" in pt:
                            freq = pt["frequency"].lower()
                            if freq in ["monthly", "yearly", "one-time"]:
                                field_value = freq
                        elif "type" in pt:
                            field_value = pt["type"]
                    elif isinstance(pt, str):
                        # Try to extract from string representation
                        pt_lower = pt.lower()
                        if "monthly" in pt_lower:
                            field_value = "monthly"
                        elif "yearly" in pt_lower or "annual" in pt_lower:
                            field_value = "yearly"
                        elif "one-time" in pt_lower or "single" in pt_lower:
                            field_value = "one-time"
                
                # Fallback: look in document text patterns
                if not field_value:
                    # Check if we can infer from the original extraction
                    if isinstance(extracted_data.get("payment_terms"), str):
                        pt_str = extracted_data["payment_terms"]
                        if "'frequency': 'monthly'" in pt_str:
                            field_value = "monthly"
                        elif "'frequency': 'yearly'" in pt_str:
                            field_value = "yearly"
            
            elif field_name == "warranty":
                # Look for warranty information
                if "warranty" in extracted_data:
                    warranty = extracted_data["warranty"]
                    if isinstance(warranty, dict):
                        if "type" in warranty:
                            w_type = warranty["type"].lower()
                            if w_type in ["standard", "non_standard"]:
                                field_value = w_type
                        elif "duration" in warranty:
                            duration = warranty["duration"].lower()
                            if "1 year" in duration or "standard" in duration:
                                field_value = "standard"
                            else:
                                field_value = "non_standard"
                    elif isinstance(warranty, str):
                        w_lower = warranty.lower()
                        if "standard" in w_lower:
                            field_value = "standard"
                        elif "non_standard" in w_lower or "non-standard" in w_lower:
                            field_value = "non_standard"
                
                # Fallback: look in string representation
                if not field_value:
                    if isinstance(extracted_data.get("warranty"), str):
                        w_str = extracted_data["warranty"]
                        if "'type': 'standard'" in w_str:
                            field_value = "standard"
                        elif "'type': 'non_standard'" in w_str:
                            field_value = "non_standard"
            
            elif field_name == "customer_name":
                # Look for customer name in various locations
                if "customer_name" in extracted_data:
                    field_value = extracted_data["customer_name"]
                elif "customer" in extracted_data:
                    customer = extracted_data["customer"]
                    if isinstance(customer, dict):
                        if "name" in customer:
                            field_value = customer["name"]
                        elif "customer_name" in customer:
                            field_value = customer["customer_name"]
                elif "name" in extracted_data:
                    field_value = extracted_data["name"]
            
            # Apply type conversion and validation
            if field_value is not None:
                # Convert to expected type
                expected_type = field_def.get("type", "string")
                if expected_type == "string" and not isinstance(field_value, str):
                    field_value = str(field_value)
                
                # Validate enum constraints
                if "enum" in field_def:
                    if field_value not in field_def["enum"]:
                        # Try to repair enum value
                        repaired_value = self._repair_enum_value(field_value, field_def["enum"])
                        if repaired_value:
                            field_value = repaired_value
                            logger.info(f"Repaired enum value for {field_name}: '{field_value}' -> '{repaired_value}'")
            
            aligned_result[field_name] = field_value
        
        logger.info(f"Aligned extraction result: {aligned_result}")
        return aligned_result
    
    def _build_empty_result(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Build empty result structure based on schema."""
        if "field" not in schema:
            return {}
        
        return {field_name: None for field_name in schema["field"].keys()}
    
    def _get_graph_stats(self) -> Dict[str, Any]:
        """Get current graph statistics."""
        if not self.knowledge_graph:
            return {}
        
        return {
            "total_entities": len(self.knowledge_graph.entities),
            "total_relations": len(self.knowledge_graph.relations),
            "entity_types": len(set(e.type for e in self.knowledge_graph.entities.values()))
        }
    
    def _calculate_final_confidence(self, result: Dict[str, Any], schema: Dict[str, Any]) -> float:
        """Calculate final confidence score based on extraction quality and convergence."""
        if not result or "field" not in schema:
            return 0.0
        
        # Base confidence from result completeness
        total_fields = len(schema["field"])
        filled_fields = sum(1 for v in result.values() if v is not None and v != "")
        completeness_score = filled_fields / total_fields if total_fields > 0 else 0.0
        
        # Convergence bonus (fewer iterations = higher confidence)
        convergence_bonus = max(0.0, 1.0 - (len(self.extraction_states) * 0.1))
        
        # Schema compliance score
        compliance_score = self._calculate_schema_compliance(result, schema)
        
        # Combine scores
        final_confidence = (
            completeness_score * 0.4 + 
            convergence_bonus * 0.3 + 
            compliance_score * 0.3
        )
        
        return min(final_confidence, 0.98)  # Cap at 98%
    
    def _calculate_schema_compliance(self, result: Dict[str, Any], schema: Dict[str, Any]) -> float:
        """Calculate how well the result complies with schema constraints."""
        if "field" not in schema:
            return 1.0
        
        compliance_scores = []
        
        for field_name, field_def in schema["field"].items():
            if field_name in result:
                value = result[field_name]
                score = 1.0
                
                # Check enum compliance
                if "enum" in field_def and value not in field_def["enum"]:
                    score *= 0.5
                
                # Check type compliance
                expected_type = field_def.get("type", "string")
                if not self.uncertainty_estimator._check_type_compliance(value, expected_type):
                    score *= 0.7
                
                compliance_scores.append(score)
            else:
                compliance_scores.append(0.0)  # Missing field
        
        return sum(compliance_scores) / len(compliance_scores) if compliance_scores else 0.0
    
    def get_extraction_metadata(self) -> Dict[str, Any]:
        """Get comprehensive metadata about the dynamic extraction process."""
        metadata = {
            "extractor_type": "dynamic_graph_rag",
            "confidence": self.last_confidence,
            "token_usage": self.last_token_usage,
            "iterations": len(self.extraction_states),
            "convergence_achieved": len(self.extraction_states) < self.dynamic_config.max_expansion_depth
        }
        
        if self.knowledge_graph:
            metadata["graph_stats"] = self._get_graph_stats()
        
        if self.extraction_states:
            # Add convergence information
            confidence_progression = []
            for state in self.extraction_states:
                avg_confidence = sum(state.confidence_scores.values()) / len(state.confidence_scores) if state.confidence_scores else 0.0
                confidence_progression.append(avg_confidence)
            
            metadata["confidence_progression"] = confidence_progression
            metadata["final_iteration_stats"] = {
                "uncertain_fields": len([us for us in self.extraction_states[-1].uncertainty_scores if us.confidence < 0.7]),
                "high_confidence_fields": len([us for us in self.extraction_states[-1].uncertainty_scores if us.confidence >= 0.7])
            }
        
        return metadata
