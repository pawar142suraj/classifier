"""
Hybrid Graph RAG Extractor - Combining Microsoft's GraphRAG with Dynamic Uncertainty Refinement

This novel implementation combines:
1. Microsoft's global community detection and hierarchical summarization
2. DocuVerse's uncertainty-driven adaptive refinement
3. Multi-scale graph reasoning (local + global + dynamic)
4. Evidence tracking with community-based support
5. Schema-aware extraction with global context
"""

import json
import re
import logging
from typing import Dict, Any, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import networkx as nx
import numpy as np

from .base import BaseExtractor
from .dynamic_graph_rag import (
    UncertaintyScore, ExtractionEvidence, ExtractionResult, 
    DynamicUncertaintyEstimator, EvidenceExtractor
)
from ..utils.graph_utils import (
    KnowledgeGraph, Entity, Relation, EntityExtractor, RelationExtractor
)
from ..utils.graph_db import GraphDatabaseManager
from ..utils.page_extractor import page_extractor, get_page_info, extract_text_with_pages

logger = logging.getLogger(__name__)


@dataclass
class Community:
    """Represents a community of related entities."""
    id: str
    entities: List[Entity]
    summary: str
    confidence: float
    schema_relevance: Dict[str, float]  # Relevance to each schema field
    size: int
    density: float


@dataclass
class HierarchicalSummary:
    """Multi-level hierarchical summary structure."""
    level: int
    communities: List[Community]
    global_summary: str
    field_summaries: Dict[str, str]  # Per-field global summaries
    confidence: float


@dataclass
class MultiScaleContext:
    """Context from multiple scales of graph analysis."""
    local_context: str          # Traditional RAG context
    community_context: str      # Community-based context
    global_context: str         # Global hierarchical context
    uncertainty_context: str    # Uncertainty-driven expanded context
    evidence_sources: List[str] # Sources for evidence tracking


class CommunityDetector:
    """Detects communities using Leiden-like algorithm adapted for schema relevance."""
    
    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema
        self.field_weights = self._calculate_field_weights()
    
    def _calculate_field_weights(self) -> Dict[str, float]:
        """Calculate importance weights for each schema field."""
        weights = {}
        if "field" in self.schema:
            # Equal weights for now, can be enhanced with field importance
            num_fields = len(self.schema["field"])
            for field_name in self.schema["field"].keys():
                weights[field_name] = 1.0 / num_fields
        return weights
    
    def detect_communities(self, kg: KnowledgeGraph, min_community_size: int = 3) -> List[Community]:
        """Detect communities with schema-aware optimization."""
        # Convert to NetworkX graph
        G = self._kg_to_networkx(kg)
        
        if len(G.nodes()) < min_community_size:
            # Create single community if too few nodes
            return [self._create_single_community(kg)]
        
        # Use modularity-based community detection with schema weights
        communities_dict = self._leiden_like_detection(G, kg)
        
        # Convert to Community objects
        communities = []
        for comm_id, entity_ids in communities_dict.items():
            if len(entity_ids) >= min_community_size:
                community = self._build_community(comm_id, entity_ids, kg)
                communities.append(community)
        
        return communities
    
    def _kg_to_networkx(self, kg: KnowledgeGraph) -> nx.Graph:
        """Convert KnowledgeGraph to NetworkX graph."""
        G = nx.Graph()
        
        # Add nodes (entities)
        for entity in kg.entities.values():
            G.add_node(entity.id, 
                      text=entity.text,
                      type=entity.type,
                      confidence=entity.confidence,
                      properties=entity.properties)
        
        # Add edges (relations)
        for relation in kg.relations:
            # Weight edges by relation confidence and schema relevance
            weight = relation.confidence * self._calculate_schema_relevance(relation)
            G.add_edge(relation.source_id, relation.target_id, 
                      weight=weight,
                      relation_type=relation.relation_type)
        
        return G
    
    def _calculate_schema_relevance(self, relation: Relation) -> float:
        """Calculate how relevant a relation is to the schema fields."""
        relevance = 0.0
        
        if "field" in self.schema:
            for field_name, field_def in self.schema["field"].items():
                # Check if relation type matches field context
                if field_name.lower() in relation.relation_type.lower():
                    relevance += self.field_weights.get(field_name, 0.0)
                
                # Check if relation involves field-relevant entities
                if "enum" in field_def:
                    for enum_val in field_def["enum"]:
                        if enum_val.lower() in relation.relation_type.lower():
                            relevance += self.field_weights.get(field_name, 0.0)
        
        return max(relevance, 0.1)  # Minimum relevance
    
    def _leiden_like_detection(self, G: nx.Graph, kg: KnowledgeGraph) -> Dict[str, List[str]]:
        """Simplified Leiden-like community detection with schema awareness."""
        # Use NetworkX's built-in community detection as base
        try:
            import networkx.algorithms.community as nx_comm
            communities_gen = nx_comm.greedy_modularity_communities(G)
            communities_dict = {}
            
            for i, community_set in enumerate(communities_gen):
                communities_dict[f"community_{i}"] = list(community_set)
            
            return communities_dict
        
        except ImportError:
            # Fallback to simple clustering if networkx community not available
            return self._fallback_clustering(G, kg)
    
    def _fallback_clustering(self, G: nx.Graph, kg: KnowledgeGraph) -> Dict[str, List[str]]:
        """Fallback clustering using simple similarity-based grouping."""
        entity_vectors = []
        entity_ids = []
        
        for entity_id in G.nodes():
            entity = kg.entities[entity_id]
            # Simple vector representation based on entity properties
            vector = self._entity_to_vector(entity)
            entity_vectors.append(vector)
            entity_ids.append(entity_id)
        
        if len(entity_vectors) == 0:
            return {}
        
        # Simple similarity-based clustering
        communities_dict = defaultdict(list)
        used_entities = set()
        cluster_id = 0
        
        for i, entity_id in enumerate(entity_ids):
            if entity_id in used_entities:
                continue
                
            # Start new cluster
            current_cluster = [entity_id]
            used_entities.add(entity_id)
            
            # Find similar entities
            for j, other_entity_id in enumerate(entity_ids):
                if other_entity_id in used_entities or i == j:
                    continue
                
                # Calculate simple similarity
                similarity = self._calculate_entity_similarity(
                    entity_vectors[i], entity_vectors[j]
                )
                
                if similarity > 0.5:  # Similarity threshold
                    current_cluster.append(other_entity_id)
                    used_entities.add(other_entity_id)
            
            if len(current_cluster) >= 2:  # Minimum cluster size
                communities_dict[f"community_{cluster_id}"] = current_cluster
                cluster_id += 1
        
        return dict(communities_dict)
    
    def _calculate_entity_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate similarity between two entity vectors."""
        if len(vec1) != len(vec2):
            return 0.0
        
        # Simple cosine similarity approximation
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _entity_to_vector(self, entity: Entity) -> List[float]:
        """Convert entity to vector representation."""
        # Simple representation: [confidence, text_length, type_encoding, schema_relevance]
        vector = [
            entity.confidence,
            len(entity.text) / 100.0,  # Normalized text length
            hash(entity.type) % 100 / 100.0,  # Type encoding
            self._calculate_entity_schema_relevance(entity)
        ]
        return vector
    
    def _calculate_entity_schema_relevance(self, entity: Entity) -> float:
        """Calculate entity relevance to schema."""
        relevance = 0.0
        
        if "field" in self.schema:
            for field_name, field_def in self.schema["field"].items():
                # Check entity text against field name
                if field_name.lower() in entity.text.lower():
                    relevance += self.field_weights.get(field_name, 0.0)
                
                # Check against enum values
                if "enum" in field_def:
                    for enum_val in field_def["enum"]:
                        if enum_val.lower() in entity.text.lower():
                            relevance += self.field_weights.get(field_name, 0.0)
        
        return min(relevance, 1.0)
    
    def _build_community(self, comm_id: str, entity_ids: List[str], kg: KnowledgeGraph) -> Community:
        """Build Community object from entity IDs."""
        entities = [kg.entities[eid] for eid in entity_ids if eid in kg.entities]
        
        # Calculate community properties
        avg_confidence = sum(e.confidence for e in entities) / len(entities) if entities else 0.0
        
        # Calculate schema relevance for each field
        schema_relevance = {}
        if "field" in self.schema:
            for field_name in self.schema["field"].keys():
                field_relevance = sum(
                    self._calculate_entity_field_relevance(entity, field_name) 
                    for entity in entities
                ) / len(entities) if entities else 0.0
                schema_relevance[field_name] = field_relevance
        
        # Generate community summary
        summary = self._generate_community_summary(entities)
        
        # Calculate density (simplified)
        density = len(entities) / max(len(kg.entities), 1)
        
        return Community(
            id=comm_id,
            entities=entities,
            summary=summary,
            confidence=avg_confidence,
            schema_relevance=schema_relevance,
            size=len(entities),
            density=density
        )
    
    def _calculate_entity_field_relevance(self, entity: Entity, field_name: str) -> float:
        """Calculate how relevant an entity is to a specific field."""
        relevance = 0.0
        
        # Direct field name match
        if field_name.lower() in entity.text.lower():
            relevance += 0.5
        
        # Field definition context
        if "field" in self.schema and field_name in self.schema["field"]:
            field_def = self.schema["field"][field_name]
            
            # Enum matching
            if "enum" in field_def:
                for enum_val in field_def["enum"]:
                    if enum_val.lower() in entity.text.lower():
                        relevance += 0.3
            
            # Description matching (if available)
            if "description" in field_def:
                description_words = field_def["description"].lower().split()
                entity_words = entity.text.lower().split()
                overlap = len(set(description_words) & set(entity_words))
                relevance += overlap / max(len(description_words), 1) * 0.2
        
        return min(relevance, 1.0)
    
    def _generate_community_summary(self, entities: List[Entity]) -> str:
        """Generate a summary for the community."""
        if not entities:
            return "Empty community"
        
        # Extract key themes from entity texts
        all_text = " ".join(entity.text for entity in entities)
        
        # Simple extractive summary (can be enhanced with LLM)
        sentences = re.split(r'[.!?]+', all_text)
        important_sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        # Take top 3 sentences or all if fewer
        summary_sentences = important_sentences[:3]
        summary = ". ".join(summary_sentences)
        
        if not summary:
            summary = f"Community of {len(entities)} entities"
        
        return summary
    
    def _create_single_community(self, kg: KnowledgeGraph) -> Community:
        """Create a single community containing all entities."""
        all_entities = list(kg.entities.values())
        
        return Community(
            id="community_0",
            entities=all_entities,
            summary=self._generate_community_summary(all_entities),
            confidence=sum(e.confidence for e in all_entities) / len(all_entities) if all_entities else 0.0,
            schema_relevance={field: 0.5 for field in self.schema.get("field", {}).keys()},
            size=len(all_entities),
            density=1.0
        )


class HierarchicalSummarizer:
    """Creates hierarchical summaries from communities."""
    
    def __init__(self, llm, schema: Dict[str, Any]):
        self.llm = llm
        self.schema = schema
    
    def create_hierarchical_summary(self, communities: List[Community]) -> HierarchicalSummary:
        """Create multi-level hierarchical summary."""
        # Level 1: Individual community summaries (already created)
        
        # Level 2: Field-specific global summaries
        field_summaries = self._create_field_summaries(communities)
        
        # Level 3: Global summary
        global_summary = self._create_global_summary(communities, field_summaries)
        
        # Calculate overall confidence
        avg_confidence = sum(c.confidence for c in communities) / len(communities) if communities else 0.0
        
        return HierarchicalSummary(
            level=3,
            communities=communities,
            global_summary=global_summary,
            field_summaries=field_summaries,
            confidence=avg_confidence
        )
    
    def _create_field_summaries(self, communities: List[Community]) -> Dict[str, str]:
        """Create summaries focused on each schema field."""
        field_summaries = {}
        
        if "field" not in self.schema:
            return field_summaries
        
        for field_name in self.schema["field"].keys():
            # Find communities most relevant to this field
            relevant_communities = sorted(
                communities,
                key=lambda c: c.schema_relevance.get(field_name, 0.0),
                reverse=True
            )[:3]  # Top 3 most relevant communities
            
            if relevant_communities:
                field_context = "\n".join(c.summary for c in relevant_communities)
                field_summary = self._summarize_field_context(field_name, field_context)
                field_summaries[field_name] = field_summary
        
        return field_summaries
    
    def _summarize_field_context(self, field_name: str, context: str) -> str:
        """Summarize context specifically for a field."""
        prompt = f"""
        Based on the following context, provide a summary specifically focused on information relevant to extracting '{field_name}':

        Context:
        {context}

        Schema field definition:
        {json.dumps(self.schema.get("field", {}).get(field_name, {}), indent=2)}

        Provide a concise summary highlighting information relevant to '{field_name}':
        """
        
        try:
            response = self.llm.generate(prompt)
            return response.strip()
        except Exception as e:
            logger.warning(f"Failed to generate field summary for {field_name}: {e}")
            return f"Summary of context relevant to {field_name}: {context[:200]}..."
    
    def _create_global_summary(self, communities: List[Community], field_summaries: Dict[str, str]) -> str:
        """Create a global summary combining all community information."""
        community_summaries = "\n".join(f"- {c.summary}" for c in communities)
        field_info = "\n".join(f"- {field}: {summary}" for field, summary in field_summaries.items())
        
        prompt = f"""
        Create a comprehensive global summary based on the following information:

        Community Summaries:
        {community_summaries}

        Field-Specific Summaries:
        {field_info}

        Provide a global summary that captures the key themes and information:
        """
        
        try:
            response = self.llm.generate(prompt)
            return response.strip()
        except Exception as e:
            logger.warning(f"Failed to generate global summary: {e}")
            return f"Global summary based on {len(communities)} communities and {len(field_summaries)} field summaries."


class MultiScaleRetriever:
    """Retrieves context at multiple scales: local, community, global, and uncertainty-driven."""
    
    def __init__(self, kg: KnowledgeGraph, hierarchical_summary: HierarchicalSummary):
        self.kg = kg
        self.hierarchical_summary = hierarchical_summary
    
    def retrieve_multiscale_context(self, field_name: str, uncertainty_score: Optional[UncertaintyScore] = None) -> MultiScaleContext:
        """Retrieve context at multiple scales for a specific field."""
        
        # Local context: Traditional entity-based retrieval
        local_context = self._retrieve_local_context(field_name)
        
        # Community context: Most relevant communities
        community_context = self._retrieve_community_context(field_name)
        
        # Global context: Hierarchical summary
        global_context = self._retrieve_global_context(field_name)
        
        # Uncertainty-driven context: Expanded based on uncertainty patterns
        uncertainty_context = self._retrieve_uncertainty_context(field_name, uncertainty_score)
        
        # Evidence sources tracking
        evidence_sources = self._collect_evidence_sources(field_name)
        
        return MultiScaleContext(
            local_context=local_context,
            community_context=community_context,
            global_context=global_context,
            uncertainty_context=uncertainty_context,
            evidence_sources=evidence_sources
        )
    
    def _retrieve_local_context(self, field_name: str) -> str:
        """Traditional local entity retrieval."""
        relevant_entities = []
        
        for entity in self.kg.entities.values():
            if field_name.lower() in entity.text.lower():
                relevant_entities.append(entity)
        
        # Sort by confidence
        relevant_entities.sort(key=lambda e: e.confidence, reverse=True)
        
        # Take top 5 entities
        top_entities = relevant_entities[:5]
        
        if top_entities:
            context = "\n".join(f"- {e.text} (confidence: {e.confidence:.2f})" for e in top_entities)
            return f"Local entities relevant to {field_name}:\n{context}"
        
        return f"No local entities found for {field_name}"
    
    def _retrieve_community_context(self, field_name: str) -> str:
        """Community-based context retrieval."""
        # Find communities most relevant to this field
        relevant_communities = sorted(
            self.hierarchical_summary.communities,
            key=lambda c: c.schema_relevance.get(field_name, 0.0),
            reverse=True
        )[:2]  # Top 2 communities
        
        if relevant_communities:
            context = "\n".join(f"Community {c.id}: {c.summary}" for c in relevant_communities)
            return f"Community context for {field_name}:\n{context}"
        
        return f"No relevant communities found for {field_name}"
    
    def _retrieve_global_context(self, field_name: str) -> str:
        """Global hierarchical context."""
        global_summary = self.hierarchical_summary.global_summary
        field_summary = self.hierarchical_summary.field_summaries.get(field_name, "")
        
        context_parts = []
        if field_summary:
            context_parts.append(f"Field-specific summary: {field_summary}")
        if global_summary:
            context_parts.append(f"Global summary: {global_summary}")
        
        return "\n".join(context_parts) if context_parts else f"No global context for {field_name}"
    
    def _retrieve_uncertainty_context(self, field_name: str, uncertainty_score: Optional[UncertaintyScore]) -> str:
        """Uncertainty-driven expanded context."""
        if not uncertainty_score:
            return f"No uncertainty context for {field_name}"
        
        context_parts = [f"Uncertainty analysis for {field_name}:"]
        context_parts.append(f"- Confidence: {uncertainty_score.confidence:.2f}")
        
        if uncertainty_score.uncertainty_reasons:
            context_parts.append("- Uncertainty reasons:")
            for reason in uncertainty_score.uncertainty_reasons:
                context_parts.append(f"  * {reason}")
        
        if uncertainty_score.suggested_expansion:
            context_parts.append("- Suggested expansion strategies:")
            for strategy in uncertainty_score.suggested_expansion:
                context_parts.append(f"  * {strategy}")
        
        return "\n".join(context_parts)
    
    def _collect_evidence_sources(self, field_name: str) -> List[str]:
        """Collect sources for evidence tracking."""
        sources = []
        
        # Add entity sources
        for entity in self.kg.entities.values():
            if field_name.lower() in entity.text.lower():
                sources.append(f"Entity: {entity.text}")
        
        # Add community sources
        for community in self.hierarchical_summary.communities:
            if community.schema_relevance.get(field_name, 0.0) > 0.3:
                sources.append(f"Community {community.id}: {community.summary[:100]}...")
        
        return sources


class HybridGraphRAGExtractor(BaseExtractor):
    """
    Hybrid Graph RAG Extractor combining Microsoft's GraphRAG with Dynamic Uncertainty Refinement.
    
    This novel approach integrates:
    1. Global community detection and hierarchical summarization (Microsoft GraphRAG)
    2. Uncertainty-driven adaptive refinement (DocuVerse Dynamic)
    3. Multi-scale context retrieval (local + community + global + uncertainty)
    4. Evidence tracking with community support
    5. Schema-aware extraction with global reasoning
    """
    
    def __init__(self, llm, schema: Dict[str, Any], config: Dict[str, Any] = None):
        # Create a mock LLMConfig for the base class
        from ..core.config import LLMConfig, LLMProvider
        mock_llm_config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model_name="llama3.2:latest",
            base_url="http://localhost:11434",
            api_key="",
            max_tokens=1000,
            temperature=0.1
        )
        super().__init__(mock_llm_config)
        
        # Store actual parameters
        self.llm = llm
        self.schema = schema
        self.config = config or {}
        
        # Initialize components
        self.community_detector = CommunityDetector(schema)
        self.hierarchical_summarizer = HierarchicalSummarizer(llm, schema)
        self.uncertainty_estimator = DynamicUncertaintyEstimator(schema)
        
        # Initialize graph database with config
        graph_db_config = self.config.get('graph_db', {})
        self.graph_db = GraphDatabaseManager(graph_db_config)
        
        # Configuration
        self.max_iterations = self.config.get('max_iterations', 3)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.8)
        self.min_community_size = self.config.get('min_community_size', 3)
        self.use_hierarchical_reasoning = self.config.get('use_hierarchical_reasoning', True)
        
        # State tracking
        self.current_kg: Optional[KnowledgeGraph] = None
        self.current_communities: List[Community] = []
        self.hierarchical_summary: Optional[HierarchicalSummary] = None
        self.multiscale_retriever: Optional[MultiScaleRetriever] = None
        
        logger.info("Initialized HybridGraphRAGExtractor combining Microsoft GraphRAG with Dynamic Refinement")
    
    def extract(self, document_text: str) -> ExtractionResult:
        """
        Extract using hybrid approach with multi-scale reasoning.
        
        Args:
            document_text: Input document text (can be str or dict with 'text' key)
        
        Returns:
            ExtractionResult: Complete extraction with evidence and community support
        """
        # Handle both string and dict inputs for compatibility with base class
        if isinstance(document_text, dict):
            text = document_text.get('text', str(document_text))
        else:
            text = str(document_text)
        
        logger.info("Starting Hybrid Graph RAG extraction")
        
        # Initialize page extraction for document
        self.document_text = text
        self.document_pages = page_extractor.extract_pages(text)
        logger.info(f"Processing document with {len(self.document_pages)} pages")
        
        # Stage 1: Build initial knowledge graph
        self.current_kg = self._build_knowledge_graph(text)
        
        # Stage 2: Community detection (Microsoft GraphRAG approach)
        self.current_communities = self.community_detector.detect_communities(
            self.current_kg, self.min_community_size
        )
        logger.info(f"Detected {len(self.current_communities)} communities")
        
        # Stage 3: Hierarchical summarization
        if self.use_hierarchical_reasoning:
            self.hierarchical_summary = self.hierarchical_summarizer.create_hierarchical_summary(
                self.current_communities
            )
            logger.info("Created hierarchical summary")
        
        # Stage 4: Initialize multi-scale retriever
        self.multiscale_retriever = MultiScaleRetriever(self.current_kg, self.hierarchical_summary)
        
        # Stage 5: Initial extraction with global context
        current_extraction = self._initial_extraction_with_global_context(text)
        
        # Stage 6: Iterative refinement with uncertainty-driven expansion
        iteration = 0
        evidence_extractor = EvidenceExtractor(text, self.schema)
        
        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"Starting hybrid refinement iteration {iteration}")
            
            # Assess uncertainty
            uncertainty_scores = self._assess_extraction_uncertainty(current_extraction)
            
            # Check convergence
            if self._has_converged(uncertainty_scores):
                logger.info(f"Extraction converged after {iteration-1} iterations")
                break
            
            # Refine using multi-scale context
            refined_extraction = self._refine_with_multiscale_context(
                text, current_extraction, uncertainty_scores
            )
            
            current_extraction = refined_extraction
        
        # Stage 7: Generate evidence with community support
        evidence_list = []
        for field_name, value in current_extraction.items():
            # Get multi-scale context for evidence
            context = self.multiscale_retriever.retrieve_multiscale_context(field_name)
            
            # Find supporting entities
            supporting_entities = self._find_supporting_entities(field_name)
            
            # Extract evidence
            evidence = evidence_extractor.extract_evidence_for_field(
                field_name, value, supporting_entities
            )
            
            # Enhance evidence with community information
            evidence = self._enhance_evidence_with_communities(evidence, context)
            evidence_list.append(evidence)
        
        # Stage 8: Store results in graph database
        self._store_hybrid_results(text, current_extraction, self.current_communities)
        
        # Calculate final confidence
        overall_confidence = self._calculate_hybrid_confidence(evidence_list)
        
        logger.info(f"Hybrid Graph RAG extraction completed. Confidence: {overall_confidence:.2f}")
        
        return ExtractionResult(
            extracted_data=current_extraction,
            evidence={evidence.field_name: evidence for evidence in evidence_list},
            metadata={
                "overall_confidence": overall_confidence,
                "graph_size": len(self.current_kg.entities) if self.current_kg else 0,
                "iterations_used": iteration,
                "communities_detected": len(self.current_communities),
                "method": "hybrid_graph_rag"
            }
        )
    
    def _build_knowledge_graph(self, document_text: str) -> KnowledgeGraph:
        """Build initial knowledge graph from document."""
        try:
            # Use existing entity and relation extractors
            entity_extractor = EntityExtractor(self.schema)
            relation_extractor = RelationExtractor()
            
            # Extract entities with schema awareness
            entities = entity_extractor.extract_entities(document_text)
            
            # Extract relations
            relations = relation_extractor.extract_relations(document_text, entities)
            
            # Build knowledge graph
            kg = KnowledgeGraph()
            
            # Add entities - handle both list and dict formats
            if isinstance(entities, list):
                for entity in entities:
                    kg.add_entity(entity)
            elif isinstance(entities, dict):
                for entity in entities.values():
                    kg.add_entity(entity)
            
            # Add relations - handle both list and dict formats  
            if isinstance(relations, list):
                for relation in relations:
                    kg.add_relation(relation)
            elif isinstance(relations, dict):
                for relation in relations.values():
                    kg.add_relation(relation)
            
            return kg
            
        except Exception as e:
            print(f"DEBUG: Error in _build_knowledge_graph: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _initial_extraction_with_global_context(self, document_text: str) -> Dict[str, Any]:
        """Initial extraction using global community context."""
        # Build global context from hierarchical summary
        global_context = ""
        if self.hierarchical_summary:
            global_context = f"""
            Global Document Summary: {self.hierarchical_summary.global_summary}
            
            Field-Specific Context:
            {chr(10).join(f"- {field}: {summary}" for field, summary in self.hierarchical_summary.field_summaries.items())}
            
            Community Information:
            {chr(10).join(f"- Community {c.id} ({c.size} entities): {c.summary}" for c in self.current_communities)}
            """
        
        # Create extraction prompt with global context
        prompt = self._build_hybrid_extraction_prompt(document_text, global_context)
        
        try:
            response = self._call_llm(prompt, "You are an expert document analysis assistant. Extract information accurately and provide structured responses.")
            extracted_data = self._parse_llm_response(response)
            return self._align_with_schema(extracted_data)
        except Exception as e:
            logger.error(f"Initial extraction failed: {e}")
            return self._build_empty_result()
    
    def _assess_extraction_uncertainty(self, extracted_data: Dict[str, Any]) -> List[UncertaintyScore]:
        """Assess uncertainty for each extracted field."""
        uncertainty_scores = []
        
        for field_name, value in extracted_data.items():
            # Get supporting entities from communities
            supporting_entities = self._find_supporting_entities(field_name)
            
            # Estimate uncertainty
            uncertainty_score = self.uncertainty_estimator.estimate_field_uncertainty(
                field_name, value, str(value), supporting_entities
            )
            uncertainty_scores.append(uncertainty_score)
        
        return uncertainty_scores
    
    def _has_converged(self, uncertainty_scores: List[UncertaintyScore]) -> bool:
        """Check if extraction has converged."""
        if not uncertainty_scores:
            return True
        
        avg_confidence = sum(us.confidence for us in uncertainty_scores) / len(uncertainty_scores)
        return avg_confidence >= self.confidence_threshold
    
    def _refine_with_multiscale_context(self, document_text: str, current_extraction: Dict[str, Any], 
                                      uncertainty_scores: List[UncertaintyScore]) -> Dict[str, Any]:
        """Refine extraction using multi-scale context."""
        # Get uncertain fields
        uncertain_fields = [us for us in uncertainty_scores if us.confidence < self.confidence_threshold]
        
        if not uncertain_fields:
            return current_extraction
        
        # Build refinement context using multi-scale retrieval
        refinement_contexts = []
        for uncertainty_score in uncertain_fields:
            context = self.multiscale_retriever.retrieve_multiscale_context(
                uncertainty_score.field_name, uncertainty_score
            )
            refinement_contexts.append(self._format_multiscale_context(context))
        
        # Create refinement prompt
        refinement_prompt = self._build_refinement_prompt(
            document_text, current_extraction, uncertain_fields, refinement_contexts
        )
        
        try:
            response = self._call_llm(refinement_prompt, "You are an expert document analysis assistant. Refine the extraction based on the provided context and uncertainty analysis.")
            refined_data = self._parse_llm_response(response)
            
            # Merge with current extraction (only update uncertain fields)
            updated_extraction = current_extraction.copy()
            for uncertainty_score in uncertain_fields:
                field_name = uncertainty_score.field_name
                if field_name in refined_data:
                    updated_extraction[field_name] = refined_data[field_name]
            
            return self._align_with_schema(updated_extraction)
        
        except Exception as e:
            logger.error(f"Refinement failed: {e}")
            return current_extraction
    
    def _find_supporting_entities(self, field_name: str) -> List[Entity]:
        """Find entities that support a specific field."""
        supporting_entities = []
        
        if not self.current_kg:
            return supporting_entities
        
        for entity in self.current_kg.entities.values():
            # Check direct field relevance
            if field_name.lower() in entity.text.lower():
                supporting_entities.append(entity)
                continue
            
            # Check schema-based relevance
            if self._is_entity_relevant_to_field(entity, field_name):
                supporting_entities.append(entity)
        
        # Sort by confidence
        supporting_entities.sort(key=lambda e: e.confidence, reverse=True)
        return supporting_entities[:10]  # Top 10 supporting entities
    
    def _is_entity_relevant_to_field(self, entity: Entity, field_name: str) -> bool:
        """Check if entity is relevant to a field based on schema."""
        if "field" not in self.schema or field_name not in self.schema["field"]:
            return False
        
        field_def = self.schema["field"][field_name]
        
        # Check enum values
        if "enum" in field_def:
            for enum_val in field_def["enum"]:
                if enum_val.lower() in entity.text.lower():
                    return True
        
        # Check field description
        if "description" in field_def:
            description_words = set(field_def["description"].lower().split())
            entity_words = set(entity.text.lower().split())
            if len(description_words & entity_words) > 0:
                return True
        
        return False
    
    def _enhance_evidence_with_communities(self, evidence: ExtractionEvidence, 
                                         context: MultiScaleContext) -> ExtractionEvidence:
        """Enhance evidence with community-based information and page details."""
        # Add community context to reasoning
        enhanced_reasoning = evidence.reasoning
        
        if context.community_context:
            enhanced_reasoning += f" | Community support: {context.community_context[:100]}..."
        
        if context.global_context:
            enhanced_reasoning += f" | Global context: {context.global_context[:100]}..."
        
        # Add page information if not already present
        if hasattr(self, 'document_text') and hasattr(self, 'document_pages'):
            if evidence.page_number is None and evidence.evidence_text:
                # Find evidence position in document
                text_pos = self.document_text.find(evidence.evidence_text)
                if text_pos != -1:
                    # Get page information
                    page_info = get_page_info(self.document_text, text_pos, text_pos + len(evidence.evidence_text))
                    evidence.page_number = page_info.get('page_number')
                    evidence.total_pages = len(self.document_pages)
                    
                    # Get enhanced page context
                    text_with_pages = extract_text_with_pages(
                        self.document_text, text_pos, text_pos + len(evidence.evidence_text), context_chars=150
                    )
                    evidence.page_context = text_with_pages.get('context', '')
        
        # Update evidence with enhanced information
        evidence.reasoning = enhanced_reasoning
        evidence.related_clauses.extend(context.evidence_sources)
        
        return evidence
    
    def _store_hybrid_results(self, document_text: str, extraction_result: Dict[str, Any], 
                            communities: List[Community]) -> None:
        """Store hybrid results in graph database."""
        try:
            # Store knowledge graph
            if self.current_kg:
                self.graph_db.store_knowledge_graph(self.current_kg, f"hybrid_extraction_{id(document_text)}")
            
            # Store community information
            community_data = {
                "communities": [
                    {
                        "id": c.id,
                        "size": c.size,
                        "summary": c.summary,
                        "confidence": c.confidence,
                        "schema_relevance": c.schema_relevance
                    }
                    for c in communities
                ],
                "extraction_result": extraction_result
            }
            
            # Store in database (implementation depends on graph_db)
            logger.info("Stored hybrid extraction results in graph database")
            
        except Exception as e:
            logger.warning(f"Failed to store hybrid results: {e}")
    
    def _calculate_hybrid_confidence(self, evidence_list: List[ExtractionEvidence]) -> float:
        """Calculate overall confidence incorporating community support."""
        if not evidence_list:
            return 0.0
        
        # Base confidence from evidence
        base_confidence = sum(e.confidence for e in evidence_list) / len(evidence_list)
        
        # Community support bonus
        community_bonus = 0.0
        if self.current_communities:
            avg_community_confidence = sum(c.confidence for c in self.current_communities) / len(self.current_communities)
            community_bonus = avg_community_confidence * 0.1  # 10% bonus from communities
        
        # Global reasoning bonus
        global_bonus = 0.0
        if self.hierarchical_summary and self.hierarchical_summary.confidence > 0.7:
            global_bonus = 0.05  # 5% bonus for strong global reasoning
        
        final_confidence = base_confidence + community_bonus + global_bonus
        return min(final_confidence, 0.95)  # Cap at 95%
    
    # Helper methods
    def _build_hybrid_extraction_prompt(self, document_text: str, global_context: str) -> str:
        """Build extraction prompt with global context."""
        return f"""
        Extract information from the following document using the provided schema and global context.

        Global Context:
        {global_context}

        Document:
        {document_text}

        Schema:
        {json.dumps(self.schema, indent=2)}

        Extract the required information in JSON format, using the global context to inform your decisions:
        """
    
    def _format_multiscale_context(self, context: MultiScaleContext) -> str:
        """Format multi-scale context for prompts."""
        formatted_parts = []
        
        if context.local_context:
            formatted_parts.append(f"Local Context: {context.local_context}")
        
        if context.community_context:
            formatted_parts.append(f"Community Context: {context.community_context}")
        
        if context.global_context:
            formatted_parts.append(f"Global Context: {context.global_context}")
        
        if context.uncertainty_context:
            formatted_parts.append(f"Uncertainty Analysis: {context.uncertainty_context}")
        
        return "\n\n".join(formatted_parts)
    
    def _build_refinement_prompt(self, document_text: str, current_extraction: Dict[str, Any],
                               uncertain_fields: List[UncertaintyScore], 
                               refinement_contexts: List[str]) -> str:
        """Build refinement prompt with multi-scale context."""
        uncertain_field_names = [uf.field_name for uf in uncertain_fields]
        
        return f"""
        Refine the extraction for uncertain fields using multi-scale context analysis.

        Current Extraction:
        {json.dumps(current_extraction, indent=2)}

        Uncertain Fields: {uncertain_field_names}

        Multi-Scale Context Analysis:
        {chr(10).join(refinement_contexts)}

        Document:
        {document_text}

        Schema:
        {json.dumps(self.schema, indent=2)}

        Focus on improving the uncertain fields using the provided multi-scale context. Return complete JSON:
        """
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to extract JSON."""
        try:
            # Try to find JSON in response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {}
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as JSON")
            return {}
    
    def _align_with_schema(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Align extracted data with schema requirements."""
        aligned_data = {}
        
        if "field" in self.schema:
            for field_name, field_def in self.schema["field"].items():
                if field_name in extracted_data:
                    value = extracted_data[field_name]
                    
                    # Enum validation
                    if "enum" in field_def and value not in field_def["enum"]:
                        # Try to find closest enum match
                        closest_match = self._find_closest_enum_match(value, field_def["enum"])
                        if closest_match:
                            aligned_data[field_name] = closest_match
                        else:
                            aligned_data[field_name] = field_def["enum"][0]  # Default to first enum
                    else:
                        aligned_data[field_name] = value
                else:
                    # Use default value if available
                    if "default" in field_def:
                        aligned_data[field_name] = field_def["default"]
                    elif "enum" in field_def:
                        aligned_data[field_name] = field_def["enum"][0]
                    else:
                        aligned_data[field_name] = None
        
        return aligned_data
    
    def _find_closest_enum_match(self, value: str, enum_values: List[str]) -> Optional[str]:
        """Find closest enum match for a value."""
        if not isinstance(value, str):
            return None
        
        value_lower = value.lower()
        
        # Exact match
        for enum_val in enum_values:
            if value_lower == enum_val.lower():
                return enum_val
        
        # Partial match
        for enum_val in enum_values:
            if value_lower in enum_val.lower() or enum_val.lower() in value_lower:
                return enum_val
        
        return None
    
    def _build_empty_result(self) -> Dict[str, Any]:
        """Build empty result structure."""
        empty_result = {}
        
        if "field" in self.schema:
            for field_name, field_def in self.schema["field"].items():
                if "default" in field_def:
                    empty_result[field_name] = field_def["default"]
                elif "enum" in field_def:
                    empty_result[field_name] = field_def["enum"][0]
                else:
                    empty_result[field_name] = None
        
        return empty_result

    def get_community_analysis(self) -> Dict[str, Any]:
        """Get analysis of detected communities."""
        if not self.current_communities:
            return {"communities": [], "analysis": "No communities detected"}
        
        analysis = {
            "total_communities": len(self.current_communities),
            "communities": [],
            "schema_coverage": {},
            "confidence_distribution": []
        }
        
        # Community details
        for community in self.current_communities:
            community_info = {
                "id": community.id,
                "size": community.size,
                "confidence": community.confidence,
                "density": community.density,
                "summary": community.summary,
                "schema_relevance": community.schema_relevance
            }
            analysis["communities"].append(community_info)
            analysis["confidence_distribution"].append(community.confidence)
        
        # Schema coverage analysis
        if "field" in self.schema:
            for field_name in self.schema["field"].keys():
                field_coverage = [c.schema_relevance.get(field_name, 0.0) for c in self.current_communities]
                analysis["schema_coverage"][field_name] = {
                    "max_relevance": max(field_coverage) if field_coverage else 0.0,
                    "avg_relevance": sum(field_coverage) / len(field_coverage) if field_coverage else 0.0,
                    "supporting_communities": sum(1 for r in field_coverage if r > 0.3)
                }
        
        return analysis

    def get_hierarchical_summary_info(self) -> Dict[str, Any]:
        """Get information about the hierarchical summary."""
        if not self.hierarchical_summary:
            return {"status": "No hierarchical summary available"}
        
        return {
            "level": self.hierarchical_summary.level,
            "confidence": self.hierarchical_summary.confidence,
            "global_summary": self.hierarchical_summary.global_summary,
            "field_summaries": self.hierarchical_summary.field_summaries,
            "communities_count": len(self.hierarchical_summary.communities)
        }
