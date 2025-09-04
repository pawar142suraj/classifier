"""
Graph RAG extractor using knowledge graphs for extraction and classification.
"""

import json
import re
from typing import Dict, Any, Optional, Union, List
import logging

from .base import BaseExtractor
from ..core.config import LLMConfig, GraphRAGConfig
from ..utils.graph_utils import (
    build_knowledge_graph, 
    GraphRetriever, 
    KnowledgeGraph,
    Entity,
    Relation
)

logger = logging.getLogger(__name__)


class GraphRAGExtractor(BaseExtractor):
    """
    Graph RAG extraction method using knowledge graphs for both classification and extraction.
    
    This extractor:
    1. Builds a knowledge graph from the document text and schema
    2. Retrieves relevant subgraphs based on the extraction task
    3. Uses graph context to enhance LLM-based extraction/classification
    4. Supports both classification (when schema has enums) and extraction tasks
    """
    
    def __init__(self, llm_config: LLMConfig, graph_config: GraphRAGConfig, schema: Optional[Dict[str, Any]] = None):
        """Initialize Graph RAG extractor."""
        super().__init__(llm_config)
        self.graph_config = graph_config
        self.schema = schema
        self.knowledge_graph: Optional[KnowledgeGraph] = None
        self.graph_retriever: Optional[GraphRetriever] = None
        
        logger.info("Initialized GraphRAGExtractor")
    
    def set_schema(self, schema: Dict[str, Any]) -> None:
        """Set the extraction schema."""
        self.schema = schema
        logger.info(f"Schema set with fields: {list(schema.get('field', {}).keys())}")
    
    def extract(self, document: Dict[str, Any], schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract using Graph RAG approach."""
        # Use provided schema or fall back to instance schema
        extraction_schema = schema or self.schema
        if not extraction_schema:
            raise ValueError("No schema provided for extraction")
        
        document_text = self._prepare_document_text(document)
        
        # Build knowledge graph from document and schema
        self.knowledge_graph = build_knowledge_graph(document_text, extraction_schema)
        self.graph_retriever = GraphRetriever(self.knowledge_graph)
        
        # Determine if this is classification or extraction task
        task_type = self._determine_task_type(extraction_schema)
        
        if task_type == "classification":
            return self._classify_with_graph(document_text, extraction_schema)
        else:
            return self._extract_with_graph(document_text, extraction_schema)
    
    def _determine_task_type(self, schema: Dict[str, Any]) -> str:
        """Determine if this is a classification or extraction task based on schema."""
        if "field" in schema:
            # Check if any fields have enum values (classification)
            for field_def in schema["field"].values():
                if "enum" in field_def:
                    return "classification"
        return "extraction"
    
    def _classify_with_graph(self, document_text: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Perform classification using graph context."""
        # Retrieve relevant subgraph for classification
        relevant_subgraph = self.graph_retriever.retrieve_for_classification(schema)
        
        # Build graph context
        graph_context = self._build_graph_context(relevant_subgraph, schema)
        
        # Create classification prompt
        prompt = self._build_classification_prompt(document_text, schema, graph_context)
        
        system_prompt = """You are an expert document classifier that uses knowledge graph information to make accurate classifications.
        
Use the provided graph context to understand the relationships between entities in the document and make informed classification decisions based on the schema definitions."""
        
        response = self._call_llm(prompt, system_prompt)
        
        logger.info(f"Raw LLM response: {response}")
        
        try:
            result = json.loads(response.strip())
            self.last_confidence = self._calculate_confidence(result, schema, relevant_subgraph)
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Graph RAG classification response: {e}")
            logger.error(f"Raw response was: {repr(response)}")
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    self.last_confidence = self._calculate_confidence(result, schema, relevant_subgraph)
                    return result
                except json.JSONDecodeError:
                    pass
            return self._build_empty_result(schema)
    
    def _extract_with_graph(self, document_text: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Perform extraction using graph context."""
        # Retrieve relevant subgraph for extraction
        relevant_subgraph = self.graph_retriever.retrieve_for_schema(schema)
        
        # Build graph context
        graph_context = self._build_graph_context(relevant_subgraph, schema)
        
        # Create extraction prompt
        prompt = self._build_extraction_prompt(document_text, schema, graph_context)
        
        system_prompt = """You are an expert information extractor that uses knowledge graph analysis to identify and extract relevant information from documents.
        
Use the provided graph context to understand entity relationships and extract accurate information according to the schema definitions."""
        
        response = self._call_llm(prompt, system_prompt)
        
        logger.info(f"Raw LLM response: {response}")
        
        try:
            result = json.loads(response.strip())
            self.last_confidence = self._calculate_confidence(result, schema, relevant_subgraph)
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Graph RAG extraction response: {e}")
            logger.error(f"Raw response was: {repr(response)}")
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    self.last_confidence = self._calculate_confidence(result, schema, relevant_subgraph)
                    return result
                except json.JSONDecodeError:
                    pass
            return self._build_empty_result(schema)
    
    def _build_graph_context(self, subgraph: KnowledgeGraph, schema: Dict[str, Any]) -> str:
        """Build a textual representation of the graph context."""
        context_parts = []
        
        # Add entities grouped by type
        context_parts.append("## Knowledge Graph Entities")
        entity_types = set()
        for entity in subgraph.entities.values():
            entity_types.add(entity.type)
        
        for entity_type in sorted(entity_types):
            entities = subgraph.get_entities_by_type(entity_type)
            if entities:
                context_parts.append(f"\n### {entity_type.replace('_', ' ').title()} Entities:")
                for entity in entities[:5]:  # Limit to top 5 per type
                    context_parts.append(f"- {entity.text} (confidence: {entity.confidence:.2f})")
        
        # Add relations
        if subgraph.relations:
            context_parts.append("\n## Entity Relations")
            relation_types = set(r.relation_type for r in subgraph.relations)
            for rel_type in sorted(relation_types):
                rels = [r for r in subgraph.relations if r.relation_type == rel_type]
                if rels:
                    context_parts.append(f"\n### {rel_type.replace('_', ' ').title()} Relations:")
                    for rel in rels[:3]:  # Limit to top 3 per type
                        source_text = subgraph.entities.get(rel.source_id, Entity("", "unknown", "", {})).text
                        target_text = subgraph.entities.get(rel.target_id, Entity("", "unknown", "", {})).text
                        context_parts.append(f"- {source_text} → {target_text}")
        
        # Add schema field mappings
        context_parts.append("\n## Schema Field Mappings")
        if "field" in schema:
            for field_name, field_def in schema["field"].items():
                field_entities = subgraph.get_entities_by_type(field_name)
                if field_entities:
                    context_parts.append(f"\n### {field_name}:")
                    context_parts.append(f"Description: {field_def.get('description', 'N/A')}")
                    if "enum" in field_def:
                        context_parts.append(f"Possible values: {field_def['enum']}")
                    context_parts.append(f"Found entities: {[e.text for e in field_entities[:3]]}")
        
        return "\n".join(context_parts)
    
    def _build_classification_prompt(self, document_text: str, schema: Dict[str, Any], graph_context: str) -> str:
        """Build classification prompt with graph context."""
        prompt_parts = [
            "# Document Classification Task",
            "",
            "## Task Description",
            "Classify the following document based on the provided schema definitions and knowledge graph analysis.",
            "",
            "## Knowledge Graph Context",
            graph_context,
            "",
            "## Classification Schema",
            self._format_schema_for_prompt(schema),
            "",
            "## Document Text",
            f"```\n{document_text}\n```",
            "",
            "## Instructions",
            "1. Analyze the knowledge graph entities and relations",
            "2. Match document content to schema field definitions",
            "3. For enum fields, select the most appropriate value based on the descriptions",
            "4. Return the result as JSON matching the schema structure",
            "",
            "## Expected Output Format",
            self._build_output_format_example(schema),
            "",
            "Provide only the JSON response:"
        ]
        
        return "\n".join(prompt_parts)
    
    def _build_extraction_prompt(self, document_text: str, schema: Dict[str, Any], graph_context: str) -> str:
        """Build extraction prompt with graph context."""
        prompt_parts = [
            "# Information Extraction Task",
            "",
            "## Task Description", 
            "Extract structured information from the following document using the knowledge graph analysis and schema definitions.",
            "",
            "## Knowledge Graph Context",
            graph_context,
            "",
            "## Extraction Schema",
            self._format_schema_for_prompt(schema),
            "",
            "## Document Text",
            f"```\n{document_text}\n```",
            "",
            "## Instructions",
            "1. Use the knowledge graph entities to identify relevant information",
            "2. Follow the schema field definitions and types",
            "3. Extract the most accurate information based on entity relationships",
            "4. Return the result as JSON matching the schema structure",
            "",
            "## Expected Output Format",
            self._build_output_format_example(schema),
            "",
            "Provide only the JSON response:"
        ]
        
        return "\n".join(prompt_parts)
    
    def _format_schema_for_prompt(self, schema: Dict[str, Any]) -> str:
        """Format schema for inclusion in prompts."""
        if "field" not in schema:
            return json.dumps(schema, indent=2)
        
        formatted_parts = []
        for field_name, field_def in schema["field"].items():
            parts = [f"**{field_name}**:"]
            parts.append(f"  - Type: {field_def.get('type', 'unknown')}")
            parts.append(f"  - Description: {field_def.get('description', 'N/A')}")
            
            if "enum" in field_def:
                parts.append(f"  - Possible values: {field_def['enum']}")
                if "enumDescriptions" in field_def:
                    parts.append("  - Value descriptions:")
                    for enum_val, desc in field_def["enumDescriptions"].items():
                        parts.append(f"    - {enum_val}: {desc}")
            
            formatted_parts.append("\n".join(parts))
        
        return "\n\n".join(formatted_parts)
    
    def _build_output_format_example(self, schema: Dict[str, Any]) -> str:
        """Build an example output format based on schema."""
        if "field" not in schema:
            return "{}"
        
        example = {}
        for field_name, field_def in schema["field"].items():
            if "enum" in field_def:
                example[field_name] = f"<one of: {field_def['enum']}>"
            elif field_def.get("type") == "string":
                example[field_name] = "<extracted_string_value>"
            else:
                example[field_name] = f"<{field_def.get('type', 'unknown')}_value>"
        
        return f"```json\n{json.dumps(example, indent=2)}\n```"
    
    def _build_empty_result(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Build an empty result structure based on schema."""
        if "field" not in schema:
            return {}
        
        result = {}
        for field_name in schema["field"].keys():
            result[field_name] = None
        
        return result
    
    def _calculate_confidence(self, result: Dict[str, Any], schema: Dict[str, Any], subgraph: KnowledgeGraph) -> float:
        """Calculate confidence score based on graph evidence and result completeness."""
        if not result or "field" not in schema:
            return 0.0
        
        total_fields = len(schema["field"])
        filled_fields = sum(1 for v in result.values() if v is not None)
        completeness_score = filled_fields / total_fields if total_fields > 0 else 0.0
        
        # Factor in graph evidence
        total_entities = len(subgraph.entities)
        relevant_entities = 0
        
        for field_name in schema["field"].keys():
            field_entities = subgraph.get_entities_by_type(field_name)
            if field_entities:
                relevant_entities += len(field_entities)
        
        evidence_score = min(relevant_entities / max(total_entities, 1), 1.0) if total_entities > 0 else 0.0
        
        # Combine scores
        confidence = (completeness_score * 0.6) + (evidence_score * 0.4)
        return min(confidence, 0.95)  # Cap at 95%
    
    def get_extraction_metadata(self) -> Dict[str, Any]:
        """Get metadata about the extraction process."""
        metadata = {
            "extractor_type": "graph_rag",
            "confidence": self.last_confidence,
            "token_usage": self.last_token_usage
        }
        
        if self.knowledge_graph:
            metadata.update({
                "graph_stats": {
                    "total_entities": len(self.knowledge_graph.entities),
                    "total_relations": len(self.knowledge_graph.relations),
                    "entity_types": len(set(e.type for e in self.knowledge_graph.entities.values()))
                }
            })
        
        return metadata
