"""
Graph utilities for knowledge graph construction and querying.
"""

import json
import re
from typing import Dict, List, Any, Tuple, Set, Optional
from dataclasses import dataclass
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Represents an entity in the knowledge graph."""
    id: str
    text: str
    type: str
    properties: Dict[str, Any]
    confidence: float = 1.0


@dataclass
class Relation:
    """Represents a relation between entities."""
    source_id: str
    target_id: str
    relation_type: str
    properties: Dict[str, Any]
    confidence: float = 1.0


class KnowledgeGraph:
    """In-memory knowledge graph for document processing."""
    
    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []
        self.entity_index: Dict[str, Set[str]] = defaultdict(set)  # type -> entity_ids
        
    def add_entity(self, entity: Entity) -> None:
        """Add an entity to the graph."""
        self.entities[entity.id] = entity
        self.entity_index[entity.type].add(entity.id)
        
    def add_relation(self, relation: Relation) -> None:
        """Add a relation to the graph."""
        self.relations.append(relation)
        
    def get_entities_by_type(self, entity_type: str) -> List[Entity]:
        """Get all entities of a specific type."""
        return [self.entities[eid] for eid in self.entity_index[entity_type]]
    
    def get_neighbors(self, entity_id: str) -> List[Tuple[Entity, Relation]]:
        """Get neighboring entities and their relations."""
        neighbors = []
        for relation in self.relations:
            if relation.source_id == entity_id:
                if relation.target_id in self.entities:
                    neighbors.append((self.entities[relation.target_id], relation))
            elif relation.target_id == entity_id:
                if relation.source_id in self.entities:
                    neighbors.append((self.entities[relation.source_id], relation))
        return neighbors
    
    def get_subgraph(self, entity_ids: List[str], max_depth: int = 2) -> 'KnowledgeGraph':
        """Extract a subgraph around specified entities."""
        subgraph = KnowledgeGraph()
        visited = set()
        queue = [(eid, 0) for eid in entity_ids if eid in self.entities]
        
        while queue:
            entity_id, depth = queue.pop(0)
            if entity_id in visited or depth > max_depth:
                continue
                
            visited.add(entity_id)
            entity = self.entities[entity_id]
            subgraph.add_entity(entity)
            
            # Add neighbors if within depth limit
            if depth < max_depth:
                for neighbor, relation in self.get_neighbors(entity_id):
                    if neighbor.id not in visited:
                        queue.append((neighbor.id, depth + 1))
                    # Add relation if both entities are in subgraph
                    if relation.source_id in visited and relation.target_id in visited:
                        subgraph.add_relation(relation)
        
        return subgraph


class EntityExtractor:
    """Extract entities from text based on schema definitions."""
    
    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema
        self.entity_patterns = self._build_entity_patterns()
        
    def _build_entity_patterns(self) -> Dict[str, List[str]]:
        """Build regex patterns for entity extraction based on schema."""
        patterns = {}
        
        if "field" in self.schema:
            for field_name, field_def in self.schema["field"].items():
                field_patterns = []
                
                # Pattern for field name mentions
                field_name_pattern = field_name.replace("_", r"\s*")
                field_patterns.append(rf'\b{field_name_pattern}\b')
                
                # Patterns for enum values if present
                if "enum" in field_def and "enumDescriptions" in field_def:
                    for enum_val, description in field_def["enumDescriptions"].items():
                        # Add the enum value itself
                        enum_val_pattern = enum_val.replace("_", r"\s*")
                        field_patterns.append(rf'\b{enum_val_pattern}\b')
                        
                        # Extract key terms from description
                        desc_terms = self._extract_key_terms(description)
                        field_patterns.extend(desc_terms)
                
                # Add description-based patterns
                if "description" in field_def:
                    desc_terms = self._extract_key_terms(field_def["description"])
                    field_patterns.extend(desc_terms)
                
                patterns[field_name] = field_patterns
                
        return patterns
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from description text."""
        # Simple keyword extraction
        keywords = []
        
        # Extract quoted terms
        quoted_terms = re.findall(r'"([^"]*)"', text)
        keywords.extend([rf'\b{term}\b' for term in quoted_terms])
        
        # Extract important phrases (basic heuristics)
        important_phrases = re.findall(r'\b(?:every|once|per|annually|monthly|yearly|standard|non-standard)\b', text.lower())
        keywords.extend([rf'\b{phrase}\b' for phrase in important_phrases])
        
        return keywords
    
    def extract_entities(self, text: str) -> List[Entity]:
        """Extract entities from text."""
        entities = []
        entity_id_counter = 0
        
        for field_name, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity_id = f"entity_{entity_id_counter}"
                    entity_id_counter += 1
                    
                    entity = Entity(
                        id=entity_id,
                        text=match.group(),
                        type=field_name,
                        properties={
                            "start": match.start(),
                            "end": match.end(),
                            "pattern": pattern
                        },
                        confidence=0.8
                    )
                    entities.append(entity)
        
        # Extract general entities (names, numbers, dates)
        self._extract_general_entities(text, entities, entity_id_counter)
        
        return entities
    
    def _extract_general_entities(self, text: str, entities: List[Entity], start_id: int) -> None:
        """Extract general entities like names, numbers, dates."""
        patterns = {
            "NUMBER": r'\b\d+(?:\.\d+)?\b',
            "MONEY": r'\$\d+(?:,\d{3})*(?:\.\d{2})?',
            "DATE": r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}\b',
            "PERCENTAGE": r'\d+(?:\.\d+)?%',
            "CAPITALIZED_WORD": r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        }
        
        entity_id_counter = start_id
        for entity_type, pattern in patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                entity_id = f"entity_{entity_id_counter}"
                entity_id_counter += 1
                
                entity = Entity(
                    id=entity_id,
                    text=match.group(),
                    type=entity_type,
                    properties={
                        "start": match.start(),
                        "end": match.end()
                    },
                    confidence=0.6
                )
                entities.append(entity)


class RelationExtractor:
    """Extract relations between entities."""
    
    def __init__(self):
        self.relation_patterns = {
            "MENTIONS": r'\b(?:mentions|refers to|indicates|states)\b',
            "IS_TYPE": r'\b(?:is|are|type of|kind of)\b',
            "HAS_VALUE": r'\b(?:has|contains|includes|specifies)\b',
            "TEMPORAL": r'\b(?:before|after|during|when|while)\b',
            "CAUSAL": r'\b(?:because|due to|results in|causes)\b'
        }
    
    def extract_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """Extract relations between entities based on text patterns."""
        relations = []
        
        # Sort entities by position in text
        entities_by_pos = sorted(entities, key=lambda e: e.properties.get("start", 0))
        
        # Look for relations between nearby entities
        for i, entity1 in enumerate(entities_by_pos):
            for j, entity2 in enumerate(entities_by_pos[i+1:], i+1):
                # Skip if entities are too far apart
                start1 = entity1.properties.get("start", 0)
                end1 = entity1.properties.get("end", 0)
                start2 = entity2.properties.get("start", 0)
                
                if start2 - end1 > 100:  # Max distance of 100 characters
                    break
                
                # Extract text between entities
                between_text = text[end1:start2]
                
                # Check for relation patterns
                for relation_type, pattern in self.relation_patterns.items():
                    if re.search(pattern, between_text, re.IGNORECASE):
                        relation = Relation(
                            source_id=entity1.id,
                            target_id=entity2.id,
                            relation_type=relation_type,
                            properties={
                                "text_between": between_text.strip(),
                                "distance": start2 - end1
                            },
                            confidence=0.7
                        )
                        relations.append(relation)
                        break  # Only add one relation type per entity pair
        
        return relations


class GraphRetriever:
    """Retrieve relevant subgraphs for extraction tasks."""
    
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg = knowledge_graph
        
    def retrieve_for_schema(self, schema: Dict[str, Any], max_entities: int = 10) -> KnowledgeGraph:
        """Retrieve relevant subgraph for schema-based extraction."""
        relevant_entities = []
        
        if "field" in schema:
            for field_name in schema["field"].keys():
                # Get entities of this type
                field_entities = self.kg.get_entities_by_type(field_name)
                relevant_entities.extend([e.id for e in field_entities[:max_entities//len(schema["field"])]])
        
        # Add some general entities for context
        general_types = ["CAPITALIZED_WORD", "NUMBER", "MONEY", "DATE"]
        for entity_type in general_types:
            type_entities = self.kg.get_entities_by_type(entity_type)
            relevant_entities.extend([e.id for e in type_entities[:2]])
        
        # Extract subgraph
        return self.kg.get_subgraph(relevant_entities[:max_entities])
    
    def retrieve_for_classification(self, schema: Dict[str, Any]) -> KnowledgeGraph:
        """Retrieve subgraph for classification tasks."""
        # For classification, we want entities that match enum values
        relevant_entities = []
        
        if "field" in schema:
            for field_name, field_def in schema["field"].items():
                if "enum" in field_def:
                    # Look for entities that might match enum values
                    field_entities = self.kg.get_entities_by_type(field_name)
                    relevant_entities.extend([e.id for e in field_entities])
        
        return self.kg.get_subgraph(relevant_entities, max_depth=1)


def build_knowledge_graph(text: str, schema: Dict[str, Any]) -> KnowledgeGraph:
    """Build a knowledge graph from text and schema."""
    kg = KnowledgeGraph()
    
    # Extract entities
    entity_extractor = EntityExtractor(schema)
    entities = entity_extractor.extract_entities(text)
    
    # Add entities to graph
    for entity in entities:
        kg.add_entity(entity)
    
    # Extract relations
    relation_extractor = RelationExtractor()
    relations = relation_extractor.extract_relations(text, entities)
    
    # Add relations to graph
    for relation in relations:
        kg.add_relation(relation)
    
    logger.info(f"Built knowledge graph with {len(entities)} entities and {len(relations)} relations")
    
    return kg
