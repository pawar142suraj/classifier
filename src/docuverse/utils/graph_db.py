"""
Graph database integration for persistent knowledge graph storage.
Supports Neo4j, ArangoDB, and other graph databases.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
import json
from dataclasses import asdict

from .graph_utils import Entity, Relation, KnowledgeGraph

logger = logging.getLogger(__name__)


class GraphDatabaseInterface(ABC):
    """Abstract interface for graph database operations."""
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to the graph database."""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the graph database."""
        pass
    
    @abstractmethod
    def store_knowledge_graph(self, kg: KnowledgeGraph, document_id: str) -> bool:
        """Store a knowledge graph in the database."""
        pass
    
    @abstractmethod
    def retrieve_knowledge_graph(self, document_id: str) -> Optional[KnowledgeGraph]:
        """Retrieve a knowledge graph from the database."""
        pass
    
    @abstractmethod
    def query_entities(self, entity_type: str, limit: int = 10) -> List[Entity]:
        """Query entities by type."""
        pass
    
    @abstractmethod
    def query_relations(self, source_id: str, relation_type: str = None) -> List[Relation]:
        """Query relations from a source entity."""
        pass
    
    @abstractmethod
    def clear_document_graph(self, document_id: str) -> bool:
        """Clear all graph data for a document."""
        pass


class Neo4jGraphDB(GraphDatabaseInterface):
    """Neo4j graph database implementation."""
    
    def __init__(self, uri: str, user: str = "neo4j", password: str = "password"):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        self._connected = False
    
    def connect(self) -> bool:
        """Connect to Neo4j database."""
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            
            # Test connection
            with self.driver.session() as session:
                session.run("MATCH (n) RETURN count(n) as count")
            
            self._connected = True
            logger.info(f"Connected to Neo4j at {self.uri}")
            return True
            
        except ImportError:
            logger.error("Neo4j driver not installed. Run: pip install neo4j")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from Neo4j."""
        if self.driver:
            self.driver.close()
            self._connected = False
            logger.info("Disconnected from Neo4j")
    
    def store_knowledge_graph(self, kg: KnowledgeGraph, document_id: str) -> bool:
        """Store knowledge graph in Neo4j."""
        if not self._connected:
            logger.error("Not connected to Neo4j")
            return False
        
        try:
            with self.driver.session() as session:
                # Clear existing data for this document
                self.clear_document_graph(document_id)
                
                # Store entities
                for entity in kg.entities.values():
                    entity_query = """
                    CREATE (e:Entity {
                        id: $id,
                        text: $text,
                        type: $type,
                        confidence: $confidence,
                        properties: $properties,
                        document_id: $document_id
                    })
                    """
                    session.run(entity_query, {
                        "id": entity.id,
                        "text": entity.text,
                        "type": entity.type,
                        "confidence": entity.confidence,
                        "properties": json.dumps(entity.properties),
                        "document_id": document_id
                    })
                
                # Store relations
                for relation in kg.relations:
                    relation_query = """
                    MATCH (source:Entity {id: $source_id, document_id: $document_id})
                    MATCH (target:Entity {id: $target_id, document_id: $document_id})
                    CREATE (source)-[r:RELATION {
                        type: $relation_type,
                        confidence: $confidence,
                        properties: $properties,
                        document_id: $document_id
                    }]->(target)
                    """
                    session.run(relation_query, {
                        "source_id": relation.source_id,
                        "target_id": relation.target_id,
                        "relation_type": relation.relation_type,
                        "confidence": relation.confidence,
                        "properties": json.dumps(relation.properties),
                        "document_id": document_id
                    })
                
                logger.info(f"Stored knowledge graph for document {document_id}: {len(kg.entities)} entities, {len(kg.relations)} relations")
                return True
                
        except Exception as e:
            logger.error(f"Failed to store knowledge graph: {e}")
            return False
    
    def retrieve_knowledge_graph(self, document_id: str) -> Optional[KnowledgeGraph]:
        """Retrieve knowledge graph from Neo4j."""
        if not self._connected:
            logger.error("Not connected to Neo4j")
            return None
        
        try:
            kg = KnowledgeGraph()
            
            with self.driver.session() as session:
                # Retrieve entities
                entity_query = """
                MATCH (e:Entity {document_id: $document_id})
                RETURN e.id as id, e.text as text, e.type as type, 
                       e.confidence as confidence, e.properties as properties
                """
                entity_result = session.run(entity_query, {"document_id": document_id})
                
                for record in entity_result:
                    entity = Entity(
                        id=record["id"],
                        text=record["text"],
                        type=record["type"],
                        confidence=record["confidence"],
                        properties=json.loads(record["properties"]) if record["properties"] else {}
                    )
                    kg.add_entity(entity)
                
                # Retrieve relations
                relation_query = """
                MATCH (source:Entity)-[r:RELATION]->(target:Entity)
                WHERE r.document_id = $document_id
                RETURN source.id as source_id, target.id as target_id,
                       r.type as relation_type, r.confidence as confidence,
                       r.properties as properties
                """
                relation_result = session.run(relation_query, {"document_id": document_id})
                
                for record in relation_result:
                    relation = Relation(
                        source_id=record["source_id"],
                        target_id=record["target_id"],
                        relation_type=record["relation_type"],
                        confidence=record["confidence"],
                        properties=json.loads(record["properties"]) if record["properties"] else {}
                    )
                    kg.add_relation(relation)
                
                logger.info(f"Retrieved knowledge graph for document {document_id}: {len(kg.entities)} entities, {len(kg.relations)} relations")
                return kg
                
        except Exception as e:
            logger.error(f"Failed to retrieve knowledge graph: {e}")
            return None
    
    def query_entities(self, entity_type: str, limit: int = 10) -> List[Entity]:
        """Query entities by type."""
        if not self._connected:
            return []
        
        try:
            with self.driver.session() as session:
                query = """
                MATCH (e:Entity {type: $entity_type})
                RETURN e.id as id, e.text as text, e.type as type,
                       e.confidence as confidence, e.properties as properties
                LIMIT $limit
                """
                result = session.run(query, {"entity_type": entity_type, "limit": limit})
                
                entities = []
                for record in result:
                    entity = Entity(
                        id=record["id"],
                        text=record["text"],
                        type=record["type"],
                        confidence=record["confidence"],
                        properties=json.loads(record["properties"]) if record["properties"] else {}
                    )
                    entities.append(entity)
                
                return entities
                
        except Exception as e:
            logger.error(f"Failed to query entities: {e}")
            return []
    
    def query_relations(self, source_id: str, relation_type: str = None) -> List[Relation]:
        """Query relations from a source entity."""
        if not self._connected:
            return []
        
        try:
            with self.driver.session() as session:
                if relation_type:
                    query = """
                    MATCH (source:Entity {id: $source_id})-[r:RELATION {type: $relation_type}]->(target:Entity)
                    RETURN source.id as source_id, target.id as target_id,
                           r.type as relation_type, r.confidence as confidence,
                           r.properties as properties
                    """
                    result = session.run(query, {"source_id": source_id, "relation_type": relation_type})
                else:
                    query = """
                    MATCH (source:Entity {id: $source_id})-[r:RELATION]->(target:Entity)
                    RETURN source.id as source_id, target.id as target_id,
                           r.type as relation_type, r.confidence as confidence,
                           r.properties as properties
                    """
                    result = session.run(query, {"source_id": source_id})
                
                relations = []
                for record in result:
                    relation = Relation(
                        source_id=record["source_id"],
                        target_id=record["target_id"],
                        relation_type=record["relation_type"],
                        confidence=record["confidence"],
                        properties=json.loads(record["properties"]) if record["properties"] else {}
                    )
                    relations.append(relation)
                
                return relations
                
        except Exception as e:
            logger.error(f"Failed to query relations: {e}")
            return []
    
    def clear_document_graph(self, document_id: str) -> bool:
        """Clear all graph data for a document."""
        if not self._connected:
            return False
        
        try:
            with self.driver.session() as session:
                # Delete relations first (due to constraints)
                session.run("MATCH ()-[r:RELATION {document_id: $document_id}]-() DELETE r", 
                           {"document_id": document_id})
                
                # Delete entities
                session.run("MATCH (e:Entity {document_id: $document_id}) DELETE e", 
                           {"document_id": document_id})
                
                logger.info(f"Cleared graph data for document {document_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to clear document graph: {e}")
            return False


class InMemoryGraphDB(GraphDatabaseInterface):
    """In-memory graph database implementation for testing."""
    
    def __init__(self):
        self.graphs: Dict[str, KnowledgeGraph] = {}
        self._connected = False
    
    def connect(self) -> bool:
        """Connect (no-op for in-memory)."""
        self._connected = True
        logger.info("Connected to in-memory graph database")
        return True
    
    def disconnect(self) -> None:
        """Disconnect (no-op for in-memory)."""
        self._connected = False
        logger.info("Disconnected from in-memory graph database")
    
    def store_knowledge_graph(self, kg: KnowledgeGraph, document_id: str) -> bool:
        """Store knowledge graph in memory."""
        self.graphs[document_id] = kg
        logger.info(f"Stored knowledge graph for document {document_id} in memory")
        return True
    
    def retrieve_knowledge_graph(self, document_id: str) -> Optional[KnowledgeGraph]:
        """Retrieve knowledge graph from memory."""
        kg = self.graphs.get(document_id)
        if kg:
            logger.info(f"Retrieved knowledge graph for document {document_id} from memory")
        return kg
    
    def query_entities(self, entity_type: str, limit: int = 10) -> List[Entity]:
        """Query entities by type across all graphs."""
        entities = []
        for kg in self.graphs.values():
            type_entities = kg.get_entities_by_type(entity_type)
            entities.extend(type_entities[:limit])
            if len(entities) >= limit:
                break
        return entities[:limit]
    
    def query_relations(self, source_id: str, relation_type: str = None) -> List[Relation]:
        """Query relations from a source entity."""
        relations = []
        for kg in self.graphs.values():
            if source_id in kg.entities:
                for relation in kg.relations:
                    if relation.source_id == source_id:
                        if relation_type is None or relation.relation_type == relation_type:
                            relations.append(relation)
        return relations
    
    def clear_document_graph(self, document_id: str) -> bool:
        """Clear graph data for a document."""
        if document_id in self.graphs:
            del self.graphs[document_id]
            logger.info(f"Cleared graph data for document {document_id} from memory")
        return True


class GraphDatabaseManager:
    """Manager for graph database operations with fallback support."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.primary_db: Optional[GraphDatabaseInterface] = None
        self.fallback_db: InMemoryGraphDB = InMemoryGraphDB()
        self._initialize_databases()
    
    def _initialize_databases(self):
        """Initialize graph databases based on configuration."""
        # Try to connect to primary database
        db_uri = self.config.get("graph_db_uri")
        
        if db_uri and db_uri.startswith("bolt://"):
            # Neo4j configuration
            self.primary_db = Neo4jGraphDB(
                uri=db_uri,
                user=self.config.get("graph_db_user", "neo4j"),
                password=self.config.get("graph_db_password", "password")
            )
            
            if not self.primary_db.connect():
                logger.warning("Failed to connect to Neo4j, falling back to in-memory storage")
                self.primary_db = None
        
        # Always initialize fallback
        self.fallback_db.connect()
    
    def store_knowledge_graph(self, kg: KnowledgeGraph, document_id: str) -> bool:
        """Store knowledge graph with fallback."""
        # Try primary database first
        if self.primary_db:
            if self.primary_db.store_knowledge_graph(kg, document_id):
                # Also store in fallback for reliability
                self.fallback_db.store_knowledge_graph(kg, document_id)
                return True
        
        # Fallback to in-memory
        return self.fallback_db.store_knowledge_graph(kg, document_id)
    
    def retrieve_knowledge_graph(self, document_id: str) -> Optional[KnowledgeGraph]:
        """Retrieve knowledge graph with fallback."""
        # Try primary database first
        if self.primary_db:
            kg = self.primary_db.retrieve_knowledge_graph(document_id)
            if kg:
                return kg
        
        # Fallback to in-memory
        return self.fallback_db.retrieve_knowledge_graph(document_id)
    
    def query_entities(self, entity_type: str, limit: int = 10) -> List[Entity]:
        """Query entities with fallback."""
        if self.primary_db:
            entities = self.primary_db.query_entities(entity_type, limit)
            if entities:
                return entities
        
        return self.fallback_db.query_entities(entity_type, limit)
    
    def close(self):
        """Close all database connections."""
        if self.primary_db:
            self.primary_db.disconnect()
        self.fallback_db.disconnect()
