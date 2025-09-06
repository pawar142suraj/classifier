#!/usr/bin/env python3
"""
Setup script for graph database support.
Installs Neo4j driver and demonstrates persistent storage capabilities.
"""

import subprocess
import sys
import logging

logger = logging.getLogger(__name__)


def install_neo4j_driver():
    """Install Neo4j Python driver."""
    print("📦 Installing Neo4j Python driver...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "neo4j"])
        print("✅ Neo4j driver installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Neo4j driver: {e}")
        return False


def test_neo4j_import():
    """Test if Neo4j can be imported."""
    try:
        import neo4j
        print("✅ Neo4j driver available")
        return True
    except ImportError:
        print("❌ Neo4j driver not available")
        return False


def demonstrate_persistent_storage():
    """Demonstrate persistent storage with mock data."""
    print("\n💾 Demonstrating Persistent Graph Storage:")
    print("-" * 50)
    
    try:
        from src.docuverse.utils.graph_db import GraphDatabaseManager, Neo4jGraphDB
        from src.docuverse.utils.graph_utils import KnowledgeGraph, Entity, Relation
        
        # Create a test knowledge graph
        kg = KnowledgeGraph()
        
        # Add test entities
        entities = [
            Entity("entity_1", "John Doe", "customer_name", {}, 0.9),
            Entity("entity_2", "monthly", "payment_terms", {}, 0.8),
            Entity("entity_3", "standard", "warranty", {}, 0.85)
        ]
        
        for entity in entities:
            kg.add_entity(entity)
        
        # Add test relations
        relations = [
            Relation("entity_1", "entity_2", "HAS_PAYMENT_TERMS", {}, 0.7),
            Relation("entity_1", "entity_3", "HAS_WARRANTY", {}, 0.75)
        ]
        
        for relation in relations:
            kg.add_relation(relation)
        
        print(f"Created test knowledge graph:")
        print(f"  • Entities: {len(kg.entities)}")
        print(f"  • Relations: {len(kg.relations)}")
        
        # Test with in-memory database (always works)
        print("\nTesting in-memory storage:")
        config = {"use_lightweight_kg": True}
        db_manager = GraphDatabaseManager(config)
        
        # Store and retrieve
        success = db_manager.store_knowledge_graph(kg, "test_document")
        print(f"  Store operation: {'✅ Success' if success else '❌ Failed'}")
        
        retrieved_kg = db_manager.retrieve_knowledge_graph("test_document")
        if retrieved_kg:
            print(f"  Retrieved: {len(retrieved_kg.entities)} entities, {len(retrieved_kg.relations)} relations")
        else:
            print("  ❌ Failed to retrieve")
        
        db_manager.close()
        
        # Test with Neo4j (if available)
        print("\nTesting Neo4j storage:")
        try:
            neo4j_config = {
                "graph_db_uri": "bolt://localhost:7687",
                "use_lightweight_kg": False
            }
            neo4j_manager = GraphDatabaseManager(neo4j_config)
            
            # This will attempt to connect to Neo4j
            success = neo4j_manager.store_knowledge_graph(kg, "test_document_neo4j")
            print(f"  Neo4j store: {'✅ Success' if success else '❌ Failed (Neo4j not running)'}")
            
            neo4j_manager.close()
            
        except Exception as e:
            print(f"  ❌ Neo4j test failed: {e}")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")


def show_neo4j_setup_instructions():
    """Show instructions for setting up Neo4j."""
    print("\n🐳 Neo4j Setup Instructions:")
    print("=" * 50)
    print("Option 1: Docker (Recommended)")
    print("  docker run -d \\")
    print("    --name neo4j \\")
    print("    -p 7474:7474 -p 7687:7687 \\")
    print("    -e NEO4J_AUTH=neo4j/password \\")
    print("    neo4j:latest")
    print("")
    print("Option 2: Local Installation")
    print("  1. Download from https://neo4j.com/download/")
    print("  2. Follow installation instructions")
    print("  3. Start Neo4j service")
    print("")
    print("Option 3: Neo4j Desktop")
    print("  1. Download Neo4j Desktop")
    print("  2. Create new database")
    print("  3. Start database")
    print("")
    print("🔗 Access Neo4j Browser: http://localhost:7474")
    print("🔑 Default credentials: neo4j/password")


def main():
    """Main setup function."""
    print("🚀 Graph Database Setup for Dynamic Graph RAG")
    print("=" * 60)
    
    # Check current status
    print("📋 Current Status:")
    neo4j_available = test_neo4j_import()
    
    if not neo4j_available:
        print("\n📦 Installing dependencies...")
        if install_neo4j_driver():
            neo4j_available = test_neo4j_import()
    
    # Demonstrate capabilities
    demonstrate_persistent_storage()
    
    # Show setup instructions
    show_neo4j_setup_instructions()
    
    print("\n📊 Summary:")
    print("=" * 60)
    print(f"✅ In-Memory Storage: Always available")
    print(f"{'✅' if neo4j_available else '❌'} Neo4j Driver: {'Available' if neo4j_available else 'Not installed'}")
    print(f"❓ Neo4j Server: Requires separate installation")
    
    print("\n🎯 Benefits of Persistent Storage:")
    print("• Knowledge graphs persist across sessions")
    print("• Multi-document knowledge aggregation")
    print("• Advanced graph queries and analytics")
    print("• Graph visualization capabilities")
    print("• Scalable for large document collections")


if __name__ == "__main__":
    main()
