#!/usr/bin/env python3
"""
Demo script showing Dynamic Graph RAG with graph database integration.
Demonstrates both in-memory and persistent graph storage options.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from docuverse.extractors.dynamic_graph_rag import DynamicGraphRAGExtractor
from docuverse.core.config import LLMConfig, DynamicGraphRAGConfig, GraphRAGConfig, LLMProvider
from docuverse.utils.graph_db import GraphDatabaseManager, InMemoryGraphDB

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demo_graph_database_options():
    """Demonstrate different graph database options."""
    print("\n🗄️ Graph Database Integration Options:")
    print("=" * 60)
    
    # Option 1: In-Memory (lightweight)
    print("\n1️⃣ In-Memory Knowledge Graph (Default)")
    print("   ✅ Fast processing")
    print("   ✅ No database setup required")
    print("   ❌ Data not persisted across sessions")
    print("   ❌ Limited to single document processing")
    
    # Option 2: Neo4j
    print("\n2️⃣ Neo4j Graph Database")
    print("   ✅ Persistent storage")
    print("   ✅ Advanced graph queries")
    print("   ✅ Multi-document knowledge aggregation")
    print("   ✅ Graph visualization")
    print("   ❌ Requires Neo4j installation")
    
    # Option 3: Check if Neo4j is available
    try:
        import neo4j
        print("   ✅ Neo4j driver available")
    except ImportError:
        print("   ❌ Neo4j driver not installed (pip install neo4j)")
    
    print("\n" + "=" * 60)


def demo_in_memory_storage():
    """Demo with in-memory storage (default)."""
    print("\n🧠 Demo: In-Memory Knowledge Graph Storage")
    print("-" * 50)
    
    # Configure for in-memory usage
    base_graph_config = GraphRAGConfig(
        use_lightweight_kg=True,  # This enables in-memory storage
        max_subgraph_size=30,
        entity_similarity_threshold=0.7
    )
    
    dynamic_config = DynamicGraphRAGConfig(
        base_graph_config=base_graph_config,
        expansion_strategy="uncertainty_based",
        max_expansion_depth=2,
        uncertainty_threshold=0.7
    )
    
    return dynamic_config


def demo_persistent_storage():
    """Demo with persistent storage attempt."""
    print("\n💾 Demo: Persistent Graph Database Storage")
    print("-" * 50)
    
    # Configure for persistent storage
    base_graph_config = GraphRAGConfig(
        use_lightweight_kg=False,  # This enables database storage
        graph_db_uri="bolt://localhost:7687",  # Neo4j default
        max_subgraph_size=30,
        entity_similarity_threshold=0.7
    )
    
    dynamic_config = DynamicGraphRAGConfig(
        base_graph_config=base_graph_config,
        expansion_strategy="uncertainty_based",
        max_expansion_depth=2,
        uncertainty_threshold=0.7
    )
    
    return dynamic_config


def test_graph_database_connectivity():
    """Test different graph database options."""
    print("\n🔌 Testing Graph Database Connectivity:")
    print("-" * 50)
    
    # Test in-memory database
    print("Testing in-memory database...")
    in_memory_db = InMemoryGraphDB()
    if in_memory_db.connect():
        print("✅ In-memory database: Available")
        in_memory_db.disconnect()
    else:
        print("❌ In-memory database: Failed")
    
    # Test Neo4j database
    print("Testing Neo4j database...")
    try:
        from docuverse.utils.graph_db import Neo4jGraphDB
        neo4j_db = Neo4jGraphDB("bolt://localhost:7687", "neo4j", "password")
        if neo4j_db.connect():
            print("✅ Neo4j database: Connected")
            neo4j_db.disconnect()
        else:
            print("❌ Neo4j database: Connection failed")
    except ImportError:
        print("❌ Neo4j database: Driver not installed")
    except Exception as e:
        print(f"❌ Neo4j database: {e}")


def load_schema(schema_path: str) -> dict:
    """Load extraction schema from file."""
    with open(schema_path, 'r') as f:
        return json.load(f)


def load_document(doc_path: str) -> dict:
    """Load document from file."""
    with open(doc_path, 'r') as f:
        content = f.read()
    
    return {
        "content": content,
        "metadata": {
            "source": doc_path,
            "timestamp": datetime.now().isoformat(),
            "type": "contract"
        }
    }


def main():
    """Run Dynamic Graph RAG with graph database integration demo."""
    print("🚀 Dynamic Graph RAG - Graph Database Integration Demo")
    print("=" * 70)
    
    # Show available options
    demo_graph_database_options()
    
    # Test connectivity
    test_graph_database_connectivity()
    
    # Load data
    schema_path = "schemas/contracts_schema_hybrid.json"
    doc_path = "data/contract1.txt"
    
    schema = load_schema(schema_path)
    document = load_document(doc_path)
    
    # Configure Ollama LLM
    llm_config = LLMConfig(
        provider=LLMProvider.OLLAMA,
        model_name="llama3.2:latest",
        ollama_base_url="http://localhost:11434",
        temperature=0.1,
        max_tokens=1024,
        timeout=120
    )
    
    # Demo 1: In-Memory Storage
    print("\n" + "=" * 70)
    dynamic_config_memory = demo_in_memory_storage()
    
    try:
        extractor_memory = DynamicGraphRAGExtractor(llm_config, dynamic_config_memory, schema)
        
        print("Running extraction with in-memory storage...")
        result_memory = extractor_memory.extract(document, schema)
        
        print("✅ In-memory extraction completed")
        print(f"Result: {json.dumps(result_memory, indent=2)}")
        
        metadata_memory = extractor_memory.get_extraction_metadata()
        print(f"Graph stats: {metadata_memory.get('graph_stats', {})}")
        
    except Exception as e:
        print(f"❌ In-memory extraction failed: {e}")
    
    # Demo 2: Persistent Storage (will fallback to in-memory if Neo4j not available)
    print("\n" + "=" * 70)
    dynamic_config_persistent = demo_persistent_storage()
    
    try:
        extractor_persistent = DynamicGraphRAGExtractor(llm_config, dynamic_config_persistent, schema)
        
        print("Running extraction with persistent storage attempt...")
        result_persistent = extractor_persistent.extract(document, schema)
        
        print("✅ Persistent storage extraction completed")
        print(f"Result: {json.dumps(result_persistent, indent=2)}")
        
        metadata_persistent = extractor_persistent.get_extraction_metadata()
        print(f"Graph stats: {metadata_persistent.get('graph_stats', {})}")
        
    except Exception as e:
        print(f"❌ Persistent storage extraction failed: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 Graph Database Integration Summary:")
    print("-" * 50)
    print("• In-Memory Storage: Fast, temporary knowledge graphs")
    print("• Persistent Storage: Durable, queryable knowledge graphs")
    print("• Automatic Fallback: System gracefully handles database unavailability")
    print("• Multi-Document Support: Persistent storage enables knowledge aggregation")
    
    print("\n🔧 To enable Neo4j persistent storage:")
    print("1. Install Neo4j: docker run -p 7474:7474 -p 7687:7687 neo4j")
    print("2. Install driver: pip install neo4j")
    print("3. Set use_lightweight_kg=False in configuration")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
