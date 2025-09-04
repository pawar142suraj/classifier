#!/usr/bin/env python3
"""
Demo script for Graph RAG extractor with contracts schema.
"""

import json
import os
import sys
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from docuverse.extractors.graph_rag import GraphRAGExtractor
from docuverse.core.config import LLMConfig, GraphRAGConfig, LLMProvider


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
            "timestamp": datetime.now().isoformat()
        }
    }


def main():
    """Run Graph RAG extraction demo."""
    print("🔄 Graph RAG Extractor Demo")
    print("=" * 50)
    
    # Load schema and document
    schema_path = "schemas/contracts_schema_hybrid.json"
    doc_path = "data/contract1.txt"
    
    print(f"📄 Loading schema from: {schema_path}")
    schema = load_schema(schema_path)
    
    print(f"📄 Loading document from: {doc_path}")
    document = load_document(doc_path)
    
    # Display schema information
    print("\n📋 Schema Information:")
    if "field" in schema:
        for field_name, field_def in schema["field"].items():
            print(f"  • {field_name}: {field_def.get('description', 'No description')}")
            if "enum" in field_def:
                print(f"    Possible values: {field_def['enum']}")
    
    # Display document preview
    print(f"\n📖 Document Preview:")
    preview = document["content"][:300] + "..." if len(document["content"]) > 300 else document["content"]
    print(f"  {preview}")
    
    # Configure LLM (using Ollama)
    print("🔑 Using Ollama configuration")
    llm_config = LLMConfig(
        provider=LLMProvider.OLLAMA,
        model_name="llama3.2:latest",  # Using your available model
        ollama_base_url="http://localhost:11434",
        temperature=0.1,
        max_tokens=2000,
        timeout=120  # Increase timeout for local models
    )
    
    # Configure Graph RAG
    graph_config = GraphRAGConfig(
        max_subgraph_size=30,
        entity_similarity_threshold=0.8,
        use_lightweight_kg=True
    )
    
    # Initialize extractor
    print("\n🚀 Initializing Graph RAG Extractor...")
    extractor = GraphRAGExtractor(llm_config, graph_config, schema)
    
    try:
        # Perform extraction
        print("\n⚡ Running Graph RAG extraction...")
        result = extractor.extract(document, schema)
        
        # Display results
        print("\n✅ Extraction Results:")
        print(json.dumps(result, indent=2))
        
        # Display metadata
        metadata = extractor.get_extraction_metadata()
        print(f"\n📊 Extraction Metadata:")
        print(f"  • Confidence: {metadata.get('confidence', 'N/A')}")
        print(f"  • Token Usage: {metadata.get('token_usage', 'N/A')}")
        
        if "graph_stats" in metadata:
            stats = metadata["graph_stats"]
            print(f"  • Graph Entities: {stats.get('total_entities', 0)}")
            print(f"  • Graph Relations: {stats.get('total_relations', 0)}")
            print(f"  • Entity Types: {stats.get('entity_types', 0)}")
        
        # Analyze results
        print("\n🔍 Result Analysis:")
        if "field" in schema:
            for field_name, field_def in schema["field"].items():
                extracted_value = result.get(field_name)
                
                if extracted_value:
                    print(f"  ✅ {field_name}: {extracted_value}")
                    
                    # Validate enum values
                    if "enum" in field_def and extracted_value not in field_def["enum"]:
                        print(f"    ⚠️  Warning: '{extracted_value}' not in allowed values {field_def['enum']}")
                else:
                    print(f"  ❌ {field_name}: Not extracted")
        
        # Save results
        output_dir = f"output/graph_rag_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save extraction result
        result_file = os.path.join(output_dir, "graph_rag_result.json")
        with open(result_file, 'w') as f:
            json.dump({
                "extraction_result": result,
                "metadata": metadata,
                "schema": schema,
                "document_info": {
                    "source": doc_path,
                    "length": len(document["content"]),
                    "preview": document["content"][:200]
                }
            }, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"\n❌ Error during extraction: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
