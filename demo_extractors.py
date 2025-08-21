#!/usr/bin/env python3
"""
Quick demonstration script showing both Few-Shot and Vector RAG extractors
working with contract data.
"""

import json
import time
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from docuverse.core.config import LLMConfig, VectorRAGConfig, ChunkingStrategy
from docuverse.extractors.few_shot import FewShotExtractor
from docuverse.extractors.vector_rag import VectorRAGExtractor


def demo_both_extractors():
    """Demonstrate both Few-Shot and Vector RAG extractors."""
    
    print("🚀 DocuVerse: Contract Information Extraction Demo")
    print("=" * 60)
    
    # Load schema
    schema_path = Path(__file__).parent / "schemas" / "contracts_schema_hybrid.json"
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    
    # Load test document
    contract_path = Path(__file__).parent / "data" / "contract1.txt"
    with open(contract_path, 'r') as f:
        contract_content = f.read()
    
    document = {
        "content": contract_content,
        "metadata": {
            "filename": "contract1.txt",
            "source": "demo"
        }
    }
    
    print(f"📄 Processing: {contract_path.name}")
    print(f"📊 Document length: {len(contract_content)} characters")
    print(f"📋 Schema fields: {list(schema['field'].keys())}")
    
    # Configure LLM
    llm_config = LLMConfig(
        provider="ollama",
        model_name="llama3.2:latest",
        ollama_base_url="http://localhost:11434",
        temperature=0.1
    )
    
    # Test Few-Shot Extractor
    print(f"\n🎯 Testing Few-Shot Extractor")
    print("-" * 40)
    
    try:
        few_shot_extractor = FewShotExtractor(
            llm_config=llm_config,
            schema=schema,
            auto_load_labels=True
        )
        
        print(f"✅ Loaded {len(few_shot_extractor.examples)} examples")
        
        start_time = time.time()
        fs_result = few_shot_extractor.extract(document)
        fs_time = time.time() - start_time
        
        fs_validation = few_shot_extractor.validate_schema_compliance(fs_result)
        
        print(f"⏱️ Processing time: {fs_time:.2f}s")
        print(f"📊 Confidence: {few_shot_extractor.last_confidence:.3f}")
        print(f"✔️ Validation: {'PASSED' if fs_validation['is_valid'] else 'FAILED'}")
        
        print(f"\n📋 Few-Shot Results:")
        for field_name, field_data in fs_result.get("fields", {}).items():
            content = field_data.get("extracted_content", "")[:50]
            classification = field_data.get("classification", "")
            print(f"  • {field_name}: {content}{'...' if len(content) == 50 else ''}")
            if classification:
                print(f"    Classification: {classification}")
        
    except Exception as e:
        print(f"❌ Few-Shot failed: {e}")
        fs_result = None
        fs_time = 0
    
    # Test Vector RAG Extractor
    print(f"\n🔍 Testing Vector RAG Extractor")
    print("-" * 40)
    
    try:
        rag_config = VectorRAGConfig(
            chunk_size=512,
            chunk_overlap=50,
            chunking_strategy=ChunkingStrategy.SEMANTIC,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            retrieval_k=5,
            rerank_top_k=3,
            use_hybrid_search=True,
            bm25_weight=0.3
        )
        
        vector_rag_extractor = VectorRAGExtractor(
            llm_config=llm_config,
            rag_config=rag_config,
            schema=schema,
            cache_embeddings=True
        )
        
        start_time = time.time()
        vr_result = vector_rag_extractor.extract(document)
        vr_time = time.time() - start_time
        
        analysis = vector_rag_extractor.get_retrieval_analysis()
        
        print(f"⏱️ Processing time: {vr_time:.2f}s")
        print(f"📊 Confidence: {analysis['confidence']:.3f}")
        print(f"🔍 Chunks processed: {analysis['chunks_processed']}")
        print(f"🔄 Retrieval time: {analysis['retrieval_time']:.3f}s")
        
        print(f"\n📋 Vector RAG Results:")
        for field_name, field_data in vr_result.get("fields", {}).items():
            content = field_data.get("extracted_content", "")[:50]
            classification = field_data.get("classification", "")
            print(f"  • {field_name}: {content}{'...' if len(content) == 50 else ''}")
            if classification:
                print(f"    Classification: {classification}")
        
    except Exception as e:
        print(f"❌ Vector RAG failed: {e}")
        vr_result = None
        vr_time = 0
    
    # Comparison
    if fs_result and vr_result:
        print(f"\n📊 Performance Comparison")
        print("-" * 40)
        
        print(f"⏱️ Speed:")
        print(f"  Few-Shot: {fs_time:.2f}s")
        print(f"  Vector RAG: {vr_time:.2f}s")
        if vr_time > 0:
            speedup = fs_time / vr_time
            print(f"  🏆 Vector RAG is {speedup:.1f}x faster")
        
        print(f"\n📊 Confidence:")
        fs_conf = few_shot_extractor.last_confidence if fs_result else 0
        vr_conf = analysis['confidence'] if vr_result else 0
        print(f"  Few-Shot: {fs_conf:.3f}")
        print(f"  Vector RAG: {vr_conf:.3f}")
        print(f"  🏆 Winner: {'Few-Shot' if fs_conf > vr_conf else 'Vector RAG'}")
        
        print(f"\n🎯 Recommendations:")
        if vr_time < fs_time * 0.6:
            print("  • Use Vector RAG for faster processing")
        if fs_conf > vr_conf + 0.1:
            print("  • Use Few-Shot for higher confidence")
        print("  • Vector RAG scales better for large documents")
        print("  • Few-Shot works well with good examples")
    
    print(f"\n🏁 Demo completed!")
    print(f"📖 See docs/VECTOR_RAG_GUIDE.md for detailed documentation")


if __name__ == "__main__":
    demo_both_extractors()
