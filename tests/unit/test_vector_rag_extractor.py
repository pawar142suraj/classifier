#!/usr/bin/env python3
"""
Test the Vector RAG extractor with real data from the data folder.
Comprehensive testing of all Vector RAG features and optimizations.
"""

import json
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from docuverse.core.config import LLMConfig, VectorRAGConfig, ChunkingStrategy
from docuverse.extractors.vector_rag import VectorRAGExtractor


def test_vector_rag_comprehensive():
    """Test Vector RAG extractor with comprehensive scenarios."""
    
    print("🚀 Testing Advanced Vector RAG Extractor")
    print("=" * 60)
    
    # Load schema
    schema_path = Path(__file__).parent.parent.parent / "schemas" / "contracts_schema_hybrid.json"
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    
    print(f"📋 Loaded schema with {len(schema.get('field', {}))} fields")
    
    # Test different LLM configurations
    llm_configs = [
        {
            "name": "Ollama (Llama 3.2)",
            "config": LLMConfig(
                provider="ollama",
                model_name="llama3.2:latest",
                ollama_base_url="http://localhost:11434",
                temperature=0.1
            )
        },
        # Add more configurations as needed
        # {
        #     "name": "OpenAI GPT-4",
        #     "config": LLMConfig(
        #         provider="openai",
        #         model_name="gpt-4-turbo-preview",
        #         temperature=0.1
        #     )
        # }
    ]
    
    # Test different RAG configurations
    rag_configs = [
        {
            "name": "Semantic Chunking + Hybrid Retrieval",
            "config": VectorRAGConfig(
                chunk_size=512,
                chunk_overlap=50,
                chunking_strategy=ChunkingStrategy.SEMANTIC,
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                retrieval_k=5,
                rerank_top_k=3,
                use_hybrid_search=True,
                bm25_weight=0.3
            )
        },
        {
            "name": "Fixed Size + High Retrieval",
            "config": VectorRAGConfig(
                chunk_size=256,
                chunk_overlap=25,
                chunking_strategy=ChunkingStrategy.FIXED_SIZE,
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                retrieval_k=8,
                rerank_top_k=4,
                use_hybrid_search=True,
                bm25_weight=0.4
            )
        },
        {
            "name": "Sliding Window + Pure Semantic",
            "config": VectorRAGConfig(
                chunk_size=384,
                chunk_overlap=75,
                chunking_strategy=ChunkingStrategy.SLIDING_WINDOW,
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                retrieval_k=6,
                rerank_top_k=3,
                use_hybrid_search=True,
                bm25_weight=0.2
            )
        },
        {
            "name": "Hierarchical + Balanced Hybrid",
            "config": VectorRAGConfig(
                chunk_size=400,
                chunk_overlap=40,
                chunking_strategy=ChunkingStrategy.HIERARCHICAL,
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                retrieval_k=7,
                rerank_top_k=3,
                use_hybrid_search=True,
                bm25_weight=0.35
            )
        }
    ]
    
    # Load test document
    contract_path = Path(__file__).parent.parent.parent / "data" / "contract1.txt"
    
    if not contract_path.exists():
        print(f"❌ Contract file not found: {contract_path}")
        return
    
    with open(contract_path, 'r') as f:
        contract_content = f.read()
    
    print(f"📄 Testing with: {contract_path.name}")
    print(f"📊 Document length: {len(contract_content)} characters")
    
    # Create document object
    document = {
        "content": contract_content,
        "metadata": {
            "filename": contract_path.name,
            "source": "test_data",
            "test_mode": True
        }
    }
    
    # Load expected results for comparison
    label_path = Path(__file__).parent.parent.parent / "data" / "labels" / "contract1_label.json"
    expected_results = None
    if label_path.exists():
        with open(label_path, 'r') as f:
            expected_results = json.load(f)
        print(f"📋 Loaded expected results for comparison")
    
    # Test all configurations
    results_summary = []
    
    for llm_config_info in llm_configs:
        llm_name = llm_config_info["name"]
        llm_config = llm_config_info["config"]
        
        print(f"\n🤖 Testing with LLM: {llm_name}")
        print("-" * 50)
        
        for rag_config_info in rag_configs:
            rag_name = rag_config_info["name"]
            rag_config = rag_config_info["config"]
            
            print(f"\n🔧 RAG Configuration: {rag_name}")
            
            try:
                # Initialize extractor
                extractor = VectorRAGExtractor(
                    llm_config=llm_config,
                    rag_config=rag_config,
                    schema=schema,
                    cache_embeddings=True
                )
                
                # Perform extraction
                start_time = time.time()
                result = extractor.extract(document)
                extraction_time = time.time() - start_time
                
                # Get analysis
                analysis = extractor.get_retrieval_analysis()
                
                print(f"✅ Extraction completed in {extraction_time:.2f}s")
                print(f"📊 Confidence: {analysis['confidence']:.3f}")
                print(f"🔍 Chunks processed: {analysis['chunks_processed']}")
                print(f"⏱️ Retrieval time: {analysis['retrieval_time']:.3f}s")
                print(f"🔄 Rerank time: {analysis['rerank_time']:.3f}s")
                
                # Validate results
                if validate_extraction_result(result, expected_results, schema):
                    print("✅ Validation: PASSED")
                    validation_status = "PASSED"
                else:
                    print("⚠️ Validation: FAILED")
                    validation_status = "FAILED"
                
                # Store results for summary
                results_summary.append({
                    "llm_config": llm_name,
                    "rag_config": rag_name,
                    "extraction_time": extraction_time,
                    "confidence": analysis['confidence'],
                    "chunks_processed": analysis['chunks_processed'],
                    "retrieval_time": analysis['retrieval_time'],
                    "rerank_time": analysis['rerank_time'],
                    "validation": validation_status,
                    "result": result
                })
                
                # Show extracted fields
                print("\n📋 Extracted Fields:")
                fields = result.get("fields", {})
                for field_name, field_data in fields.items():
                    extracted_content = field_data.get("extracted_content", "")
                    classification = field_data.get("classification", "")
                    
                    print(f"  • {field_name}:")
                    print(f"    Content: {extracted_content[:100]}{'...' if len(extracted_content) > 100 else ''}")
                    if classification:
                        print(f"    Classification: {classification}")
                
                # Compare with expected if available
                if expected_results:
                    comparison = compare_with_expected(result, expected_results)
                    print(f"\n📊 Comparison with Expected:")
                    for field, match_info in comparison.items():
                        status = "✅" if match_info["content_match"] else "❌"
                        print(f"  {status} {field}: Content Match = {match_info['content_match']}")
                        if "classification_match" in match_info:
                            cls_status = "✅" if match_info["classification_match"] else "❌"
                            print(f"    {cls_status} Classification Match = {match_info['classification_match']}")
                
            except Exception as e:
                print(f"❌ Extraction failed: {e}")
                import traceback
                print(f"Error details: {traceback.format_exc()}")
                
                results_summary.append({
                    "llm_config": llm_name,
                    "rag_config": rag_name,
                    "error": str(e),
                    "validation": "ERROR"
                })
            
            print("-" * 30)
    
    # Print comprehensive summary
    print_results_summary(results_summary)


def validate_extraction_result(result: dict, expected_results: dict, schema: dict) -> bool:
    """Validate extraction result against schema and expected results."""
    if not result or "fields" not in result:
        return False
    
    fields = result["fields"]
    schema_fields = schema.get("field", {})
    
    # Check if all schema fields are present
    for field_name in schema_fields:
        if field_name not in fields:
            print(f"❌ Missing field: {field_name}")
            return False
        
        field_data = fields[field_name]
        field_def = schema_fields[field_name]
        
        # Check structure
        if "extracted_content" not in field_data:
            print(f"❌ Missing extracted_content for {field_name}")
            return False
        
        # Check classification for enum fields
        if "enum" in field_def:
            if "classification" not in field_data:
                print(f"❌ Missing classification for {field_name}")
                return False
            
            classification = field_data["classification"]
            if classification not in field_def["enum"]:
                print(f"❌ Invalid classification for {field_name}: {classification}")
                return False
    
    return True


def compare_with_expected(result: dict, expected: dict) -> dict:
    """Compare extraction result with expected results."""
    comparison = {}
    
    result_fields = result.get("fields", {})
    expected_fields = expected.get("fields", {})
    
    for field_name in expected_fields:
        comparison[field_name] = {}
        
        if field_name in result_fields:
            result_field = result_fields[field_name]
            expected_field = expected_fields[field_name]
            
            # Compare extracted content
            result_content = result_field.get("extracted_content", "").strip().lower()
            expected_content = expected_field.get("extracted_content", "").strip().lower()
            
            # Check for substantial overlap (not exact match due to variations)
            content_match = (
                result_content == expected_content or
                result_content in expected_content or
                expected_content in result_content or
                len(set(result_content.split()) & set(expected_content.split())) > 0
            )
            
            comparison[field_name]["content_match"] = content_match
            
            # Compare classification if present
            if "classification" in expected_field:
                result_classification = result_field.get("classification", "")
                expected_classification = expected_field.get("classification", "")
                comparison[field_name]["classification_match"] = (
                    result_classification == expected_classification
                )
        else:
            comparison[field_name]["content_match"] = False
            comparison[field_name]["classification_match"] = False
    
    return comparison


def print_results_summary(results: list):
    """Print comprehensive summary of all test results."""
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE VECTOR RAG TEST RESULTS SUMMARY")
    print("=" * 80)
    
    if not results:
        print("❌ No results to summarize")
        return
    
    # Filter successful results
    successful_results = [r for r in results if "error" not in r]
    failed_results = [r for r in results if "error" in r]
    
    print(f"\n✅ Successful Extractions: {len(successful_results)}")
    print(f"❌ Failed Extractions: {len(failed_results)}")
    
    if successful_results:
        # Performance metrics
        avg_time = sum(r["extraction_time"] for r in successful_results) / len(successful_results)
        avg_confidence = sum(r["confidence"] for r in successful_results) / len(successful_results)
        avg_chunks = sum(r["chunks_processed"] for r in successful_results) / len(successful_results)
        
        print(f"\n📈 Performance Metrics (Average):")
        print(f"  ⏱️ Extraction Time: {avg_time:.2f}s")
        print(f"  📊 Confidence: {avg_confidence:.3f}")
        print(f"  🔍 Chunks Processed: {avg_chunks:.1f}")
        
        # Best performing configuration
        best_result = max(successful_results, key=lambda x: x["confidence"])
        print(f"\n🏆 Best Performing Configuration:")
        print(f"  🤖 LLM: {best_result['llm_config']}")
        print(f"  🔧 RAG: {best_result['rag_config']}")
        print(f"  📊 Confidence: {best_result['confidence']:.3f}")
        print(f"  ⏱️ Time: {best_result['extraction_time']:.2f}s")
        
        # Validation summary
        passed_validations = sum(1 for r in successful_results if r["validation"] == "PASSED")
        print(f"\n✅ Validation Success Rate: {passed_validations}/{len(successful_results)} ({passed_validations/len(successful_results)*100:.1f}%)")
        
        # Configuration performance breakdown
        print(f"\n📋 Configuration Performance Breakdown:")
        config_performance = {}
        for result in successful_results:
            key = f"{result['rag_config']}"
            if key not in config_performance:
                config_performance[key] = []
            config_performance[key].append(result)
        
        for config_name, config_results in config_performance.items():
            avg_conf = sum(r["confidence"] for r in config_results) / len(config_results)
            avg_time = sum(r["extraction_time"] for r in config_results) / len(config_results)
            passed = sum(1 for r in config_results if r["validation"] == "PASSED")
            
            print(f"\n  🔧 {config_name}:")
            print(f"    📊 Avg Confidence: {avg_conf:.3f}")
            print(f"    ⏱️ Avg Time: {avg_time:.2f}s")
            print(f"    ✅ Validation: {passed}/{len(config_results)}")
    
    if failed_results:
        print(f"\n❌ Failed Configurations:")
        for result in failed_results:
            print(f"  • {result['llm_config']} + {result['rag_config']}: {result['error']}")
    
    print("\n" + "=" * 80)


def test_individual_components():
    """Test individual Vector RAG components."""
    print("\n🧪 Testing Individual Vector RAG Components")
    print("=" * 60)
    
    # Test chunking strategies
    print("\n📄 Testing Chunking Strategies:")
    
    sample_text = """
    SERVICE AGREEMENT CONTRACT
    Contract Number: SA-2024-001
    Date: January 15, 2024
    
    PAYMENT TERMS:
    Payments are due every month on the 15th.
    
    WARRANTY INFORMATION:
    Standard warranty is provided for 1 year.
    
    CUSTOMER DETAILS:
    Customer Name: John Doe
    """
    
    from docuverse.extractors.vector_rag import AdvancedChunker
    
    strategies = ["fixed_size", "semantic", "sliding_window", "hierarchical"]
    
    for strategy in strategies:
        try:
            chunker = AdvancedChunker(chunk_size=100, overlap=20, strategy=strategy)
            chunks = chunker.chunk_document(sample_text)
            
            print(f"  🔧 {strategy.title()}: {len(chunks)} chunks")
            for i, chunk in enumerate(chunks[:2]):  # Show first 2 chunks
                print(f"    Chunk {i+1}: {chunk.content[:50]}...")
                print(f"    Score: {chunk.importance_score:.3f}")
            
        except Exception as e:
            print(f"  ❌ {strategy}: Failed - {e}")
    
    print("\n✅ Component testing completed")


if __name__ == "__main__":
    # Test individual components first
    test_individual_components()
    
    # Then run comprehensive tests
    test_vector_rag_comprehensive()
