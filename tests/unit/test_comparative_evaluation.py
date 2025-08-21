#!/usr/bin/env python3
"""
Comparative evaluation between Few-Shot and Vector RAG extractors.
Tests both methods with the same data and provides detailed comparison.
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from docuverse.core.config import LLMConfig, VectorRAGConfig, ChunkingStrategy
from docuverse.extractors.few_shot import FewShotExtractor
from docuverse.extractors.vector_rag import VectorRAGExtractor


def run_comparative_evaluation():
    """Run comprehensive comparison between Few-Shot and Vector RAG methods."""
    
    print("🏁 COMPARATIVE EVALUATION: Few-Shot vs Vector RAG")
    print("=" * 70)
    
    # Load schema
    schema_path = Path(__file__).parent.parent.parent / "schemas" / "contracts_schema_hybrid.json"
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    
    # LLM configuration
    llm_config = LLMConfig(
        provider="ollama",
        model_name="llama3.2:latest",
        ollama_base_url="http://localhost:11434",
        temperature=0.1
    )
    
    # Test documents
    test_docs = load_test_documents()
    
    print(f"📊 Testing with {len(test_docs)} documents")
    print(f"📋 Schema has {len(schema.get('field', {}))} fields")
    
    # Results storage
    results = {
        "few_shot": [],
        "vector_rag": []
    }
    
    # Test Few-Shot Extractor
    print(f"\n🎯 Testing Few-Shot Extractor")
    print("-" * 40)
    
    few_shot_extractor = FewShotExtractor(
        llm_config=llm_config,
        schema=schema,
        auto_load_labels=True
    )
    
    print(f"✅ Loaded {len(few_shot_extractor.examples)} examples")
    
    for i, doc in enumerate(test_docs):
        print(f"\n📄 Document {i+1}: {doc['name']}")
        
        try:
            start_time = time.time()
            result = few_shot_extractor.extract(doc['document'])
            extraction_time = time.time() - start_time
            
            validation = few_shot_extractor.validate_schema_compliance(result)
            
            result_info = {
                "document": doc['name'],
                "result": result,
                "extraction_time": extraction_time,
                "confidence": few_shot_extractor.last_confidence,
                "validation_passed": validation["is_valid"],
                "validation_errors": validation.get("missing_fields", []) + 
                                   validation.get("invalid_enums", []) + 
                                   validation.get("structure_errors", []),
                "expected": doc.get('expected'),
                "method": "few_shot"
            }
            
            results["few_shot"].append(result_info)
            
            print(f"  ✅ Confidence: {few_shot_extractor.last_confidence:.3f}")
            print(f"  ⏱️ Time: {extraction_time:.2f}s")
            print(f"  ✔️ Validation: {'PASSED' if validation['is_valid'] else 'FAILED'}")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            results["few_shot"].append({
                "document": doc['name'],
                "error": str(e),
                "method": "few_shot"
            })
    
    # Test Vector RAG Extractor
    print(f"\n🔍 Testing Vector RAG Extractor")
    print("-" * 40)
    
    # Test different RAG configurations
    rag_configs = [
        {
            "name": "Optimized Config",
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
        }
    ]
    
    for rag_config_info in rag_configs:
        rag_name = rag_config_info["name"]
        rag_config = rag_config_info["config"]
        
        print(f"\n🔧 RAG Config: {rag_name}")
        
        try:
            vector_rag_extractor = VectorRAGExtractor(
                llm_config=llm_config,
                rag_config=rag_config,
                schema=schema,
                cache_embeddings=True
            )
            
            for i, doc in enumerate(test_docs):
                print(f"\n📄 Document {i+1}: {doc['name']}")
                
                try:
                    start_time = time.time()
                    result = vector_rag_extractor.extract(doc['document'])
                    extraction_time = time.time() - start_time
                    
                    analysis = vector_rag_extractor.get_retrieval_analysis()
                    
                    # Validate structure
                    validation_passed = validate_vector_rag_result(result, schema)
                    
                    result_info = {
                        "document": doc['name'],
                        "result": result,
                        "extraction_time": extraction_time,
                        "confidence": analysis['confidence'],
                        "chunks_processed": analysis['chunks_processed'],
                        "retrieval_time": analysis['retrieval_time'],
                        "rerank_time": analysis['rerank_time'],
                        "validation_passed": validation_passed,
                        "expected": doc.get('expected'),
                        "method": "vector_rag",
                        "rag_config": rag_name
                    }
                    
                    results["vector_rag"].append(result_info)
                    
                    print(f"  ✅ Confidence: {analysis['confidence']:.3f}")
                    print(f"  ⏱️ Time: {extraction_time:.2f}s")
                    print(f"  🔍 Chunks: {analysis['chunks_processed']}")
                    print(f"  ✔️ Validation: {'PASSED' if validation_passed else 'FAILED'}")
                    
                except Exception as e:
                    print(f"  ❌ Failed: {e}")
                    results["vector_rag"].append({
                        "document": doc['name'],
                        "error": str(e),
                        "method": "vector_rag",
                        "rag_config": rag_name
                    })
        
        except Exception as e:
            print(f"❌ Failed to initialize Vector RAG: {e}")
    
    # Generate comprehensive comparison
    print_comparative_analysis(results, schema)


def load_test_documents() -> List[Dict[str, Any]]:
    """Load test documents with expected results."""
    test_docs = []
    
    # Load main contract
    contract_path = Path(__file__).parent.parent.parent / "data" / "contract1.txt"
    if contract_path.exists():
        with open(contract_path, 'r') as f:
            content = f.read()
        
        # Load expected results
        label_path = Path(__file__).parent.parent.parent / "data" / "labels" / "contract1_label.json"
        expected = None
        if label_path.exists():
            with open(label_path, 'r') as f:
                expected = json.load(f)
        
        test_docs.append({
            "name": "contract1.txt",
            "document": {
                "content": content,
                "metadata": {
                    "filename": "contract1.txt",
                    "source": "test_data"
                }
            },
            "expected": expected
        })
    
    # Load additional test documents if available
    test_data_dir = Path(__file__).parent.parent / "data" / "data"
    if test_data_dir.exists():
        for doc_file in test_data_dir.glob("test_document_*.txt"):
            try:
                with open(doc_file, 'r') as f:
                    content = f.read()
                
                # Look for corresponding label
                label_file = test_data_dir.parent / "labels" / f"{doc_file.stem}_label.json"
                expected = None
                if label_file.exists():
                    with open(label_file, 'r') as f:
                        expected = json.load(f)
                
                test_docs.append({
                    "name": doc_file.name,
                    "document": {
                        "content": content,
                        "metadata": {
                            "filename": doc_file.name,
                            "source": "test_data"
                        }
                    },
                    "expected": expected
                })
                
                # Limit to first few documents for testing
                if len(test_docs) >= 3:
                    break
                    
            except Exception as e:
                print(f"Warning: Failed to load {doc_file}: {e}")
    
    return test_docs


def validate_vector_rag_result(result: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """Validate Vector RAG result structure."""
    if not result or "fields" not in result:
        return False
    
    fields = result["fields"]
    schema_fields = schema.get("field", {})
    
    for field_name, field_def in schema_fields.items():
        if field_name not in fields:
            return False
        
        field_data = fields[field_name]
        
        if "extracted_content" not in field_data:
            return False
        
        if "enum" in field_def and "classification" not in field_data:
            return False
        
        if "classification" in field_data:
            classification = field_data["classification"]
            if classification not in field_def.get("enum", []):
                return False
    
    return True


def calculate_accuracy_scores(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate accuracy scores for results with expected values."""
    scores = {
        "field_accuracy": 0.0,
        "classification_accuracy": 0.0,
        "overall_accuracy": 0.0
    }
    
    valid_results = [r for r in results if "error" not in r and r.get("expected")]
    
    if not valid_results:
        return scores
    
    total_fields = 0
    correct_fields = 0
    total_classifications = 0
    correct_classifications = 0
    
    for result_info in valid_results:
        result = result_info["result"]
        expected = result_info["expected"]
        
        result_fields = result.get("fields", {})
        expected_fields = expected.get("fields", {})
        
        for field_name, expected_field in expected_fields.items():
            total_fields += 1
            
            if field_name in result_fields:
                result_field = result_fields[field_name]
                
                # Check content accuracy
                expected_content = expected_field.get("extracted_content", "").strip().lower()
                result_content = result_field.get("extracted_content", "").strip().lower()
                
                # Simple overlap check
                if (expected_content and result_content and 
                    (expected_content in result_content or result_content in expected_content or
                     len(set(expected_content.split()) & set(result_content.split())) > 0)):
                    correct_fields += 1
                
                # Check classification accuracy
                if "classification" in expected_field:
                    total_classifications += 1
                    expected_class = expected_field["classification"]
                    result_class = result_field.get("classification", "")
                    
                    if expected_class == result_class:
                        correct_classifications += 1
    
    if total_fields > 0:
        scores["field_accuracy"] = correct_fields / total_fields
    
    if total_classifications > 0:
        scores["classification_accuracy"] = correct_classifications / total_classifications
    
    scores["overall_accuracy"] = (scores["field_accuracy"] + scores["classification_accuracy"]) / 2
    
    return scores


def print_comparative_analysis(results: Dict[str, List], schema: Dict[str, Any]):
    """Print comprehensive comparative analysis."""
    print("\n" + "=" * 80)
    print("📊 COMPARATIVE ANALYSIS: Few-Shot vs Vector RAG")
    print("=" * 80)
    
    # Filter successful results
    few_shot_success = [r for r in results["few_shot"] if "error" not in r]
    vector_rag_success = [r for r in results["vector_rag"] if "error" not in r]
    
    print(f"\n📈 Success Rates:")
    print(f"  🎯 Few-Shot: {len(few_shot_success)}/{len(results['few_shot'])} ({len(few_shot_success)/len(results['few_shot'])*100:.1f}%)")
    print(f"  🔍 Vector RAG: {len(vector_rag_success)}/{len(results['vector_rag'])} ({len(vector_rag_success)/len(results['vector_rag'])*100:.1f}%)")
    
    if few_shot_success and vector_rag_success:
        # Performance comparison
        fs_avg_time = sum(r["extraction_time"] for r in few_shot_success) / len(few_shot_success)
        vr_avg_time = sum(r["extraction_time"] for r in vector_rag_success) / len(vector_rag_success)
        
        fs_avg_conf = sum(r["confidence"] for r in few_shot_success) / len(few_shot_success)
        vr_avg_conf = sum(r["confidence"] for r in vector_rag_success) / len(vector_rag_success)
        
        print(f"\n⏱️ Average Extraction Time:")
        print(f"  🎯 Few-Shot: {fs_avg_time:.2f}s")
        print(f"  🔍 Vector RAG: {vr_avg_time:.2f}s")
        print(f"  🏆 Winner: {'Few-Shot' if fs_avg_time < vr_avg_time else 'Vector RAG'} ({abs(fs_avg_time - vr_avg_time):.2f}s faster)")
        
        print(f"\n📊 Average Confidence:")
        print(f"  🎯 Few-Shot: {fs_avg_conf:.3f}")
        print(f"  🔍 Vector RAG: {vr_avg_conf:.3f}")
        print(f"  🏆 Winner: {'Few-Shot' if fs_avg_conf > vr_avg_conf else 'Vector RAG'} ({abs(fs_avg_conf - vr_avg_conf):.3f} higher)")
        
        # Accuracy comparison
        fs_accuracy = calculate_accuracy_scores(few_shot_success)
        vr_accuracy = calculate_accuracy_scores(vector_rag_success)
        
        print(f"\n🎯 Accuracy Comparison:")
        print(f"  📝 Field Accuracy:")
        print(f"    🎯 Few-Shot: {fs_accuracy['field_accuracy']:.3f}")
        print(f"    🔍 Vector RAG: {vr_accuracy['field_accuracy']:.3f}")
        print(f"    🏆 Winner: {'Few-Shot' if fs_accuracy['field_accuracy'] > vr_accuracy['field_accuracy'] else 'Vector RAG'}")
        
        print(f"  🏷️ Classification Accuracy:")
        print(f"    🎯 Few-Shot: {fs_accuracy['classification_accuracy']:.3f}")
        print(f"    🔍 Vector RAG: {vr_accuracy['classification_accuracy']:.3f}")
        print(f"    🏆 Winner: {'Few-Shot' if fs_accuracy['classification_accuracy'] > vr_accuracy['classification_accuracy'] else 'Vector RAG'}")
        
        print(f"  🎖️ Overall Accuracy:")
        print(f"    🎯 Few-Shot: {fs_accuracy['overall_accuracy']:.3f}")
        print(f"    🔍 Vector RAG: {vr_accuracy['overall_accuracy']:.3f}")
        print(f"    🏆 Winner: {'Few-Shot' if fs_accuracy['overall_accuracy'] > vr_accuracy['overall_accuracy'] else 'Vector RAG'}")
        
        # Validation comparison
        fs_validation_rate = sum(1 for r in few_shot_success if r["validation_passed"]) / len(few_shot_success)
        vr_validation_rate = sum(1 for r in vector_rag_success if r["validation_passed"]) / len(vector_rag_success)
        
        print(f"\n✔️ Validation Success Rate:")
        print(f"  🎯 Few-Shot: {fs_validation_rate:.3f}")
        print(f"  🔍 Vector RAG: {vr_validation_rate:.3f}")
        print(f"  🏆 Winner: {'Few-Shot' if fs_validation_rate > vr_validation_rate else 'Vector RAG'}")
        
        # Detailed field-by-field comparison
        print(f"\n📋 Field-by-Field Analysis:")
        schema_fields = schema.get("field", {})
        
        for field_name, field_def in schema_fields.items():
            print(f"\n  🔍 {field_name}:")
            
            # Few-Shot field results
            fs_field_results = []
            for r in few_shot_success:
                field_data = r["result"].get("fields", {}).get(field_name, {})
                if field_data:
                    fs_field_results.append(field_data)
            
            # Vector RAG field results
            vr_field_results = []
            for r in vector_rag_success:
                field_data = r["result"].get("fields", {}).get(field_name, {})
                if field_data:
                    vr_field_results.append(field_data)
            
            # Content extraction success
            fs_content_success = sum(1 for f in fs_field_results if f.get("extracted_content", "").strip()) / max(len(fs_field_results), 1)
            vr_content_success = sum(1 for f in vr_field_results if f.get("extracted_content", "").strip()) / max(len(vr_field_results), 1)
            
            print(f"    📝 Content Extraction: FS={fs_content_success:.2f}, VR={vr_content_success:.2f}")
            
            # Classification success (if applicable)
            if "enum" in field_def:
                fs_class_success = sum(1 for f in fs_field_results if f.get("classification") in field_def["enum"]) / max(len(fs_field_results), 1)
                vr_class_success = sum(1 for f in vr_field_results if f.get("classification") in field_def["enum"]) / max(len(vr_field_results), 1)
                
                print(f"    🏷️ Classification: FS={fs_class_success:.2f}, VR={vr_class_success:.2f}")
    
    # Recommendations
    print(f"\n🚀 RECOMMENDATIONS:")
    
    if not few_shot_success and not vector_rag_success:
        print("  ❌ Both methods failed. Check LLM configuration and connectivity.")
    elif few_shot_success and not vector_rag_success:
        print("  🎯 Use Few-Shot method. Vector RAG needs dependency installation.")
    elif not few_shot_success and vector_rag_success:
        print("  🔍 Use Vector RAG method. Few-Shot may need more examples.")
    else:
        # Both succeeded, provide nuanced recommendations
        fs_avg_time = sum(r["extraction_time"] for r in few_shot_success) / len(few_shot_success)
        vr_avg_time = sum(r["extraction_time"] for r in vector_rag_success) / len(vector_rag_success)
        
        fs_avg_conf = sum(r["confidence"] for r in few_shot_success) / len(few_shot_success)
        vr_avg_conf = sum(r["confidence"] for r in vector_rag_success) / len(vector_rag_success)
        
        if fs_avg_conf > vr_avg_conf + 0.1:
            print("  🎯 Recommend Few-Shot for higher accuracy and confidence.")
        elif vr_avg_conf > fs_avg_conf + 0.1:
            print("  🔍 Recommend Vector RAG for better retrieval and accuracy.")
        elif fs_avg_time < vr_avg_time * 0.5:
            print("  🎯 Recommend Few-Shot for faster processing.")
        else:
            print("  🤝 Both methods perform similarly. Choose based on:")
            print("    • Few-Shot: Better for small, consistent documents with good examples")
            print("    • Vector RAG: Better for large, varied documents with complex content")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    run_comparative_evaluation()
