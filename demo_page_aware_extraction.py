#!/usr/bin/env python3
"""
Demo: Page-Aware Evidence Extraction Across All RAG Extractors

This script demonstrates enhanced evidence tracking with page number extraction
across all RAG extractors: Hybrid Graph RAG, Dynamic Graph RAG, Vector RAG, 
and Reasoning extractors.

Features demonstrated:
- Page boundary detection and text location tracking
- Evidence enhancement with page numbers and context
- Comprehensive page-aware extraction across all methods
- Detailed evidence reporting with page information
"""

import sys
import os
from pathlib import Path
import json
from datetime import datetime
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from docuverse.extractors.hybrid_graph_rag import HybridGraphRAGExtractor
from docuverse.extractors.dynamic_graph_rag import DynamicGraphRAGExtractor
from docuverse.extractors.vector_rag import VectorRAGExtractor
from docuverse.extractors.reasoning import ReasoningExtractor
from docuverse.core.config import LLMConfig, LLMProvider, VectorRAGConfig, ReasoningConfig, ChunkingStrategy, ExtractionMethod
from docuverse.utils.page_extractor import page_extractor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mock LLM for demonstration
class MockLLM:
    def __init__(self):
        self.model_name = "mock-llama3.2"
    
    def generate(self, prompt, **kwargs):
        # Simple mock responses for demo
        if "customer" in prompt.lower():
            return "Acme Corporation"
        elif "payment" in prompt.lower():
            return "monthly"
        elif "warranty" in prompt.lower():
            return "standard"
        else:
            return "Extracted information"

def load_complex_contract():
    """Load a complex contract for testing."""
    contract_path = Path("data/complex_contract1.txt")
    if contract_path.exists():
        with open(contract_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        # Use a sample multi-page contract
        return """
PAGE 1 - MASTER SERVICE AGREEMENT

This Master Service Agreement ("Agreement") is entered into on [DATE] 
between Acme Corporation ("Customer") and TechService Inc. ("Contractor").

SECTION 1: PAYMENT TERMS
Customer shall pay Contractor on a monthly basis within 30 days of 
invoice receipt. Payment schedule: Monthly invoicing for services rendered.

SECTION 2: WARRANTY PROVISIONS  
Contractor provides standard warranty coverage for all deliverables
for a period of 12 months from completion date.

---PAGE BREAK---

PAGE 2 - TECHNICAL SPECIFICATIONS

SECTION 3: SERVICE DELIVERY
The Contractor shall provide comprehensive technical services including
but not limited to system integration, maintenance, and support.

SECTION 4: TERMINATION CONDITIONS
Either party may terminate this agreement with 30 days written notice.
Termination procedures are outlined in the standard terms.

SECTION 5: LIABILITY LIMITATIONS
Contractor's liability is limited to the total contract value.
Standard liability exclusions apply as per industry practice.

---PAGE BREAK---

PAGE 3 - COMPLIANCE REQUIREMENTS

SECTION 6: FEDERAL REQUIREMENTS
This contract complies with all applicable federal regulations
and includes standard federal compliance provisions.

SECTION 7: LABOR PROVISIONS
Prevailing wages apply where required by law.
Labor escalation clauses are included for multi-year terms.

SECTION 8: INTELLECTUAL PROPERTY
All work product remains property of Customer upon payment.
Standard licensing terms apply for pre-existing IP.

---PAGE BREAK---

PAGE 4 - EXECUTION

SECTION 9: SIGNATURES
Customer: _________________ Date: _________
Contractor: _______________ Date: _________

This agreement supersedes all previous agreements between the parties.
"""

def create_sample_schema():
    """Create a sample schema for extraction."""
    return {
        "field": {
            "customer_name": {
                "type": "string",
                "description": "Name of the customer or client organization"
            },
            "payment_terms": {
                "type": "string", 
                "enum": ["monthly", "yearly", "one-time"],
                "description": "Payment frequency and terms"
            },
            "warranty": {
                "type": "string",
                "enum": ["standard", "non_standard"],
                "description": "Type of warranty coverage provided"
            },
            "termination": {
                "type": "string",
                "enum": ["Yes", "No"],
                "description": "Whether termination clauses are present"
            },
            "federal_requirements": {
                "type": "string",
                "enum": ["Yes", "No"], 
                "description": "Federal compliance requirements included"
            }
        }
    }

def demonstrate_page_aware_extraction():
    """Demonstrate page-aware extraction across all RAG extractors."""
    print("=" * 80)
    print("PAGE-AWARE EVIDENCE EXTRACTION DEMONSTRATION")
    print("=" * 80)
    
    # Load test document
    contract_text = load_complex_contract()
    schema = create_sample_schema()
    
    # Initialize mock LLM
    mock_llm = MockLLM()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"output/page_aware_extraction_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDocument loaded: {len(contract_text)} characters")
    
    # Extract page boundaries for analysis
    pages = page_extractor.extract_pages(contract_text)
    print(f"Detected {len(pages)} pages in document")
    
    # Results storage
    all_results = {}
    
    # 1. HYBRID GRAPH RAG EXTRACTION
    print("\n" + "=" * 50)
    print("1. HYBRID GRAPH RAG EXTRACTION")
    print("=" * 50)
    
    try:
        hybrid_extractor = HybridGraphRAGExtractor(
            llm=mock_llm,
            schema=schema,
            config={"page_aware": True}
        )
        
        hybrid_result = hybrid_extractor.extract(contract_text)
        all_results["hybrid_graph_rag"] = hybrid_result
        
        print(f"✓ Hybrid extraction completed")
        print(f"  - Fields extracted: {len(hybrid_result.extracted_data)}")
        print(f"  - Evidence entries: {len(hybrid_result.evidence)}")
        print(f"  - Overall confidence: {hybrid_result.metadata.get('overall_confidence', 0):.2%}")
        
        # Show page-aware evidence
        for field_name, evidence in hybrid_result.evidence.items():
            page_info = f"Page {evidence.page_number}/{evidence.total_pages}" if evidence.page_number else "Page unknown"
            print(f"  - {field_name}: {evidence.extracted_value} ({page_info})")
            
    except Exception as e:
        print(f"✗ Hybrid extraction failed: {e}")
        all_results["hybrid_graph_rag"] = {"error": str(e)}
    
    # 2. DYNAMIC GRAPH RAG EXTRACTION
    print("\n" + "=" * 50)
    print("2. DYNAMIC GRAPH RAG EXTRACTION")
    print("=" * 50)
    
    try:
        # Use the existing demo script pattern
        dynamic_config = {
            'extraction_schema': schema,
            'uncertainty_threshold': 0.7,
            'max_refinement_iterations': 2,
            'community_detection_resolution': 1.0,
            'enable_hierarchical_summarization': True,
            'llm_config': {
                'provider': 'ollama',
                'model_name': 'llama3.2:latest',
                'base_url': 'http://localhost:11434',
                'temperature': 0.1,
                'max_tokens': 2000
            }
        }
        
        dynamic_extractor = DynamicGraphRAGExtractor(config=dynamic_config)
        dynamic_result = dynamic_extractor.extract_information(contract_text)
        all_results["dynamic_graph_rag"] = dynamic_result
        
        print(f"✓ Dynamic extraction completed")
        
        # Check for evidence with page information
        if 'evidence' in dynamic_result:
            evidence_count = len(dynamic_result['evidence'])
            print(f"  - Evidence entries: {evidence_count}")
            
            for field_name, evidence in dynamic_result['evidence'].items():
                if hasattr(evidence, 'page_number') and evidence.page_number:
                    page_info = f"Page {evidence.page_number}/{evidence.total_pages}"
                    print(f"  - {field_name}: {evidence.extracted_value} ({page_info})")
                else:
                    print(f"  - {field_name}: {getattr(evidence, 'extracted_value', 'N/A')}")
        
    except Exception as e:
        print(f"✗ Dynamic extraction failed: {e}")
        all_results["dynamic_graph_rag"] = {"error": str(e)}
    
    # 3. VECTOR RAG EXTRACTION
    print("\n" + "=" * 50)
    print("3. VECTOR RAG EXTRACTION")
    print("=" * 50)
    
    try:
        # Create configs
        llm_config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model_name="llama3.2:latest",
            base_url="http://localhost:11434",
            api_key="",
            max_tokens=1000,
            temperature=0.1
        )
        
        rag_config = VectorRAGConfig(
            chunk_size=512,
            chunk_overlap=50,
            chunking_strategy=ChunkingStrategy.SEMANTIC,
            retrieval_k=5,
            rerank_top_k=3,
            use_hybrid_search=True,
            bm25_weight=0.3,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        vector_extractor = VectorRAGExtractor(
            llm_config=llm_config,
            rag_config=rag_config,
            schema=schema
        )
        
        document = {"text": contract_text, "metadata": {}}
        vector_result = vector_extractor.extract(document)
        all_results["vector_rag"] = vector_result
        
        print(f"✓ Vector RAG extraction completed")
        print(f"  - Fields processed: {len(vector_result.get('fields', {}))}")
        print(f"  - Chunks processed: {vector_result.get('metadata', {}).get('chunks_processed', 0)}")
        
        # Show field results with page info
        for field_name, field_data in vector_result.get('fields', {}).items():
            extracted_content = field_data.get('extracted_content', 'N/A')
            chunks_used = field_data.get('retrieval_metadata', {}).get('chunks_used', 0)
            print(f"  - {field_name}: {extracted_content} ({chunks_used} chunks)")
            
    except Exception as e:
        print(f"✗ Vector RAG extraction failed: {e}")
        all_results["vector_rag"] = {"error": str(e)}
    
    # 4. REASONING EXTRACTION (CoT)
    print("\n" + "=" * 50)
    print("4. REASONING EXTRACTION (CoT)")
    print("=" * 50)
    
    try:
        reasoning_config = ReasoningConfig(
            max_reasoning_steps=3,
            uncertainty_threshold=0.3,
            verification_enabled=True,
            use_retrieval_context=True
        )
        
        reasoning_extractor = ReasoningExtractor(
            llm_config=llm_config,
            reasoning_config=reasoning_config,
            method_type=ExtractionMethod.REASONING_COT,
            schema=schema,
            use_vector_rag=True
        )
        
        document = {"text": contract_text, "metadata": {}}
        reasoning_result = reasoning_extractor.extract(document)
        all_results["reasoning_cot"] = reasoning_result
        
        print(f"✓ Reasoning (CoT) extraction completed")
        print(f"  - Fields extracted: {len(reasoning_result.get('fields', {}))}")
        print(f"  - Reasoning steps: {reasoning_result.get('metadata', {}).get('reasoning_steps', 0)}")
        print(f"  - Overall confidence: {reasoning_result.get('metadata', {}).get('overall_confidence', 0):.2%}")
        
        # Show evidence with page information
        if 'evidence' in reasoning_result.get('metadata', {}):
            for evidence in reasoning_result['metadata']['evidence']:
                page_info = f"Page {evidence.page_number}/{evidence.total_pages}" if hasattr(evidence, 'page_number') and evidence.page_number else "Page unknown"
                print(f"  - {evidence.field_name}: {evidence.extracted_value} ({page_info})")
        
    except Exception as e:
        print(f"✗ Reasoning extraction failed: {e}")
        all_results["reasoning_cot"] = {"error": str(e)}
    
    # GENERATE COMPREHENSIVE REPORT
    print("\n" + "=" * 80)
    print("PAGE-AWARE EXTRACTION SUMMARY")
    print("=" * 80)
    
    # Page analysis
    print(f"\nDOCUMENT ANALYSIS:")
    print(f"- Total pages: {len(pages)}")
    print(f"- Total characters: {len(contract_text)}")
    print(f"- Average page length: {len(contract_text) // len(pages) if pages else 0} characters")
    
    # Method comparison
    print(f"\nMETHOD COMPARISON:")
    successful_methods = []
    failed_methods = []
    
    for method_name, result in all_results.items():
        if "error" in result:
            failed_methods.append(method_name)
            print(f"✗ {method_name}: Failed - {result['error']}")
        else:
            successful_methods.append(method_name)
            print(f"✓ {method_name}: Successful")
    
    print(f"\nSUCCESS RATE: {len(successful_methods)}/{len(all_results)} methods successful")
    
    # Save detailed results
    results_file = output_dir / "page_aware_extraction_results.json"
    
    # Prepare serializable results
    serializable_results = {}
    for method_name, result in all_results.items():
        try:
            # Convert dataclass objects to dictionaries
            if hasattr(result, '__dict__'):
                serializable_results[method_name] = vars(result)
            else:
                serializable_results[method_name] = result
        except Exception as e:
            serializable_results[method_name] = {"serialization_error": str(e)}
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": timestamp,
            "document_info": {
                "total_pages": len(pages),
                "total_characters": len(contract_text),
                "page_boundaries": [{"page": i+1, "start": page["start"], "end": page["end"]} 
                                  for i, page in enumerate(pages)]
            },
            "extraction_results": serializable_results,
            "summary": {
                "successful_methods": successful_methods,
                "failed_methods": failed_methods,
                "success_rate": len(successful_methods) / len(all_results) if all_results else 0
            }
        }, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    # Generate evidence report
    evidence_file = output_dir / "page_evidence_report.txt"
    with open(evidence_file, 'w', encoding='utf-8') as f:
        f.write("PAGE-AWARE EVIDENCE EXTRACTION REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Document: Multi-page contract ({len(pages)} pages)\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for method_name, result in all_results.items():
            f.write(f"\n{method_name.upper().replace('_', ' ')}\n")
            f.write("-" * 40 + "\n")
            
            if "error" in result:
                f.write(f"Status: FAILED - {result['error']}\n")
            else:
                f.write(f"Status: SUCCESS\n")
                
                # Try to extract evidence information
                if hasattr(result, 'evidence'):
                    for field_name, evidence in result.evidence.items():
                        f.write(f"\nField: {field_name}\n")
                        f.write(f"Value: {evidence.extracted_value}\n")
                        f.write(f"Page: {evidence.page_number}/{evidence.total_pages}\n")
                        f.write(f"Confidence: {evidence.confidence:.2%}\n")
                        f.write(f"Context: {evidence.page_context[:100]}...\n")
    
    print(f"Evidence report saved to: {evidence_file}")
    
    print(f"\n🎉 Page-aware extraction demonstration completed successfully!")
    print(f"📁 Results available in: {output_dir}")
    
    return all_results

if __name__ == "__main__":
    try:
        results = demonstrate_page_aware_extraction()
        print("\n✅ Demonstration completed successfully!")
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
