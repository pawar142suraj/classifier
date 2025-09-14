#!/usr/bin/env python3
"""
Demo script for Hybrid Graph RAG Extractor
Combining Microsoft's GraphRAG with DocuVerse Dynamic Graph RAG
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from docuverse.extractors.hybrid_graph_rag import HybridGraphRAGExtractor
from docuverse.core.config import LLMConfig, LLMProvider
from docuverse.utils.llm_client import OllamaClient

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_contract_schema():
    """Load the hybrid contract schema."""
    schema_path = "schemas/contracts_schema_hybrid.json"
    try:
        with open(schema_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Schema file not found: {schema_path}")
        # Fallback schema
        return {
            "field": {
                "customer_name": {
                    "type": "string",
                    "description": "Name of the customer or client"
                },
                "payment_terms": {
                    "type": "string",
                    "enum": ["monthly", "yearly", "one-time"],
                    "description": "Payment frequency and terms"
                },
                "warranty": {
                    "type": "string", 
                    "enum": ["standard", "non_standard"],
                    "description": "Type of warranty provided"
                }
            }
        }

def load_sample_contract():
    """Load sample contract for testing."""
    contract_path = "data/contract1.txt"
    try:
        with open(contract_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        logger.warning(f"Contract file not found: {contract_path}, using sample text")
        return """
        SERVICE AGREEMENT
        
        This Service Agreement is entered into between TechCorp Solutions and Global Manufacturing Inc.
        
        CUSTOMER INFORMATION:
        Customer: Global Manufacturing Inc.
        Contact: John Smith, Procurement Manager
        
        PAYMENT TERMS:
        Payments are due monthly on the 15th of each month.
        Invoicing will be handled electronically through our billing system.
        
        WARRANTY INFORMATION:
        This service comes with our standard warranty coverage.
        All hardware components are covered under manufacturer warranty.
        Software support is provided as part of the standard warranty package.
        
        SERVICE DETAILS:
        - 24/7 technical support
        - Regular maintenance schedules
        - Emergency response within 4 hours
        
        This agreement is effective immediately upon signing.
        """

def create_output_directory():
    """Create output directory for results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output/hybrid_graph_rag_extraction_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def save_results(output_dir: str, results: dict, community_analysis: dict, hierarchical_info: dict):
    """Save extraction results and analysis."""
    
    # Save main results
    with open(f"{output_dir}/hybrid_extraction_result.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save community analysis
    with open(f"{output_dir}/community_analysis.json", 'w') as f:
        json.dump(community_analysis, f, indent=2, default=str)
    
    # Save hierarchical summary info
    with open(f"{output_dir}/hierarchical_summary.json", 'w') as f:
        json.dump(hierarchical_info, f, indent=2, default=str)
    
    # Create comprehensive summary
    summary_content = f"""# Hybrid Graph RAG Extraction Summary

## Extraction Results
- **Customer Name**: {results.get('extracted_fields', {}).get('customer_name', 'N/A')}
- **Payment Terms**: {results.get('extracted_fields', {}).get('payment_terms', 'N/A')}
- **Warranty**: {results.get('extracted_fields', {}).get('warranty', 'N/A')}

## Performance Metrics
- **Overall Confidence**: {results.get('overall_confidence', 0):.2%}
- **Graph Size**: {results.get('graph_size', 0)} entities
- **Iterations Used**: {results.get('iterations_used', 0)}

## Community Analysis
- **Total Communities**: {community_analysis.get('total_communities', 0)}
- **Average Community Confidence**: {sum(community_analysis.get('confidence_distribution', [0])) / max(len(community_analysis.get('confidence_distribution', [1])), 1):.2%}

## Schema Coverage
"""
    
    # Add schema coverage details
    for field_name, coverage in community_analysis.get('schema_coverage', {}).items():
        summary_content += f"- **{field_name}**: {coverage.get('max_relevance', 0):.2%} max relevance, {coverage.get('supporting_communities', 0)} supporting communities\n"
    
    summary_content += f"""
## Hierarchical Summary
- **Confidence**: {hierarchical_info.get('confidence', 0):.2%}
- **Communities Analyzed**: {hierarchical_info.get('communities_count', 0)}

### Global Summary
{hierarchical_info.get('global_summary', 'N/A')}

### Field-Specific Summaries
"""
    
    for field_name, summary in hierarchical_info.get('field_summaries', {}).items():
        summary_content += f"- **{field_name}**: {summary}\n"
    
    summary_content += """
## Evidence Details
"""
    
    # Add evidence details
    for evidence in results.get('evidence', []):
        summary_content += f"""
### {evidence.get('field_name', 'Unknown Field')}
- **Extracted Value**: {evidence.get('extracted_value', 'N/A')}
- **Confidence**: {evidence.get('confidence', 0):.2%}
- **Evidence Text**: {evidence.get('evidence_text', 'N/A')}
- **Reasoning**: {evidence.get('reasoning', 'N/A')}
- **Supporting Entities**: {', '.join(evidence.get('supporting_entities', []))}
- **Related Clauses**: {len(evidence.get('related_clauses', []))} clauses found
"""
    
    with open(f"{output_dir}/extraction_summary.md", 'w') as f:
        f.write(summary_content)

def demonstrate_hybrid_capabilities(extractor, document_text):
    """Demonstrate the hybrid extractor's unique capabilities."""
    
    print("\n" + "="*60)
    print("🔬 HYBRID GRAPH RAG DEMONSTRATION")
    print("="*60)
    
    print("\n📋 Document Preview:")
    print("-" * 40)
    print(document_text[:300] + "..." if len(document_text) > 300 else document_text)
    
    print("\n🚀 Starting Hybrid Extraction...")
    print("   Combining Microsoft's GraphRAG + DocuVerse Dynamic Refinement")
    
    # Perform extraction
    extraction_result = extractor.extract(document_text)
    
    print("\n✅ Extraction Complete!")
    print(f"   Confidence: {extraction_result.metadata.get('overall_confidence', 0):.2%}")
    print(f"   Graph Size: {extraction_result.metadata.get('graph_size', 0)} entities")
    print(f"   Iterations: {extraction_result.metadata.get('iterations_used', 0)}")
    
    # Display results
    print("\n📊 EXTRACTION RESULTS:")
    print("-" * 40)
    for field_name, value in extraction_result.extracted_data.items():
        print(f"   {field_name}: {value}")
    
    # Community analysis
    print("\n🏘️ COMMUNITY ANALYSIS:")
    print("-" * 40)
    community_analysis = extractor.get_community_analysis()
    print(f"   Total Communities: {community_analysis.get('total_communities', 0)}")
    
    for community in community_analysis.get('communities', [])[:3]:  # Show top 3
        print(f"   • {community['id']}: {community['size']} entities, {community['confidence']:.2%} confidence")
        print(f"     Summary: {community['summary'][:100]}...")
    
    # Hierarchical summary
    print("\n🔄 HIERARCHICAL ANALYSIS:")
    print("-" * 40)
    hierarchical_info = extractor.get_hierarchical_summary_info()
    if hierarchical_info.get('global_summary'):
        print(f"   Global Summary: {hierarchical_info['global_summary'][:150]}...")
        print(f"   Field Summaries: {len(hierarchical_info.get('field_summaries', {}))} fields analyzed")
    
    # Evidence tracking
    print("\n🔍 EVIDENCE TRACKING:")
    print("-" * 40)
    for evidence in list(extraction_result.evidence.values())[:2]:  # Show first 2
        print(f"   Field: {evidence.field_name}")
        print(f"   Value: {evidence.extracted_value}")
        print(f"   Confidence: {evidence.confidence:.2%}")
        print(f"   Evidence: {evidence.evidence_text[:100]}...")
        print(f"   Supporting Entities: {len(evidence.supporting_entities)}")
        print()
    
    return extraction_result, community_analysis, hierarchical_info

def compare_approaches():
    """Display comparison of the three approaches."""
    
    print("\n" + "="*80)
    print("📊 APPROACH COMPARISON")
    print("="*80)
    
    comparison_table = """
┌─────────────────────────┬──────────────────┬──────────────────┬──────────────────┐
│ Feature                 │ Microsoft        │ DocuVerse        │ Hybrid           │
│                         │ GraphRAG         │ Dynamic          │ (This)           │
├─────────────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Community Detection     │ ✅ Leiden Algo   │ ❌ No            │ ✅ Schema-Aware  │
│ Hierarchical Summary    │ ✅ Multi-level   │ ❌ No            │ ✅ Field-Focused │
│ Uncertainty Tracking   │ ❌ No            │ ✅ Multi-dim     │ ✅ Enhanced      │
│ Iterative Refinement   │ ❌ Single-pass   │ ✅ Adaptive      │ ✅ Multi-scale   │
│ Evidence Extraction     │ ❌ No            │ ✅ Clause-level  │ ✅ Community+    │
│ Schema Awareness        │ ❌ Generic       │ ✅ Schema-driven │ ✅ Global+Schema │
│ Auto-repair             │ ❌ No            │ ✅ Yes           │ ✅ Enhanced      │
│ Global Reasoning        │ ✅ Community     │ ❌ Local only    │ ✅ Multi-scale   │
└─────────────────────────┴──────────────────┴──────────────────┴──────────────────┘
"""
    
    print(comparison_table)
    
    print("\n🎯 KEY INNOVATIONS IN HYBRID APPROACH:")
    print("   1. 🌐 Multi-scale Context: Local + Community + Global + Uncertainty")
    print("   2. 🔄 Adaptive Communities: Schema-aware community detection")
    print("   3. 📊 Hierarchical Evidence: Community-supported evidence tracking")
    print("   4. 🎛️ Dynamic Refinement: Uncertainty-driven with global context")
    print("   5. 🛠️ Enhanced Auto-repair: Using community consensus")

def main():
    """Main demonstration function."""
    
    print("🚀 Hybrid Graph RAG Extractor Demo")
    print("   Combining Microsoft's GraphRAG with DocuVerse Dynamic Refinement")
    print("   " + "="*60)
    
    # Load schema and document
    schema = load_contract_schema()
    document_text = load_sample_contract()
    
    # Create output directory
    output_dir = create_output_directory()
    print(f"\n📁 Output directory: {output_dir}")
    
    # Initialize LLM (Ollama)
    try:
        llm_config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model_name="llama3.2:latest",
            base_url="http://localhost:11434",
            api_key="",
            max_tokens=1000,
            temperature=0.1
        )
        llm = OllamaClient(llm_config)
        print("✅ LLM initialized: Ollama llama3.2:latest")
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        print("❌ LLM initialization failed. Please ensure Ollama is running.")
        return
    
    # Configure hybrid extractor
    config = {
        'max_iterations': 3,
        'confidence_threshold': 0.8,
        'min_community_size': 2,  # Smaller for demo
        'use_hierarchical_reasoning': True
    }
    
    # Initialize hybrid extractor
    try:
        extractor = HybridGraphRAGExtractor(llm, schema, config)
        print("✅ Hybrid extractor initialized")
    except Exception as e:
        logger.error(f"Failed to initialize extractor: {e}")
        print("❌ Extractor initialization failed")
        return
    
    # Demonstrate capabilities
    try:
        extraction_result, community_analysis, hierarchical_info = demonstrate_hybrid_capabilities(
            extractor, document_text
        )
        
        # Convert extraction result to serializable format
        result_dict = {
            'extracted_fields': extraction_result.extracted_data,
            'overall_confidence': extraction_result.metadata.get('overall_confidence', 0),
            'evidence': [
                {
                    'field_name': e.field_name,
                    'extracted_value': e.extracted_value,
                    'evidence_text': e.evidence_text,
                    'confidence': e.confidence,
                    'reasoning': e.reasoning,
                    'source_location': e.source_location,
                    'supporting_entities': e.supporting_entities,
                    'related_clauses': e.related_clauses
                }
                for e in extraction_result.evidence.values()
            ],
            'graph_size': extraction_result.metadata.get('graph_size', 0),
            'iterations_used': extraction_result.metadata.get('iterations_used', 0)
        }
        
        # Save results
        save_results(output_dir, result_dict, community_analysis, hierarchical_info)
        print(f"\n💾 Results saved to: {output_dir}")
        
        # Show comparison
        compare_approaches()
        
        print("\n🎉 Demo completed successfully!")
        print(f"   Check {output_dir} for detailed results and analysis.")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print("❌ Demo failed. Check logs for details.")

if __name__ == "__main__":
    main()
