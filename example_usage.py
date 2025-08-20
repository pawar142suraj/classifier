"""
Example usage of the DocuVerse library for document extraction research.
"""

import os
import json
from pathlib import Path

# Add the src directory to the path for imports
import sys
sys.path.append(str(Path(__file__).parent / "src"))

from src.docuverse import DocumentExtractor, ExtractionConfig, Evaluator
from src.docuverse.core.config import ExtractionMethod, EvaluationMetric


def main():
    """Demonstrate the DocuVerse library capabilities."""
    
    print("🔬 DocuVerse: Advanced Document Information Extraction Research")
    print("=" * 70)
    
    # Configure extraction methods to compare
    config = ExtractionConfig(
        methods=[
            ExtractionMethod.FEW_SHOT,
            ExtractionMethod.VECTOR_RAG,
            ExtractionMethod.GRAPH_RAG,
            ExtractionMethod.REASONING_COT,
            ExtractionMethod.DYNAMIC_GRAPH_RAG
        ],
        schema_path='schemas/invoice_schema.json',
        evaluation_metrics=[
            EvaluationMetric.ACCURACY,
            EvaluationMetric.F1,
            EvaluationMetric.SEMANTIC_SIMILARITY
        ],
        few_shot_examples=[
            {
                "input": "INVOICE #12345\nDate: 2024-01-15\nAcme Corp\nTotal: $1,250.00",
                "output": {
                    "invoice_number": "12345",
                    "invoice_date": "2024-01-15",
                    "vendor": {"name": "Acme Corp"},
                    "total_amount": 1250.00,
                    "currency": "USD"
                }
            }
        ]
    )
    
    # Initialize extractor
    print("📋 Initializing DocumentExtractor...")
    extractor = DocumentExtractor(config)
    
    # Create sample document for testing
    sample_document = {
        "content": """
        INVOICE
        
        Invoice Number: INV-2024-001
        Date: March 15, 2024
        Due Date: April 15, 2024
        
        Bill To:
        TechStart Inc.
        123 Innovation Drive
        Silicon Valley, CA 94000
        
        From:
        CloudServices LLC
        456 Enterprise Blvd
        Business City, NY 10001
        Tax ID: 12-3456789
        
        Description                    Qty    Unit Price    Total
        Cloud Storage (1TB/month)       1      $99.99      $99.99
        Premium Support                 1      $199.99     $199.99
        Additional Bandwidth           2       $49.99      $99.98
        
        Subtotal:                                          $399.96
        Tax (8.5%):                                         $34.00
        Total:                                             $433.96
        
        Payment Terms: Net 30 days
        Currency: USD
        """,
        "file_type": "text",
        "metadata": {"source": "sample_invoice"}
    }
    
    # Create corresponding ground truth
    ground_truth = {
        "invoice_number": "INV-2024-001",
        "invoice_date": "2024-03-15",
        "due_date": "2024-04-15",
        "vendor": {
            "name": "CloudServices LLC",
            "address": "456 Enterprise Blvd, Business City, NY 10001",
            "tax_id": "12-3456789"
        },
        "customer": {
            "name": "TechStart Inc.",
            "address": "123 Innovation Drive, Silicon Valley, CA 94000"
        },
        "line_items": [
            {
                "description": "Cloud Storage (1TB/month)",
                "quantity": 1,
                "unit_price": 99.99,
                "total_price": 99.99
            },
            {
                "description": "Premium Support",
                "quantity": 1,
                "unit_price": 199.99,
                "total_price": 199.99
            },
            {
                "description": "Additional Bandwidth",
                "quantity": 2,
                "unit_price": 49.99,
                "total_price": 99.98
            }
        ],
        "subtotal": 399.96,
        "tax_amount": 34.00,
        "total_amount": 433.96,
        "currency": "USD"
    }
    
    print("🚀 Running extraction with all methods...")
    
    # Save sample document to a temporary file for testing
    temp_doc_path = "temp_sample_invoice.json"
    with open(temp_doc_path, 'w') as f:
        json.dump(sample_document, f, indent=2)
    
    try:
        # Run extraction and evaluation
        results = extractor.extract_and_evaluate(
            document_path=temp_doc_path,
            ground_truth=ground_truth
        )
        
        print("✅ Extraction completed! Generating comparison report...")
        
        # Generate comparison report
        evaluator = Evaluator()
        
        # Generate HTML report
        html_report = evaluator.generate_report(
            results,
            output_path="results/comparison_report.html",
            format="html"
        )
        
        # Generate Markdown report
        md_report = evaluator.generate_report(
            results,
            format="markdown"
        )
        
        print("\n📊 RESULTS SUMMARY")
        print("-" * 50)
        
        # Display results summary
        if "evaluation_results" in results:
            comparison = evaluator.compare_methods(results)
            
            print("\n🏆 Method Rankings:")
            for i, (method, score) in enumerate(comparison["overall_ranking"], 1):
                print(f"  {i}. {method}: {score:.3f}")
            
            print("\n📈 Detailed Metrics:")
            for method, metrics in results["evaluation_results"].items():
                if isinstance(metrics, dict) and "error" not in metrics:
                    print(f"\n  {method}:")
                    for metric, value in metrics.items():
                        print(f"    {metric}: {value:.3f}")
                else:
                    error_msg = metrics.get("error", "Unknown error") if isinstance(metrics, dict) else "Unknown error"
                    print(f"\n  {method}: ❌ {error_msg}")
        
        print(f"\n📄 Reports generated:")
        print(f"  - HTML: results/comparison_report.html")
        print(f"  - Markdown report content ready for papers")
        
        print("\n" + "=" * 70)
        print("🎉 Demo completed successfully!")
        print("\nNext steps for your research:")
        print("1. Run with real documents and datasets")
        print("2. Implement full Vector RAG with embeddings")
        print("3. Add graph database integration")
        print("4. Enhance reasoning with verifiers")
        print("5. Develop the full Dynamic Graph-RAG approach")
        
    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        print("Note: This is expected as LLM credentials are not configured")
        print("Set your OpenAI/Anthropic API key to run actual extractions")
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_doc_path):
            os.remove(temp_doc_path)


if __name__ == "__main__":
    main()
