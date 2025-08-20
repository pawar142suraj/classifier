"""
Classification example demonstrating enum-based document classification.
"""

import os
import json
import sys
from pathlib import Path

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    """Demonstrate classification capabilities."""
    
    print("🔍 DocuVerse: Document Classification with Enum Definitions")
    print("=" * 70)
    
    try:
        from docuverse.extractors.classification import ClassificationExtractor
        from docuverse.core.config import LLMConfig
        
        # Configure LLM (you'll need to set your API key)
        llm_config = LLMConfig(
            provider="openai",  # or "anthropic"
            model_name="gpt-4-turbo-preview",
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY")  # Set this environment variable
        )
        
        # Initialize classification extractor with schema
        extractor = ClassificationExtractor(
            llm_config=llm_config,
            schema_path="schemas/classification_schema.json"
        )
        
        print("📋 Initialized ClassificationExtractor with enum definitions")
        
        # Test documents with different characteristics
        test_documents = [
            {
                "name": "Urgent Invoice",
                "content": """
                URGENT: PAYMENT OVERDUE NOTICE
                
                Invoice Number: INV-2024-001
                Date: January 15, 2024
                Due Date: February 15, 2024 (30 DAYS OVERDUE)
                
                Dear Valued Customer,
                
                This is an urgent notice that your payment of $2,500.00 is now 
                30 days overdue. IMMEDIATE ACTION is required to avoid collection 
                proceedings and potential legal action.
                
                Please remit payment within 24 hours or contact us immediately.
                
                Billing Department
                XYZ Corporation
                """,
                "expected": {
                    "document_type": "invoice",
                    "priority_level": "urgent", 
                    "sentiment": "negative"
                }
            },
            {
                "name": "Service Agreement",
                "content": """
                SOFTWARE LICENSE AGREEMENT
                
                Agreement Number: SLA-2024-005
                Effective Date: March 1, 2024
                Expiration Date: March 1, 2025
                
                PARTIES:
                Licensor: TechSoft Solutions Inc.
                Licensee: Business Corp Ltd.
                
                This agreement grants Business Corp Ltd. the right to use 
                TechSoft's enterprise software suite for a period of one year.
                
                KEY TERMS:
                - Annual license fee: $50,000
                - Support included: 24/7 technical support
                - Termination: Either party may terminate with 30 days notice
                
                This agreement shall be governed by the laws of California.
                
                Signatures:
                [Signature blocks]
                """,
                "expected": {
                    "document_type": "contract",
                    "priority_level": "medium",
                    "sentiment": "neutral"
                }
            },
            {
                "name": "Thank You Email",
                "content": """
                Subject: Thank you for your excellent service!
                
                Dear Support Team,
                
                I wanted to reach out and express my sincere appreciation for 
                the outstanding service we received during our recent software 
                implementation. The team was professional, knowledgeable, and 
                went above and beyond to ensure everything was perfect.
                
                The project was completed ahead of schedule and under budget. 
                We're extremely satisfied with the results and look forward to 
                continuing our partnership.
                
                Best regards,
                Sarah Johnson
                Project Manager
                """,
                "expected": {
                    "document_type": "email",
                    "priority_level": "low",
                    "sentiment": "positive"
                }
            },
            {
                "name": "Compliance Report",
                "content": """
                QUARTERLY COMPLIANCE REPORT - Q1 2024
                
                Report ID: CR-2024-Q1-001
                Prepared by: Compliance Department
                Date: March 31, 2024
                
                EXECUTIVE SUMMARY:
                This report reviews compliance status across all business units 
                for Q1 2024. Overall compliance rating: 98.5%
                
                KEY FINDINGS:
                - All mandatory training completed: 100%
                - Policy violations: 2 minor incidents (resolved)
                - Audit recommendations: 3 items (2 completed, 1 in progress)
                - Regulatory updates: 5 new requirements implemented
                
                RECOMMENDATIONS:
                1. Update data privacy procedures by April 30
                2. Conduct additional training for new hires
                3. Review vendor compliance quarterly
                
                Status: All major compliance requirements met.
                """,
                "expected": {
                    "document_type": "report",
                    "priority_level": "medium",
                    "sentiment": "neutral"
                }
            }
        ]
        
        print(f"\n🔬 Testing classification on {len(test_documents)} documents...")
        
        # Process each document
        results = []
        for i, doc in enumerate(test_documents, 1):
            print(f"\n📄 Document {i}: {doc['name']}")
            print("-" * 40)
            
            # Create document object
            document = {
                "content": doc["content"],
                "file_type": "text",
                "metadata": {"name": doc["name"]}
            }
            
            if llm_config.api_key:
                # Run actual classification if API key is available
                try:
                    result = extractor.extract(document)
                    results.append(result)
                    
                    print("✅ Classification Results:")
                    for field, value in result.items():
                        if field != "extracted_content":
                            expected = doc["expected"].get(field, "N/A")
                            status = "✅" if value == expected else "⚠️"
                            print(f"  {field}: {value} {status} (expected: {expected})")
                    
                    # Show evidence if available
                    if "extracted_content" in result and "classification_evidence" in result["extracted_content"]:
                        print("\n📝 Classification Evidence:")
                        evidence = result["extracted_content"]["classification_evidence"]
                        for evidence_type, indicators in evidence.items():
                            if indicators:
                                print(f"  {evidence_type}: {indicators[:2]}...")  # Show first 2 indicators
                    
                    print(f"\n🎯 Confidence: {extractor.last_confidence:.2f}")
                    
                except Exception as e:
                    print(f"❌ Classification failed: {e}")
                    results.append({"error": str(e)})
            else:
                # Simulate results if no API key
                print("⚠️  No API key provided - showing expected results:")
                simulated_result = doc["expected"].copy()
                simulated_result["extracted_content"] = {
                    "key_entities": ["Sample entities"],
                    "classification_evidence": {
                        "document_type_indicators": ["Sample evidence"],
                        "priority_indicators": ["Sample priority evidence"],
                        "sentiment_indicators": ["Sample sentiment evidence"]
                    }
                }
                results.append(simulated_result)
                
                for field, value in doc["expected"].items():
                    print(f"  {field}: {value} ✅")
        
        # Generate classification statistics
        if results and not all("error" in r for r in results):
            print(f"\n📊 CLASSIFICATION STATISTICS")
            print("=" * 50)
            
            stats = extractor.get_classification_stats(results)
            
            print(f"Total documents: {stats['total_documents']}")
            print(f"Successful classifications: {stats['successful_classifications']}")
            
            if stats['field_distributions']:
                print("\n📈 Field Distributions:")
                for field, distribution in stats['field_distributions'].items():
                    print(f"\n  {field}:")
                    for value, count in distribution.items():
                        percentage = (count / stats['successful_classifications'] * 100) if stats['successful_classifications'] > 0 else 0
                        print(f"    {value}: {count} ({percentage:.1f}%)")
            
            if stats['confidence_stats']['mean'] > 0:
                print(f"\n🎯 Confidence Statistics:")
                print(f"  Mean: {stats['confidence_stats']['mean']:.2f}")
                print(f"  Min:  {stats['confidence_stats']['min']:.2f}")
                print(f"  Max:  {stats['confidence_stats']['max']:.2f}")
        
        print(f"\n🎉 Classification demonstration completed!")
        
        if not llm_config.api_key:
            print("\n💡 To run with actual LLM classification:")
            print("1. Set your API key: export OPENAI_API_KEY='your-key'")
            print("2. Run this script again")
        
        print("\n🔬 Classification Schema Features:")
        print("✅ Enum-based classification with detailed definitions")
        print("✅ Evidence extraction for classification decisions")
        print("✅ Automatic validation and repair")
        print("✅ Confidence scoring")
        print("✅ Batch processing capabilities")
        print("✅ Statistical analysis")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you've installed the dependencies and the framework is set up correctly.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
