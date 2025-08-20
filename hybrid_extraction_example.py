"""
Hybrid extraction and classification example.
Demonstrates extracting content AND classifying it simultaneously.
"""

import os
import json
import sys
from pathlib import Path

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    """Demonstrate hybrid extraction and classification capabilities."""
    
    print("🔍 DocuVerse: Hybrid Extraction + Classification")
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
        
        # Initialize hybrid extractor with schema
        extractor = ClassificationExtractor(
            llm_config=llm_config,
            schema_path="schemas/hybrid_schema.json"
        )
        
        print("📋 Initialized Hybrid Extractor (Extract + Classify)")
        
        # Test documents with various payment terms and contract details
        test_documents = [
            {
                "name": "Standard Contract with Net 30",
                "content": """
                SOFTWARE LICENSE AGREEMENT
                Contract Number: SLA-2024-008
                Date: August 1, 2024
                
                PARTIES:
                Vendor: TechSolutions Inc.
                Email: contracts@techsolutions.com
                Phone: (555) 123-4567
                
                Client: Business Corp
                Contact: John Smith, Procurement Manager
                Email: john.smith@businesscorp.com
                Phone: (555) 987-6543
                
                CONTRACT VALUE: $75,000 annually
                Currency: USD
                
                PAYMENT TERMS: Net 30 days from invoice date
                All invoices must be submitted by the 25th of each month.
                
                DELIVERY TERMS: Standard delivery within 5-7 business days
                Software deployment and training included.
                
                This agreement complies with all industry standards and 
                includes required data protection clauses per GDPR requirements.
                """,
                "expected": {
                    "payment_terms": {
                        "classification": "standard",
                        "extracted_text": "Net 30 days from invoice date"
                    },
                    "contract_value": {
                        "classification": "medium_value",
                        "amount": 75000
                    }
                }
            },
            {
                "name": "Non-Standard Contract with Custom Terms",
                "content": """
                URGENT: RUSH DELIVERY CONTRACT
                Contract ID: RUSH-2024-001
                
                This is a high-priority contract requiring immediate attention.
                
                PAYMENT TERMS: Payment due within 120 days of project completion
                with a 5% discount if paid within 10 days. Special arrangement
                for this client due to their preferred vendor status.
                
                CONTRACT VALUE: $2,500,000 (Two Million Five Hundred Thousand USD)
                This represents a significant enterprise engagement.
                
                DELIVERY TERMS: RUSH DELIVERY - All components must be delivered
                within 24 hours of order confirmation. Expedited shipping required.
                
                COMPLIANCE NOTE: This contract is missing standard compliance
                clauses and requires legal review before execution.
                
                Contact: Sarah Wilson (urgent@company.com, 555-URGENT)
                """,
                "expected": {
                    "payment_terms": {
                        "classification": "non_standard",
                        "extracted_text": "Payment due within 120 days of project completion with a 5% discount if paid within 10 days"
                    },
                    "contract_value": {
                        "classification": "enterprise",
                        "amount": 2500000
                    },
                    "delivery_terms": {
                        "classification": "rush"
                    }
                }
            },
            {
                "name": "Simple Receipt with Due on Receipt",
                "content": """
                RECEIPT
                
                Date: August 15, 2024
                Receipt #: RCP-001234
                
                Items:
                - Office supplies: $250.00
                - Software license: $99.00
                Total: $349.00
                
                PAYMENT TERMS: Due on Receipt
                Thank you for your business!
                
                Questions? Contact support@vendor.com
                """,
                "expected": {
                    "payment_terms": {
                        "classification": "standard",
                        "extracted_text": "Due on Receipt"
                    }
                }
            },
            {
                "name": "Contract with Incomplete Information",
                "content": """
                CONSULTING AGREEMENT
                
                This agreement outlines consulting services.
                Payment will be arranged separately.
                
                Services will be delivered as needed.
                
                For questions, please contact us.
                """,
                "expected": {
                    "payment_terms": {
                        "classification": "unknown",
                        "extracted_text": "Payment will be arranged separately"
                    }
                }
            }
        ]
        
        print(f"\n🔬 Testing hybrid extraction on {len(test_documents)} documents...")
        
        # Process each document
        results = []
        for i, doc in enumerate(test_documents, 1):
            print(f"\n📄 Document {i}: {doc['name']}")
            print("-" * 50)
            
            # Create document object
            document = {
                "content": doc["content"],
                "file_type": "text",
                "metadata": {"name": doc["name"]}
            }
            
            if llm_config.api_key:
                # Run actual extraction if API key is available
                try:
                    result = extractor.extract(document)
                    results.append(result)
                    
                    print("✅ Hybrid Extraction Results:")
                    
                    # Show payment terms extraction + classification
                    if "payment_terms" in result:
                        pt = result["payment_terms"]
                        print(f"\n💰 Payment Terms:")
                        print(f"   Extracted Text: '{pt.get('extracted_text', 'N/A')}'")
                        print(f"   Classification: {pt.get('classification', 'N/A')}")
                        print(f"   Standardized: {pt.get('standardized_value', 'N/A')}")
                        
                        expected_pt = doc.get("expected", {}).get("payment_terms", {})
                        if expected_pt:
                            class_match = "✅" if pt.get('classification') == expected_pt.get('classification') else "⚠️"
                            print(f"   Expected Classification: {expected_pt.get('classification')} {class_match}")
                    
                    # Show contract value extraction + classification
                    if "contract_value" in result:
                        cv = result["contract_value"]
                        print(f"\n💼 Contract Value:")
                        print(f"   Extracted Text: '{cv.get('extracted_text', 'N/A')}'")
                        print(f"   Amount: ${cv.get('amount', 'N/A'):,}" if cv.get('amount') else "   Amount: N/A")
                        print(f"   Currency: {cv.get('currency', 'N/A')}")
                        print(f"   Classification: {cv.get('classification', 'N/A')}")
                        
                        expected_cv = doc.get("expected", {}).get("contract_value", {})
                        if expected_cv:
                            class_match = "✅" if cv.get('classification') == expected_cv.get('classification') else "⚠️"
                            print(f"   Expected Classification: {expected_cv.get('classification')} {class_match}")
                    
                    # Show delivery terms if present
                    if "delivery_terms" in result:
                        dt = result["delivery_terms"]
                        print(f"\n🚚 Delivery Terms:")
                        print(f"   Extracted Text: '{dt.get('extracted_text', 'N/A')}'")
                        print(f"   Classification: {dt.get('classification', 'N/A')}")
                        print(f"   Timeframe: {dt.get('delivery_timeframe', 'N/A')}")
                    
                    # Show contact information if present
                    if "contact_information" in result:
                        ci = result["contact_information"]
                        print(f"\n📞 Contact Information:")
                        print(f"   Classification: {ci.get('classification', 'N/A')}")
                        contacts = ci.get('extracted_contacts', [])
                        if contacts:
                            for j, contact in enumerate(contacts[:2], 1):  # Show first 2 contacts
                                print(f"   Contact {j}: {contact.get('name', 'N/A')} ({contact.get('email', 'N/A')})")
                    
                    # Show document-level classification
                    print(f"\n📋 Document Classification:")
                    print(f"   Type: {result.get('document_type', 'N/A')}")
                    print(f"   Priority: {result.get('priority_level', 'N/A')}")
                    
                    print(f"\n🎯 Confidence: {extractor.last_confidence:.2f}")
                    
                except Exception as e:
                    print(f"❌ Extraction failed: {e}")
                    results.append({"error": str(e)})
            else:
                # Show expected results if no API key
                print("⚠️  No API key provided - showing expected results:")
                expected = doc.get("expected", {})
                
                if "payment_terms" in expected:
                    pt = expected["payment_terms"]
                    print(f"\n💰 Payment Terms:")
                    print(f"   Extracted Text: '{pt.get('extracted_text', 'N/A')}'")
                    print(f"   Classification: {pt.get('classification', 'N/A')} ✅")
                
                if "contract_value" in expected:
                    cv = expected["contract_value"]
                    print(f"\n💼 Contract Value:")
                    print(f"   Amount: ${cv.get('amount', 'N/A'):,}" if cv.get('amount') else "   Amount: N/A")
                    print(f"   Classification: {cv.get('classification', 'N/A')} ✅")
                
                # Add simulated result
                simulated_result = expected.copy()
                if not simulated_result:
                    simulated_result = {"document_type": "contract", "priority_level": "medium"}
                results.append(simulated_result)
        
        # Summary
        print(f"\n📊 HYBRID EXTRACTION SUMMARY")
        print("=" * 50)
        
        successful_extractions = sum(1 for r in results if "error" not in r)
        print(f"Successful extractions: {successful_extractions}/{len(results)}")
        
        # Analyze payment terms classifications
        payment_classifications = {}
        for result in results:
            if "payment_terms" in result and "classification" in result["payment_terms"]:
                classification = result["payment_terms"]["classification"]
                payment_classifications[classification] = payment_classifications.get(classification, 0) + 1
        
        if payment_classifications:
            print(f"\n💰 Payment Terms Classifications:")
            for classification, count in payment_classifications.items():
                print(f"   {classification}: {count}")
        
        print(f"\n🎉 Hybrid extraction demonstration completed!")
        
        if not llm_config.api_key:
            print("\n💡 To run with actual LLM extraction:")
            print("1. Set your API key: export OPENAI_API_KEY='your-key'")
            print("2. Run this script again")
        
        print("\n🔬 Hybrid Schema Features Demonstrated:")
        print("✅ Extract exact text content from documents")
        print("✅ Classify extracted content using enum definitions")
        print("✅ Handle both simple classification and hybrid fields")
        print("✅ Parse structured data (amounts, contacts, etc.)")
        print("✅ Provide evidence for both extraction and classification")
        print("✅ Handle missing or unclear information gracefully")
        
        print("\n🎯 Use Cases for Hybrid Extraction:")
        print("• Payment terms analysis (extract text + classify as standard/custom)")
        print("• Contract value assessment (extract amount + classify by tier)")
        print("• Delivery requirements (extract terms + classify urgency)")
        print("• Compliance checking (extract clauses + assess compliance level)")
        print("• Contact management (extract details + assess completeness)")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you've installed the dependencies and the framework is set up correctly.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
