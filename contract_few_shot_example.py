"""
Contract-focused few-shot extraction with dynamic ground truth loading.
Demonstrates hybrid extraction + classification using *.labels.json files.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    """Demonstrate few-shot extraction with contract focus and dynamic ground truth."""
    
    print("🔍 DocuVerse: Contract Few-Shot Extraction with Dynamic Ground Truth")
    print("=" * 80)
    
    try:
        from docuverse.extractors.few_shot import FewShotExtractor
        from docuverse.core.config import LLMConfig
        
        # Configure LLM
        llm_config = LLMConfig(
            provider="openai",  # or "anthropic"
            model_name="gpt-4-turbo-preview",
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY")  # Set this environment variable
        )
        
        # Path to contract data with ground truth labels
        data_path = Path(__file__).parent / "data" / "contracts"
        schema_path = Path(__file__).parent / "schemas" / "hybrid_schema.json"
        
        print(f"📁 Loading training examples from: {data_path}")
        print(f"📋 Using schema: {schema_path}")
        
        # Initialize few-shot extractor with dynamic ground truth loading
        extractor = FewShotExtractor(
            llm_config=llm_config,
            data_path=data_path,
            max_examples=3,
            schema_path=schema_path
        )
        
        # Show example summary
        summary = extractor.get_example_summary()
        print(f"\n📊 Training Examples Summary:")
        print(f"   Examples loaded: {summary['count']}")
        print(f"   Document types: {', '.join(summary['document_types'])}")
        print(f"   Fields covered: {len(summary['fields_covered'])}")
        
        if summary['count'] > 0:
            print(f"   Example sources: {[Path(f).name for f in summary['example_sources']]}")
            print(f"\n🎯 Key fields in training data:")
            for field in summary['fields_covered'][:8]:  # Show first 8 fields
                print(f"      • {field}")
            if len(summary['fields_covered']) > 8:
                print(f"      ... and {len(summary['fields_covered']) - 8} more")
        
        # Test document for extraction
        test_contract = {
            "name": "New Software License Agreement",
            "content": """
            ENTERPRISE SOFTWARE LICENSE AGREEMENT
            Contract Number: ENT-2024-005
            Date: August 16, 2025
            
            PARTIES:
            Vendor: CloudTech Solutions Inc.
            Address: 789 Cloud Drive, San Francisco, CA 94105
            Primary Contact: Alex Rodriguez, Sales Director
            Email: alex.rodriguez@cloudtech.com
            Phone: (415) 555-7890
            
            Client: Global Manufacturing Corp
            Address: 100 Industrial Parkway, Detroit, MI 48201
            Contact: Maria Santos, IT Director  
            Email: maria.santos@globalmfg.com
            Phone: (313) 555-4567
            
            CONTRACT VALUE: $850,000 over 3 years
            Annual payment: $283,333.33
            Currency: USD
            
            PAYMENT TERMS: Net 45 days from invoice date
            Invoices issued quarterly in advance.
            Late payment fee: 2% per month on overdue amounts.
            
            DELIVERY TERMS: Cloud deployment within 10 business days
            On-site training and setup included.
            24/7 premium support package included.
            
            COMPLIANCE: This agreement includes comprehensive data protection
            measures compliant with GDPR, CCPA, and SOX requirements.
            Regular security audits and penetration testing included.
            All data encrypted at rest and in transit.
            
            CONTACT INFORMATION:
            Technical Support: support@cloudtech.com, (415) 555-TECH
            Account Manager: Jennifer Kim, jennifer.kim@cloudtech.com, (415) 555-9999
            Legal Inquiries: legal@cloudtech.com
            
            PRIORITY: Standard enterprise agreement requiring management approval.
            
            GOVERNING LAW: California
            TERMINATION: 30-day notice required for termination.
            """,
            "expected_classification": {
                "payment_terms": "standard",
                "contract_value": "enterprise", 
                "delivery_terms": "standard_shipping",
                "compliance_requirements": "fully_compliant",
                "contact_information": "complete",
                "document_type": "contract",
                "priority_level": "high"
            }
        }
        
        print(f"\n🔬 Testing few-shot extraction on new contract...")
        print(f"📄 Document: {test_contract['name']}")
        print("-" * 60)
        
        # Create document object  
        document = {
            "content": test_contract["content"],
            "file_type": "text",
            "metadata": {"name": test_contract["name"]}
        }
        
        if llm_config.api_key and summary['count'] > 0:
            # Run actual extraction if API key is available and we have examples
            try:
                result = extractor.extract(document)
                
                print("✅ Few-Shot Extraction Results:")
                
                # Show payment terms
                if "payment_terms" in result:
                    pt = result["payment_terms"]
                    print(f"\n💰 Payment Terms:")
                    print(f"   Extracted Text: '{pt.get('extracted_text', 'N/A')}'")
                    print(f"   Classification: {pt.get('classification', 'N/A')}")
                    print(f"   Standardized: {pt.get('standardized_value', 'N/A')}")
                    
                    expected = test_contract["expected_classification"]["payment_terms"]
                    match = "✅" if pt.get('classification') == expected else "⚠️"
                    print(f"   Expected: {expected} {match}")
                
                # Show contract value
                if "contract_value" in result:
                    cv = result["contract_value"]
                    print(f"\n💼 Contract Value:")
                    print(f"   Extracted Text: '{cv.get('extracted_text', 'N/A')}'")
                    print(f"   Amount: ${cv.get('amount', 'N/A'):,}" if cv.get('amount') else "   Amount: N/A")
                    print(f"   Currency: {cv.get('currency', 'N/A')}")
                    print(f"   Classification: {cv.get('classification', 'N/A')}")
                    
                    expected = test_contract["expected_classification"]["contract_value"]
                    match = "✅" if cv.get('classification') == expected else "⚠️"
                    print(f"   Expected: {expected} {match}")
                
                # Show delivery terms
                if "delivery_terms" in result:
                    dt = result["delivery_terms"]
                    print(f"\n🚚 Delivery Terms:")
                    print(f"   Extracted Text: '{dt.get('extracted_text', 'N/A')}'")
                    print(f"   Classification: {dt.get('classification', 'N/A')}")
                    print(f"   Timeframe: {dt.get('delivery_timeframe', 'N/A')}")
                    
                    expected = test_contract["expected_classification"]["delivery_terms"]
                    match = "✅" if dt.get('classification') == expected else "⚠️"
                    print(f"   Expected: {expected} {match}")
                
                # Show compliance
                if "compliance_requirements" in result:
                    cr = result["compliance_requirements"]
                    print(f"\n🛡️ Compliance Requirements:")
                    print(f"   Classification: {cr.get('classification', 'N/A')}")
                    missing = cr.get('missing_elements', [])
                    print(f"   Missing Elements: {missing if missing else 'None'}")
                    
                    expected = test_contract["expected_classification"]["compliance_requirements"]
                    match = "✅" if cr.get('classification') == expected else "⚠️"
                    print(f"   Expected: {expected} {match}")
                
                # Show contact information
                if "contact_information" in result:
                    ci = result["contact_information"]
                    print(f"\n📞 Contact Information:")
                    print(f"   Classification: {ci.get('classification', 'N/A')}")
                    contacts = ci.get('extracted_contacts', [])
                    print(f"   Contacts Found: {len(contacts)}")
                    for i, contact in enumerate(contacts[:3], 1):  # Show first 3
                        print(f"      {i}. {contact.get('name', 'N/A')} ({contact.get('role', 'N/A')})")
                    
                    expected = test_contract["expected_classification"]["contact_information"]
                    match = "✅" if ci.get('classification') == expected else "⚠️"
                    print(f"   Expected: {expected} {match}")
                
                # Show document-level classification
                print(f"\n📋 Document Classification:")
                print(f"   Type: {result.get('document_type', 'N/A')}")
                print(f"   Priority: {result.get('priority_level', 'N/A')}")
                
                print(f"\n🎯 Extraction Confidence: {extractor.last_confidence:.2f}")
                
                # Count correct classifications
                correct_count = 0
                total_count = 0
                
                field_mapping = {
                    "payment_terms": "payment_terms",
                    "contract_value": "contract_value",
                    "delivery_terms": "delivery_terms", 
                    "compliance_requirements": "compliance_requirements",
                    "contact_information": "contact_information",
                    "document_type": "document_type",
                    "priority_level": "priority_level"
                }
                
                for field, expected in test_contract["expected_classification"].items():
                    if field in result:
                        if isinstance(result[field], dict):
                            actual = result[field].get('classification')
                        else:
                            actual = result[field]
                        
                        if actual == expected:
                            correct_count += 1
                        total_count += 1
                
                accuracy = correct_count / total_count if total_count > 0 else 0
                print(f"\n📊 Classification Accuracy: {correct_count}/{total_count} ({accuracy:.1%})")
                
            except Exception as e:
                print(f"❌ Extraction failed: {e}")
                
        elif not llm_config.api_key:
            print("⚠️  No API key provided - showing expected workflow:")
            print("\n💡 With API key, this would:")
            print("1. Use few-shot examples to guide extraction")
            print("2. Extract exact text from the new contract") 
            print("3. Classify content using learned patterns")
            print("4. Return structured hybrid results")
            
        elif summary['count'] == 0:
            print("⚠️  No training examples found - showing expected results:")
            print("\n💡 With training data, this would:")
            print("1. Load examples from *.labels.json files")
            print("2. Use examples to guide extraction patterns")
            print("3. Improve accuracy through few-shot learning")
        
        # Show training data structure
        print(f"\n📚 Training Data Structure:")
        print("   data/contracts/")
        print("   ├── contract_001.txt          # Document content")
        print("   ├── contract_001.labels.json  # Ground truth labels")
        print("   ├── contract_002.txt")
        print("   ├── contract_002.labels.json")
        print("   └── contract_003.txt")
        print("       contract_003.labels.json")
        
        print(f"\n🎓 Few-Shot Learning Benefits:")
        print("✅ Dynamic loading from *.labels.json files")
        print("✅ Contract-specific extraction patterns")
        print("✅ Hybrid extraction + classification examples")
        print("✅ Improved accuracy through learned patterns")
        print("✅ Easy addition of new training examples")
        print("✅ Schema-aware prompt construction")
        
        print(f"\n🚀 Next Steps:")
        print("• Add more contract examples with *.labels.json files")
        print("• Experiment with different max_examples settings")
        print("• Compare few-shot vs zero-shot performance")
        print("• Use for training data augmentation")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you've installed the dependencies and the framework is set up correctly.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
