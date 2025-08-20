"""
Training data management example for contract few-shot extraction.
Demonstrates creating and managing *.labels.json files.
"""

import sys
from pathlib import Path

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    """Demonstrate training data management for few-shot contract extraction."""
    
    print("📚 DocuVerse: Training Data Management for Contract Extraction")
    print("=" * 70)
    
    try:
        from docuverse.utils.training_data_manager import TrainingDataManager
        
        # Initialize training data manager
        data_dir = Path(__file__).parent / "data" / "contracts"
        manager = TrainingDataManager(data_dir)
        
        print(f"📁 Training data directory: {data_dir}")
        
        # Validate existing training data
        print(f"\n🔍 Validating existing training data...")
        validation_report = manager.validate_training_data()
        
        print(f"📊 Validation Results:")
        print(f"   Total documents: {validation_report['total_documents']}")
        print(f"   Valid pairs: {validation_report['valid_pairs']}")
        print(f"   Invalid pairs: {validation_report['invalid_pairs']}")
        
        if validation_report['missing_labels']:
            print(f"   Missing labels: {len(validation_report['missing_labels'])}")
            for missing in validation_report['missing_labels'][:3]:
                print(f"      • {Path(missing).name}")
        
        if validation_report['missing_documents']:
            print(f"   Missing documents: {len(validation_report['missing_documents'])}")
            for missing in validation_report['missing_documents'][:3]:
                print(f"      • {Path(missing).name}")
        
        if validation_report['validation_errors']:
            print(f"   Validation errors: {len(validation_report['validation_errors'])}")
            for error in validation_report['validation_errors'][:2]:
                print(f"      • {Path(error['file']).name}: {error['error']}")
        
        # Show field coverage
        print(f"\n📋 Field Coverage:")
        for field, count in validation_report['field_coverage'].items():
            coverage = (count / validation_report['valid_pairs'] * 100) if validation_report['valid_pairs'] > 0 else 0
            print(f"   {field}: {count}/{validation_report['valid_pairs']} ({coverage:.0f}%)")
        
        # Show document types
        print(f"\n📄 Document Types:")
        for doc_type, count in validation_report['document_types'].items():
            print(f"   {doc_type}: {count}")
        
        # Get detailed training summary
        print(f"\n📈 Training Summary:")
        summary = manager.get_training_summary()
        print(f"   Data directory: {summary['data_directory']}")
        print(f"   Valid examples: {summary['total_examples']}")
        print(f"   Invalid examples: {summary['invalid_examples']}")
        
        print(f"\n🏆 Most Common Fields:")
        for field, count in summary['most_common_fields'][:5]:
            percentage = summary['coverage_percentage'].get(field, 0)
            print(f"   {field}: {count} examples ({percentage:.0f}%)")
        
        # Demonstrate creating a new training example
        print(f"\n🆕 Creating New Training Example...")
        
        new_contract_content = """
        PROFESSIONAL SERVICES AGREEMENT
        Contract Number: PSA-2024-004
        Date: August 20, 2025
        
        CONSULTING SERVICES for Digital Transformation
        
        CLIENT: Innovation Corp
        VENDOR: Digital Solutions LLC
        Contact: Sarah Wilson, Project Manager
        Email: sarah@digitalsolutions.com
        Phone: (555) 888-9999
        
        CONTRACT VALUE: $125,000 for 6-month engagement
        Payment Schedule: Monthly payments of $20,833.33
        
        PAYMENT TERMS: Net 15 days from invoice date
        Monthly invoicing on the 1st of each month.
        
        DELIVERY: Services delivered remotely with weekly check-ins.
        Final deliverables due within 180 days of contract start.
        
        COMPLIANCE: Standard professional services agreement.
        Confidentiality and IP protection clauses included.
        
        This agreement requires client approval before execution.
        """
        
        try:
            # Create document file
            new_doc_path, new_labels_path = manager.create_example_from_template(
                document_content=new_contract_content.strip(),
                filename="contract_004_example",
                payment_terms="Net 15 days from invoice date",
                contract_value=125000,
                document_type="contract"
            )
            
            print(f"✅ Created new training example:")
            print(f"   Document: {new_doc_path.name}")
            print(f"   Labels: {new_labels_path.name}")
            
            # Load and show the created labels
            import json
            with open(new_labels_path, 'r') as f:
                labels = json.load(f)
            
            print(f"\n📝 Generated Labels Preview:")
            print(f"   Payment Terms: {labels['payment_terms']['classification']}")
            print(f"   Contract Value: {labels['contract_value']['classification']}")
            print(f"   Document Type: {labels['document_type']}")
            
        except Exception as e:
            print(f"⚠️  Example creation skipped: {e}")
        
        # Export training data
        print(f"\n📤 Exporting Training Data...")
        try:
            export_path = Path(__file__).parent / "exported_training_data.json"
            exported_file = manager.export_training_data(export_path)
            print(f"✅ Exported training data to: {exported_file.name}")
            
            # Show export summary
            with open(exported_file, 'r') as f:
                export_data = json.load(f)
            
            metadata = export_data['metadata']
            print(f"   Exported examples: {metadata['total_examples']}")
            print(f"   Source directory: {Path(metadata['export_source']).name}")
            
        except Exception as e:
            print(f"⚠️  Export failed: {e}")
        
        # Show best practices
        print(f"\n💡 Training Data Best Practices:")
        print("✅ Use descriptive filenames (contract_001.txt, invoice_001.txt)")
        print("✅ Create corresponding *.labels.json for each document")
        print("✅ Include diverse examples (standard, non-standard, edge cases)")
        print("✅ Validate labels structure and completeness")
        print("✅ Maintain consistent field naming across examples")
        print("✅ Include hybrid fields with both extracted_text and classification")
        
        print(f"\n🔧 Label File Structure:")
        print("""   {
     "payment_terms": {
       "extracted_text": "Net 30 days from invoice date",
       "classification": "standard",
       "standardized_value": "Net 30"
     },
     "contract_value": {
       "extracted_text": "$75,000 annually", 
       "amount": 75000,
       "currency": "USD",
       "classification": "medium_value"
     },
     "document_type": "contract",
     "priority_level": "medium"
   }""")
        
        print(f"\n🎯 Usage in Few-Shot Extraction:")
        print("1. Place documents and labels in data/contracts/")
        print("2. Initialize FewShotExtractor with data_path parameter")
        print("3. System automatically loads examples for training")
        print("4. Examples guide extraction on new documents")
        print("5. Better accuracy through learned patterns")
        
        print(f"\n🚀 Advanced Features:")
        print("• Automatic validation of labels structure")
        print("• Field coverage analysis across training set")
        print("• Export to single JSON for model training")
        print("• Template-based example creation")
        print("• Integration with schema validation")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure the training data manager is properly set up.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
