"""
Quick test of the DocuVerse library framework.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all core imports work."""
    print("🔧 Testing DocuVerse imports...")
    
    try:
        from docuverse.core.config import ExtractionConfig, ExtractionMethod, EvaluationMetric
        print("✅ Core config imports successful")
        
        from docuverse.core.extractor import DocumentExtractor, ExtractionResult
        print("✅ Core extractor imports successful")
        
        from docuverse.evaluation.evaluator import Evaluator
        print("✅ Evaluator imports successful")
        
        from docuverse.extractors.few_shot import FewShotExtractor
        print("✅ Few-shot extractor imports successful")
        
        from docuverse.utils.document_loader import DocumentLoader
        print("✅ Document loader imports successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_configuration():
    """Test configuration creation."""
    print("\n🔧 Testing configuration...")
    
    try:
        from docuverse.core.config import ExtractionConfig, ExtractionMethod, EvaluationMetric
        
        config = ExtractionConfig(
            methods=[ExtractionMethod.FEW_SHOT, ExtractionMethod.VECTOR_RAG],
            evaluation_metrics=[EvaluationMetric.ACCURACY, EvaluationMetric.F1],
            schema_path='schemas/invoice_schema.json'
        )
        
        print(f"✅ Configuration created with {len(config.methods)} methods")
        print(f"✅ Evaluation metrics: {config.evaluation_metrics}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_document_loader():
    """Test document loader."""
    print("\n🔧 Testing document loader...")
    
    try:
        from docuverse.utils.document_loader import DocumentLoader
        
        loader = DocumentLoader()
        
        # Test with a simple text document
        test_content = "Test invoice\nInvoice #123\nTotal: $100"
        test_file = Path("test_doc.txt")
        
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        document = loader.load(test_file)
        
        print(f"✅ Document loaded: {document['file_type']}")
        print(f"✅ Content length: {len(document['content'])}")
        
        # Cleanup
        test_file.unlink()
        
        return True
        
    except Exception as e:
        print(f"❌ Document loader test failed: {e}")
        return False

def test_schema_validation():
    """Test schema validation."""
    print("\n🔧 Testing schema validation...")
    
    try:
        from docuverse.utils.schema_validator import SchemaValidator
        
        validator = SchemaValidator("schemas/invoice_schema.json")
        
        # Test valid data
        valid_data = {
            "invoice_number": "123",
            "invoice_date": "2024-01-01",
            "vendor": {"name": "Test Corp"},
            "line_items": [{"description": "Test", "quantity": 1, "unit_price": 10, "total_price": 10}],
            "total_amount": 10
        }
        
        result = validator.validate(valid_data)
        print(f"✅ Schema validation: {result.is_valid}")
        
        return True
        
    except Exception as e:
        print(f"❌ Schema validation test failed: {e}")
        return False

def test_classification_extractor():
    """Test classification extractor with enum definitions."""
    print("\n🔧 Testing classification extractor...")
    
    try:
        from docuverse.extractors.classification import ClassificationExtractor
        from src.docuverse.core.config import LLMConfig
        
        # Create LLM config (won't actually call LLM in test)
        llm_config = LLMConfig(
            provider="openai",
            model_name="gpt-4",
            api_key="test-key"
        )
        
        # Initialize classification extractor
        extractor = ClassificationExtractor(
            llm_config=llm_config,
            schema_path="schemas/classification_schema.json"
        )
        
        print("✅ Classification extractor initialized")
        
        # Test enum field extraction
        enum_fields = extractor._extract_enum_fields()
        print(f"✅ Found {len(enum_fields)} enum fields")
        
        # Test document classification structure
        test_document = {
            "content": """
            URGENT: Invoice Payment Overdue
            
            Dear Customer,
            
            This is to notify you that invoice #INV-2024-001 for $1,500.00 
            is now 30 days overdue. Immediate payment is required to avoid 
            collection actions.
            
            Please remit payment within 24 hours.
            
            Regards,
            Billing Department
            """,
            "file_type": "email"
        }
        
        # Test prompt building
        document_text = extractor._prepare_document_text(test_document)
        prompt = extractor._build_classification_prompt(document_text, enum_fields)
        
        print("✅ Classification prompt generated")
        print(f"✅ Prompt length: {len(prompt)} characters")
        
        # Test classification confidence calculation
        mock_result = {
            "document_type": "email",
            "priority_level": "urgent",
            "sentiment": "negative",
            "extracted_content": {
                "classification_evidence": {
                    "priority_indicators": ["URGENT", "immediate payment", "within 24 hours"],
                    "sentiment_indicators": ["overdue", "collection actions"]
                }
            }
        }
        
        confidence = extractor._calculate_classification_confidence(mock_result)
        print(f"✅ Classification confidence: {confidence:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Classification extractor test failed: {e}")
        return False

def test_few_shot_extractor():
    """Test few-shot extractor with dynamic ground truth loading."""
    print("\n🔧 Testing few-shot extractor...")
    
    try:
        from docuverse.extractors.few_shot import FewShotExtractor
        from docuverse.core.config import LLMConfig
        
        # Create LLM config (won't actually call LLM in test)
        llm_config = LLMConfig(
            provider="openai",
            model_name="gpt-4",
            api_key="test-key"
        )
        
        # Test with manual examples
        manual_examples = [
            {
                "input": "Contract payment terms: Net 30 days",
                "output": {
                    "payment_terms": {
                        "extracted_text": "Net 30 days",
                        "classification": "standard"
                    }
                }
            }
        ]
        
        extractor = FewShotExtractor(
            llm_config=llm_config,
            examples=manual_examples
        )
        
        print("✅ Few-shot extractor initialized with manual examples")
        
        # Test with data path
        data_path = Path(__file__).parent / "data" / "contracts"
        if data_path.exists():
            extractor_with_data = FewShotExtractor(
                llm_config=llm_config,
                data_path=data_path,
                max_examples=2
            )
            
            summary = extractor_with_data.get_example_summary()
            print(f"✅ Few-shot extractor loaded {summary['count']} examples from data path")
            print(f"   Document types: {summary['document_types']}")
            print(f"   Fields covered: {len(summary['fields_covered'])}")
        else:
            print("⚠️  No training data directory found, skipping data path test")
        
        # Test prompt building
        test_document = "Test contract with Net 45 payment terms"
        prompt = extractor._build_few_shot_prompt(test_document)
        
        print("✅ Few-shot prompt generation working")
        
        # Test confidence calculation
        test_result = {
            "payment_terms": {
                "extracted_text": "Test terms",
                "classification": "standard"
            },
            "document_type": "contract"
        }
        
        confidence = extractor._calculate_extraction_confidence(test_result)
        print(f"✅ Confidence calculation: {confidence:.2f}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Few-shot extractor import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Few-shot extractor test failed: {e}")
        return False

def test_training_data_manager():
    """Test training data management utilities."""
    print("\n🔧 Testing training data manager...")
    
    try:
        from docuverse.utils.training_data_manager import TrainingDataManager
        
        # Test with existing data directory
        data_path = Path(__file__).parent / "data" / "contracts"
        if data_path.exists():
            manager = TrainingDataManager(data_path)
            
            validation_report = manager.validate_training_data()
            print(f"✅ Validated {validation_report['valid_pairs']} training pairs")
            
            summary = manager.get_training_summary()
            print(f"✅ Training summary generated: {summary['total_examples']} examples")
            
        else:
            print("⚠️  No training data directory found, skipping detailed tests")
            # Still test basic initialization
            temp_path = Path(__file__).parent / "temp_test_data"
            manager = TrainingDataManager(temp_path)
            print("✅ Training data manager basic initialization working")
            
            # Clean up
            if temp_path.exists():
                import shutil
                shutil.rmtree(temp_path)
        
        return True
        
    except ImportError as e:
        print(f"❌ Training data manager import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Training data manager test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 DocuVerse Framework Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_configuration,
        test_document_loader,
        test_schema_validation,
        test_classification_extractor,
        test_few_shot_extractor,
        test_training_data_manager
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! DocuVerse framework is ready for research.")
        print("\nNext steps:")
        print("1. Set up LLM API keys (OpenAI/Anthropic)")
        print("2. Prepare your document datasets")
        print("3. Run the full example with: python example_usage.py")
        print("4. Start implementing your research experiments!")
        print("5. Try classification with: python classification_example.py")
    else:
        print("⚠️  Some tests failed. Please check the setup.")

if __name__ == "__main__":
    main()
