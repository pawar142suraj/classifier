#!/usr/bin/env python3
"""
Test the unified FewShotExtractor with real data from the data folder.
"""

import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from docuverse.core.config import LLMConfig
from docuverse.extractors.few_shot import FewShotExtractor


def test_with_real_data():
    """Test the extractor with the actual contract data."""
    
    print("🧪 Testing Unified Extractor with Real Data")
    print("=" * 50)
    
    # Load schema
    schema_path = Path(__file__).parent.parent.parent / "schemas" / "contracts_schema_hybrid.json"
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    
    # Simple LLM config for testing (you can adjust this)
    llm_config = LLMConfig(
        provider="ollama",
        model_name="llama3.2:latest",
        ollama_base_url="http://localhost:11434",
        temperature=0.1
    )
    
    # Initialize extractor
    extractor = FewShotExtractor(
        llm_config=llm_config,
        schema=schema,
        auto_load_labels=True
    )
    
    print(f"✅ Loaded {len(extractor.examples)} examples from data/labels")
    
    # Test with the actual contract document
    contract_path = Path(__file__).parent.parent.parent / "data" / "contract1.txt"
    
    if contract_path.exists():
        with open(contract_path, 'r') as f:
            contract_content = f.read()
        
        print(f"📄 Testing with: {contract_path.name}")
        print(f"Document length: {len(contract_content)} characters")
        
        # Create document object
        document = {
            "content": contract_content,
            "metadata": {
                "filename": contract_path.name,
                "source": "test_data"
            }
        }
        
        # Extract information
        print("\n🔄 Performing extraction...")
        try:
            result = extractor.extract(document)
            confidence = extractor.last_confidence
            
            print(f"✅ Extraction completed with confidence: {confidence:.2f}")
            print("\n📊 Results:")
            print(json.dumps(result, indent=2))
            
            # Validate
            validation = extractor.validate_schema_compliance(result)
            if validation["is_valid"]:
                print("\n✅ Validation: PASSED")
            else:
                print("\n⚠️ Validation: FAILED")
                for error in validation["missing_fields"] + validation["invalid_enums"] + validation["structure_errors"]:
                    print(f"  - {error}")
        
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
    
    else:
        print(f"❌ Contract file not found: {contract_path}")
    
    # Show example analysis
    print("\n📈 Example Analysis:")
    summary = extractor.get_example_summary()
    for source in summary['example_sources']:
        print(f"  • {source['base_name']}")
    
    print(f"\n📋 Field Coverage:")
    for field, coverage in summary['field_coverage'].items():
        print(f"  • {field}: {coverage['percentage']:.1f}%")


if __name__ == "__main__":
    test_with_real_data()
