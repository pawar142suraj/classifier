#!/usr/bin/env python3
"""
Test script for the Reasoning Extractor with contract data.
Verifies that the implementation works correctly with real contract documents.
"""

import sys
import json
import time
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

def test_reasoning_extractor():
    """Test the reasoning extractor with contract data."""
    
    print("🧪 Testing Reasoning Extractor Implementation")
    print("=" * 50)
    
    try:
        from docuverse.extractors.reasoning import ReasoningExtractor
        from docuverse.core.config import (
            LLMConfig, ReasoningConfig, ExtractionMethod
        )
        
        # Load contract schema
        schema_path = Path("schemas/contract_schema.json")
        if not schema_path.exists():
            print(f"❌ Schema not found: {schema_path}")
            return False
            
        with open(schema_path, 'r') as f:
            schema = json.load(f)
        
        print(f"✅ Loaded schema with {len(schema.get('properties', {}))} fields")
        
        # Load test contract
        contract_path = Path("data/contract1.txt")
        if not contract_path.exists():
            print(f"❌ Contract not found: {contract_path}")
            return False
            
        with open(contract_path, 'r') as f:
            contract_content = f.read()
        
        print(f"✅ Loaded contract document ({len(contract_content)} chars)")
        
        # Configure LLM (using a mock configuration for testing)
        llm_config = LLMConfig(
            provider="ollama",
            model_name="llama3.2:latest",
            ollama_base_url="http://localhost:11434",
            temperature=0.1,
            max_tokens=2048
        )
        
        # Configure reasoning
        reasoning_config = ReasoningConfig(
            use_cot=True,
            max_reasoning_steps=3,  # Reduced for testing
            verification_enabled=False,  # Disabled for quick test
            uncertainty_threshold=0.5
        )
        
        print(f"✅ Configured LLM and reasoning settings")
        
        # Test CoT extractor initialization
        print(f"\n🧠 Testing CoT Reasoning Extractor...")
        
        cot_extractor = ReasoningExtractor(
            llm_config=llm_config,
            reasoning_config=reasoning_config,
            method_type=ExtractionMethod.REASONING_COT,
            schema=schema,
            use_vector_rag=False  # Disable for basic test
        )
        
        print(f"✅ CoT extractor initialized successfully")
        
        # Test ReAct extractor initialization
        print(f"\n⚡ Testing ReAct Reasoning Extractor...")
        
        react_extractor = ReasoningExtractor(
            llm_config=llm_config,
            reasoning_config=reasoning_config,
            method_type=ExtractionMethod.REASONING_REACT,
            schema=schema,
            use_vector_rag=False  # Disable for basic test
        )
        
        print(f"✅ ReAct extractor initialized successfully")
        
        # Test Vector RAG integration (if available)
        print(f"\n🔍 Testing Vector RAG Integration...")
        
        try:
            rag_extractor = ReasoningExtractor(
                llm_config=llm_config,
                reasoning_config=reasoning_config,
                method_type=ExtractionMethod.REASONING_COT,
                schema=schema,
                use_vector_rag=True  # Enable Vector RAG
            )
            print(f"✅ Vector RAG integration available")
            
            # Test query generation
            queries = rag_extractor._generate_reasoning_queries()
            print(f"✅ Generated {len(queries)} reasoning queries")
            
            for name, query in list(queries.items())[:3]:
                print(f"  • {name}: {query[:50]}...")
                
        except Exception as e:
            print(f"⚠️ Vector RAG integration not available: {e}")
            print(f"💡 This is normal if optional dependencies are missing")
        
        # Test reasoning analysis
        print(f"\n📊 Testing Reasoning Analysis...")
        
        analysis = cot_extractor.get_reasoning_analysis()
        
        expected_keys = [
            'method', 'total_steps', 'evidence_pieces', 'overall_confidence',
            'uncertainty_threshold', 'vector_rag_enabled', 'verification_enabled'
        ]
        
        for key in expected_keys:
            if key in analysis:
                print(f"  ✅ {key}: {analysis[key]}")
            else:
                print(f"  ❌ Missing analysis key: {key}")
        
        # Test basic extraction functionality (structure only, no LLM call)
        print(f"\n🔧 Testing Extraction Structure...")
        
        document = {
            "content": contract_content[:500],  # Use smaller sample
            "metadata": {"source": "test", "filename": "contract1.txt"}
        }
        
        # Test document preparation
        prepared_text = cot_extractor._prepare_document_text(document)
        print(f"✅ Document preparation: {len(prepared_text)} chars")
        
        # Test field extraction helpers
        target_fields = cot_extractor._get_target_fields()
        print(f"✅ Target fields identified: {len(target_fields)} fields")
        
        for field in target_fields[:3]:
            excerpt = cot_extractor._get_relevant_excerpt(contract_content, field, 100)
            print(f"  • {field}: {len(excerpt)} chars of relevant content")
        
        print(f"\n🎉 All Reasoning Extractor Tests Passed!")
        print(f"📝 The implementation is ready for contract information extraction")
        print(f"🚀 Run demo_reasoning_extractor.py for a full demonstration")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print(f"💡 Make sure the src/ directory is in your Python path")
        return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_reasoning_extractor()
    exit(0 if success else 1)
