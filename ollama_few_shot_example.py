"""
Example showing how to use Ollama with DocuVerse few-shot extraction.
This demonstrates contract extraction using the local Ollama model.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    """Demonstrate Ollama integration with few-shot extraction."""
    
    print("🤖 DocuVerse + Ollama: Few-Shot Contract Extraction")
    print("=" * 60)
    
    try:
        from docuverse.core.config import LLMConfig, LLMProvider
        from docuverse.extractors.few_shot import FewShotExtractor
        
        # Sample contract for testing
        test_contract = {
            "content": """
            PROFESSIONAL SERVICES AGREEMENT
            Contract ID: PSA-2024-OLLAMA
            Date: August 19, 2025
            
            PAYMENT TERMS: Net 15 days from invoice date
            CONTRACT VALUE: $125,000 over 12 months
            DELIVERY: Weekly deliverables starting September 1st
            TERMINATION: Either party may terminate with 30 days notice
            
            This agreement covers AI consulting services for document processing.
            The client will provide all necessary data and infrastructure access.
            """,
            "file_type": "text",
            "metadata": {"source": "ollama_test_contract"}
        }
        
        print("📋 Test Contract Loaded:")
        print(f"   Contract ID: PSA-2024-OLLAMA")
        print(f"   Value: $125,000")
        print(f"   Terms: Net 15 days")
        
        # Configure Ollama
        print("\n🔧 Configuring Ollama LLM...")
        
        ollama_config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            ollama_base_url="http://localhost:11434",
            ollama_model="llama3.2:3b",  # Using available model
            temperature=0.1,
            max_tokens=1024
        )
        
        print(f"   Provider: {ollama_config.provider}")
        print(f"   Model: {ollama_config.ollama_model}")
        print(f"   Server: {ollama_config.ollama_base_url}")
        print(f"   Temperature: {ollama_config.temperature}")
        
        # Initialize extractor with training data
        print("\n📚 Initializing Few-Shot Extractor...")
        
        data_path = Path(__file__).parent / "data" / "contracts"
        
        extractor = FewShotExtractor(
            llm_config=ollama_config,
            data_path=data_path if data_path.exists() else None,
            max_examples=2  # Use 2 examples for few-shot learning
        )
        
        # Check training data
        summary = extractor.get_example_summary()
        print(f"   Training examples loaded: {summary['count']}")
        if summary['count'] > 0 and 'fields' in summary:
            print(f"   Example fields: {', '.join(summary['fields'])}")
        
        # Test Ollama connection first
        print("\n🔍 Testing Ollama connection...")
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m.get('name', 'unknown') for m in models]
                print(f"   ✅ Ollama server is running")
                print(f"   Available models: {', '.join(model_names) if model_names else 'None'}")
                
                if 'llama3.2:3b' not in model_names:
                    print("   ⚠️  Warning: llama3.2:3b model not found")
                    print("   💡 Run: ollama pull llama3.2:3b")
                    
            else:
                print(f"   ❌ Ollama server error: {response.status_code}")
                return
                
        except Exception as e:
            print(f"   ❌ Cannot connect to Ollama: {e}")
            print("   💡 Make sure Ollama is running: ollama serve")
            return
        
        # Perform extraction
        print(f"\n🚀 Extracting information with Ollama...")
        print("   (This may take 10-30 seconds depending on your hardware)")
        
        try:
            result = extractor.extract(test_contract)
            
            print("\n✅ Extraction Complete!")
            print("=" * 40)
            
            # Display results in a formatted way
            if isinstance(result, dict):
                for field, value in result.items():
                    if isinstance(value, dict):
                        print(f"\n📄 {field.replace('_', ' ').title()}:")
                        for sub_key, sub_value in value.items():
                            print(f"   {sub_key}: {sub_value}")
                    else:
                        print(f"\n📄 {field.replace('_', ' ').title()}: {value}")
            else:
                print(f"📄 Result: {result}")
            
            # Show confidence and performance
            print(f"\n📊 Extraction Statistics:")
            print(f"   Confidence: {extractor.last_confidence:.2f}")
            print(f"   Fields extracted: {len(result) if isinstance(result, dict) else 1}")
            
            # Suggestions for improvement
            print(f"\n💡 Performance Tips:")
            print(f"   • Add more training examples in data/contracts/")
            print(f"   • Try different Ollama models: ollama pull mistral")
            print(f"   • Adjust temperature for more/less creative responses")
            print(f"   • Use larger models for better accuracy")
            
        except Exception as e:
            print(f"\n❌ Extraction failed: {e}")
            print("\n🔧 Troubleshooting:")
            print("   1. Check if Ollama server is running: ollama serve")
            print("   2. Verify llama2 model is installed: ollama pull llama2")
            print("   3. Check server logs for errors")
            print("   4. Try with a simpler prompt or smaller document")
        
        # Show available Ollama models
        print(f"\n🔧 Ollama Model Options:")
        print("   • llama2 (7B, good general purpose)")
        print("   • mistral (7B, fast and accurate)")
        print("   • codellama (7B, good for structured data)")
        print("   • llama2:13b (13B, better accuracy, slower)")
        print("   • neural-chat (7B, optimized for conversations)")
        
        print("\n🎯 To switch models:")
        print("   1. ollama pull <model-name>")
        print("   2. Update ollama_model in the config")
        print("   3. Restart your script")
        
        print(f"\n✨ Ollama + DocuVerse integration complete!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you've run the setup script first:")
        print("python setup_vllm.py")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
