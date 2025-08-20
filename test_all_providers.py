#!/usr/bin/env python3
"""Unified test script for all LLM providers."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_all_providers():
    """Test all available LLM providers."""
    print("🧪 Testing all LLM providers...")
    print("=" * 40)
    
    results = {}
    
    # Test Ollama
    print("\n🤖 Testing Ollama...")
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama server is running")
            results["ollama"] = True
        else:
            print("⚠️  Ollama server not responding")
            results["ollama"] = False
    except Exception as e:
        print(f"❌ Ollama not available: {e}")
        results["ollama"] = False
    
    # Test HuggingFace
    print("\n🤗 Testing HuggingFace...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
        print("✅ HuggingFace models accessible")
        results["huggingface"] = True
    except Exception as e:
        print(f"❌ HuggingFace not available: {e}")
        results["huggingface"] = False
    
    # Test DocuVerse integration
    print("\n📚 Testing DocuVerse integration...")
    try:
        from docuverse.core.config import LLMConfig, LLMProvider
        from docuverse.extractors.few_shot import FewShotExtractor
        
        # Test with Ollama config
        if results["ollama"]:
            config = LLMConfig(
                provider=LLMProvider.OLLAMA,
                ollama_base_url="http://localhost:11434",
                ollama_model="llama2",
                temperature=0.1
            )
            
            extractor = FewShotExtractor(llm_config=config)
            print("✅ DocuVerse + Ollama integration ready")
        
        elif results["huggingface"]:
            config = LLMConfig(
                provider=LLMProvider.HUGGINGFACE,
                hf_model_id="microsoft/DialoGPT-large",
                temperature=0.1
            )
            
            extractor = FewShotExtractor(llm_config=config)
            print("✅ DocuVerse + HuggingFace integration ready")
        
        else:
            print("⚠️  No local providers available")
            
    except Exception as e:
        print(f"❌ DocuVerse integration error: {e}")
    
    # Summary
    print("\n� Summary:")
    print("=" * 20)
    for provider, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {provider.title()}: {'Ready' if status else 'Not available'}")
    
    if any(results.values()):
        print("\n🎉 At least one provider is working! You're ready to use DocuVerse.")
    else:
        print("\n⚠️  No providers available. Check installation steps.")

if __name__ == "__main__":
    test_all_providers()
