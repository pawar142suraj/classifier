"""
Example showing how to use vLLM and other open-source models for contract extraction.
Demonstrates setup and usage of different LLM providers.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    """Demonstrate LLM setup and usage with different providers."""
    
    print("🚀 DocuVerse: LLM Provider Setup and Usage")
    print("=" * 60)
    
    try:
        from src.docuverse.core.config import LLMConfig, LLMProvider
        from src.docuverse.extractors.few_shot import FewShotExtractor
        
        # Sample contract for testing
        test_contract = {
            "content": """
            SOFTWARE LICENSE AGREEMENT
            Contract Number: SLA-2024-TEST
            Date: August 16, 2025
            
            PAYMENT TERMS: Net 30 days from invoice date
            CONTRACT VALUE: $50,000 annually
            DELIVERY: Standard delivery within 7 business days
            
            This is a standard enterprise software license agreement.
            """,
            "file_type": "text",
            "metadata": {"source": "test_contract"}
        }
        
        print("📋 Available LLM Provider Configurations:")
        print("-" * 40)
        
        # 1. vLLM Configuration (for open-source models)
        print("\n🔓 1. vLLM (Open-Source Models)")
        print("   Perfect for: Llama, Mistral, CodeLlama, etc.")
        
        vllm_config = LLMConfig(
            provider=LLMProvider.VLLM,
            model_name="meta-llama/Llama-2-7b-chat-hf",
            vllm_server_url="http://localhost:8000",
            temperature=0.1,
            max_tokens=2048,
            top_p=0.9,
            top_k=50,
            vllm_gpu_memory_utilization=0.9
        )
        
        print(f"   Model: {vllm_config.model_name}")
        print(f"   Server: {vllm_config.vllm_server_url}")
        print(f"   GPU Memory: {vllm_config.vllm_gpu_memory_utilization}")
        
        # 2. Ollama Configuration
        print("\n🤖 2. Ollama (Local Models)")
        print("   Perfect for: CPU inference, easy setup")
        
        ollama_config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            ollama_base_url="http://localhost:11434",
            ollama_model="llama2",
            temperature=0.1,
            max_tokens=2048
        )
        
        print(f"   Model: {ollama_config.ollama_model}")
        print(f"   Server: {ollama_config.ollama_base_url}")
        
        # 3. HuggingFace Configuration
        print("\n🤗 3. HuggingFace Transformers")
        print("   Perfect for: Custom models, fine-tuned models")
        
        hf_config = LLMConfig(
            provider=LLMProvider.HUGGINGFACE,
            hf_model_id="microsoft/DialoGPT-medium",
            hf_device="auto",
            hf_quantization="4bit",
            temperature=0.1,
            max_tokens=512
        )
        
        print(f"   Model: {hf_config.hf_model_id}")
        print(f"   Device: {hf_config.hf_device}")
        print(f"   Quantization: {hf_config.hf_quantization}")
        
        # 4. OpenAI Configuration (for comparison)
        print("\n☁️ 4. OpenAI (Cloud)")
        print("   Perfect for: High quality, no setup required")
        
        openai_config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model_name="gpt-4-turbo-preview",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.1,
            max_tokens=4096
        )
        
        print(f"   Model: {openai_config.model_name}")
        print(f"   API Key: {'Set' if openai_config.api_key else 'Not set'}")
        
        # 5. Configuration with Fallback
        print("\n🔄 5. vLLM with OpenAI Fallback")
        print("   Perfect for: Reliability with cost control")
        
        fallback_config = LLMConfig(
            provider=LLMProvider.VLLM,
            model_name="meta-llama/Llama-2-7b-chat-hf",
            vllm_server_url="http://localhost:8000",
            
            # Fallback settings
            fallback_provider=LLMProvider.OPENAI,
            fallback_model="gpt-3.5-turbo",
            api_key=os.getenv("OPENAI_API_KEY"),
            
            max_retries=2,
            retry_delay=1.0
        )
        
        print(f"   Primary: {fallback_config.provider} ({fallback_config.model_name})")
        print(f"   Fallback: {fallback_config.fallback_provider} ({fallback_config.fallback_model})")
        
        # Demonstrate usage
        print("\n🔬 Testing Contract Extraction:")
        print("-" * 40)
        
        # Choose configuration to test (you can modify this)
        test_configs = [
            ("OpenAI GPT-4", openai_config),
            ("vLLM Llama2", vllm_config),
            ("Ollama", ollama_config),
        ]
        
        for config_name, config in test_configs:
            print(f"\n📄 Testing with {config_name}:")
            
            try:
                # Initialize extractor with training data
                data_path = Path(__file__).parent / "data" / "contracts"
                
                extractor = FewShotExtractor(
                    llm_config=config,
                    data_path=data_path if data_path.exists() else None,
                    max_examples=2
                )
                
                # Check if we have training data
                summary = extractor.get_example_summary()
                print(f"   Training examples: {summary['count']}")
                
                # Test extraction (only if API key is available or it's a local model)
                can_test = (
                    config.provider in [LLMProvider.VLLM, LLMProvider.OLLAMA, LLMProvider.HUGGINGFACE] or
                    (config.provider == LLMProvider.OPENAI and config.api_key)
                )
                
                if can_test:
                    print("   Attempting extraction...")
                    # Note: This will only work if the respective servers are running
                    # or API keys are configured
                    try:
                        result = extractor.extract(test_contract)
                        
                        print("   ✅ Extraction successful!")
                        
                        # Show key results
                        if "payment_terms" in result:
                            pt = result["payment_terms"]
                            if isinstance(pt, dict):
                                print(f"      Payment Terms: {pt.get('classification', 'N/A')}")
                                print(f"      Extracted: '{pt.get('extracted_text', 'N/A')}'")
                            else:
                                print(f"      Payment Terms: {pt}")
                        
                        if "contract_value" in result:
                            cv = result["contract_value"]
                            if isinstance(cv, dict):
                                print(f"      Contract Value: {cv.get('classification', 'N/A')}")
                                print(f"      Amount: ${cv.get('amount', 'N/A')}")
                            else:
                                print(f"      Contract Value: {cv}")
                        
                        # Show confidence
                        print(f"      Confidence: {extractor.last_confidence:.2f}")
                        
                    except Exception as e:
                        print(f"   ⚠️  Extraction failed: {e}")
                        print(f"      (This is expected if server isn't running)")
                        
                else:
                    print("   ⚠️  Skipped - requires API key or local server")
                    
            except Exception as e:
                print(f"   ❌ Setup failed: {e}")
        
        # Show setup instructions
        print(f"\n🛠️  Setup Instructions:")
        print("=" * 40)
        
        print("\n📦 1. Install Dependencies:")
        print("   pip install vllm  # For vLLM")
        print("   pip install transformers torch  # For HuggingFace")
        print("   # Download Ollama from https://ollama.com/")
        
        print("\n🚀 2. Start vLLM Server:")
        print("   python -m vllm.entrypoints.openai.api_server \\")
        print("       --model meta-llama/Llama-2-7b-chat-hf \\")
        print("       --host 0.0.0.0 \\")
        print("       --port 8000")
        
        print("\n🤖 3. Start Ollama:")
        print("   ollama pull llama2")
        print("   ollama serve")
        
        print("\n🔑 4. Set API Keys (optional):")
        print("   export OPENAI_API_KEY='your-key'")
        print("   export ANTHROPIC_API_KEY='your-key'")
        
        print("\n🎯 Popular Open-Source Models:")
        print("   • meta-llama/Llama-2-7b-chat-hf")
        print("   • mistralai/Mistral-7B-Instruct-v0.2")
        print("   • codellama/CodeLlama-7b-Instruct-hf")
        print("   • HuggingFaceH4/zephyr-7b-beta")
        print("   • openchat/openchat-3.5-1210")
        
        print("\n💡 Tips:")
        print("   • Start with 7B models for faster inference")
        print("   • Use 4bit quantization to reduce memory usage")
        print("   • Monitor GPU usage with nvidia-smi")
        print("   • Set up fallbacks for reliability")
        
        print(f"\n✨ Complete! You can now use open-source models with DocuVerse.")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you've set up the environment correctly.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
