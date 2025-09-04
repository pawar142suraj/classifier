#!/usr/bin/env python3
"""
Test script for Azure OpenAI integration with the reasoning extractor.
"""

import os
from src.docuverse.core.config import LLMConfig, LLMProvider

def test_azure_openai_config():
    """Test Azure OpenAI configuration."""
    
    # Example Azure OpenAI configuration
    azure_config = LLMConfig(
        provider=LLMProvider.AZURE_OPENAI,
        model_name="gpt-4",  # Your deployed model name
        temperature=0.1,
        max_tokens=2000,
        timeout=300,
        
        # Azure-specific settings (replace with your values)
        api_key="your-azure-openai-key",
        azure_endpoint="https://your-resource.openai.azure.com/",
        azure_deployment="your-deployment-name",
        azure_api_version="2024-02-15-preview"
    )
    
    print("Azure OpenAI Configuration:")
    print(f"Provider: {azure_config.provider}")
    print(f"Model: {azure_config.model_name}")
    print(f"Endpoint: {azure_config.azure_endpoint}")
    print(f"Deployment: {azure_config.azure_deployment}")
    
    # Test provider-specific config generation
    provider_config = azure_config.get_provider_config()
    print("\nProvider-specific config:")
    for key, value in provider_config.items():
        if 'key' in key.lower():
            print(f"{key}: {'*' * len(str(value)) if value else None}")
        else:
            print(f"{key}: {value}")
    
    return azure_config

def test_azure_client_creation():
    """Test creating Azure OpenAI client through factory."""
    from src.docuverse.utils.llm_client import LLMClientFactory
    
    config = test_azure_openai_config()
    
    try:
        # Create Azure OpenAI client
        client = LLMClientFactory.create_client(config)
        print(f"\n✅ Successfully created Azure OpenAI client: {type(client).__name__}")
        
        # Test a simple generation (requires valid credentials)
        if config.api_key != "your-azure-openai-key":  # Only test if real credentials
            try:
                response = client.generate("Hello, how are you?")
                print(f"✅ Test generation successful: {response[:100]}...")
            except Exception as e:
                print(f"⚠️  Generation test failed (likely auth): {e}")
        else:
            print("ℹ️  Skipping generation test - update credentials in script")
        
        return client
        
    except Exception as e:
        print(f"❌ Failed to create Azure OpenAI client: {e}")
        return None

def demonstrate_azure_reasoning_usage():
    """Show how to use Azure OpenAI with reasoning extractor."""
    
    print("\n" + "="*60)
    print("Azure OpenAI Reasoning Extractor Usage Example")
    print("="*60)
    
    example_code = '''
from src.docuverse.extractors.reasoning import ReasoningExtractor
from src.docuverse.core.config import LLMConfig, LLMProvider

# Configure Azure OpenAI
llm_config = LLMConfig(
    provider=LLMProvider.AZURE_OPENAI,
    model_name="gpt-4",  # Your deployment name
    temperature=0.1,
    max_tokens=2000,
    timeout=300,
    api_key="your-azure-openai-key",
    azure_endpoint="https://your-resource.openai.azure.com/",
    azure_deployment="your-deployment-name",
    azure_api_version="2024-02-15-preview"
)

# Initialize reasoning extractor with Azure OpenAI
extractor = ReasoningExtractor(
    llm_config=llm_config,
    schema_path="schemas/contracts_schema_hybrid.json"
)

# Extract from document using Chain of Thought reasoning
result = extractor.extract_cot(
    text="Your contract text here...",
    output_path="output/azure_extraction"
)

# Extract using ReAct reasoning
result = extractor.extract_react(
    text="Your contract text here...",
    output_path="output/azure_extraction"
)
'''
    
    print(example_code)
    print("\n📝 Make sure to:")
    print("1. Replace placeholder values with your actual Azure OpenAI credentials")
    print("2. Set your correct deployment name as model_name")
    print("3. Use your Azure OpenAI resource endpoint")
    print("4. Ensure your deployment has sufficient quota for reasoning tasks")

if __name__ == "__main__":
    print("🧪 Testing Azure OpenAI Integration")
    print("="*50)
    
    # Test configuration
    config = test_azure_openai_config()
    
    # Test client creation
    client = test_azure_client_creation()
    
    # Show usage examples
    demonstrate_azure_reasoning_usage()
    
    print("\n✅ Azure OpenAI integration test completed!")
    print("\n💡 Next steps:")
    print("1. Update credentials in this script or demo_reasoning_extractor.py")
    print("2. Test with actual Azure OpenAI deployment")
    print("3. Run reasoning extraction with Azure OpenAI backend")
