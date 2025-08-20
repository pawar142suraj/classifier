# LLM Setup Guide for DocuVerse

This guide shows how to set up various LLM providers, including vLLM for open-source models.

## 📋 Quick Setup

### 1. Install Dependencies

```bash
# Basic requirements
pip install requests

# For OpenAI
pip install openai

# For Anthropic
pip install anthropic

# For vLLM (GPU required)
pip install vllm

# For Ollama (local models)
# Download from https://ollama.com/

# For HuggingFace
pip install transformers torch accelerate bitsandbytes
```

### 2. Provider-Specific Setup

## 🔓 Open-Source Models with vLLM

vLLM is excellent for running open-source models like Llama, Mistral, CodeLlama, etc., with high performance.

### Setup vLLM Server

```bash
# Install vLLM
pip install vllm

# Start vLLM server with a model (example: Llama-2-7B-Chat)
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.9
```

### Popular Open-Source Models for vLLM

```python
# Some popular models you can use:
models = {
    "llama2-7b": "meta-llama/Llama-2-7b-chat-hf",
    "llama2-13b": "meta-llama/Llama-2-13b-chat-hf",
    "codellama": "codellama/CodeLlama-7b-Instruct-hf",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "zephyr": "HuggingFaceH4/zephyr-7b-beta",
    "openchat": "openchat/openchat-3.5-1210",
    "vicuna": "lmsys/vicuna-7b-v1.5",
    # Note: You may need to request access for some models
}
```

### Configure DocuVerse for vLLM

```python
from docuverse.core.config import LLMConfig, LLMProvider

# vLLM configuration
llm_config = LLMConfig(
    provider=LLMProvider.VLLM,
    model_name="meta-llama/Llama-2-7b-chat-hf",  # Must match server model
    vllm_server_url="http://localhost:8000",
    temperature=0.1,
    max_tokens=2048,
    top_p=0.9,
    top_k=50
)
```

## 🤖 Ollama (Local Models)

Ollama is great for running models locally without GPU requirements.

### Setup Ollama

```bash
# Download and install from https://ollama.com/
# Then pull a model
ollama pull llama2
ollama pull mistral
ollama pull codellama
```

### Configure DocuVerse for Ollama

```python
llm_config = LLMConfig(
    provider=LLMProvider.OLLAMA,
    ollama_base_url="http://localhost:11434",
    ollama_model="llama2",  # or "mistral", "codellama", etc.
    temperature=0.1,
    max_tokens=2048
)
```

## 🤗 HuggingFace Transformers

Run models directly with transformers (good for fine-tuned models).

```python
llm_config = LLMConfig(
    provider=LLMProvider.HUGGINGFACE,
    hf_model_id="microsoft/DialoGPT-medium",
    hf_device="auto",  # or "cuda", "cpu"
    hf_quantization="4bit",  # Optional: "4bit" or "8bit"
    temperature=0.1,
    max_tokens=512
)
```

## ☁️ Cloud Providers

### OpenAI

```python
llm_config = LLMConfig(
    provider=LLMProvider.OPENAI,
    model_name="gpt-4-turbo-preview",
    api_key="your-openai-api-key",
    temperature=0.1,
    max_tokens=4096
)
```

### Anthropic Claude

```python
llm_config = LLMConfig(
    provider=LLMProvider.ANTHROPIC,
    model_name="claude-3-sonnet-20240229",
    api_key="your-anthropic-api-key",
    temperature=0.1,
    max_tokens=4096
)
```

### Azure OpenAI

```python
llm_config = LLMConfig(
    provider=LLMProvider.AZURE_OPENAI,
    model_name="gpt-4",
    api_key="your-azure-key",
    azure_endpoint="https://your-resource.openai.azure.com/",
    azure_deployment="your-deployment-name",
    azure_api_version="2024-02-15-preview"
)
```

## 🔧 Advanced Configuration

### With Fallback

```python
llm_config = LLMConfig(
    provider=LLMProvider.VLLM,
    model_name="meta-llama/Llama-2-7b-chat-hf",
    vllm_server_url="http://localhost:8000",
    
    # Fallback to OpenAI if vLLM fails
    fallback_provider=LLMProvider.OPENAI,
    fallback_model="gpt-3.5-turbo",
    api_key="your-openai-key",  # For fallback
    
    max_retries=3,
    retry_delay=1.0
)
```

### Performance Tuning

```python
llm_config = LLMConfig(
    provider=LLMProvider.VLLM,
    model_name="meta-llama/Llama-2-7b-chat-hf",
    
    # vLLM performance settings
    vllm_gpu_memory_utilization=0.9,
    vllm_tensor_parallel_size=2,  # Use 2 GPUs
    vllm_max_model_len=4096,
    
    # Generation settings
    temperature=0.1,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.1,
    
    # Rate limiting
    rate_limit_rpm=60,
    cost_tracking=True
)
```

## 📊 Usage Examples

### Complete Example with vLLM

```python
from docuverse.extractors.few_shot import FewShotExtractor
from docuverse.core.config import LLMConfig, LLMProvider

# Configure vLLM
llm_config = LLMConfig(
    provider=LLMProvider.VLLM,
    model_name="meta-llama/Llama-2-7b-chat-hf",
    vllm_server_url="http://localhost:8000",
    temperature=0.1,
    max_tokens=2048
)

# Initialize extractor
extractor = FewShotExtractor(
    llm_config=llm_config,
    data_path="data/contracts",
    max_examples=3
)

# Extract from document
document = {
    "content": "Your contract text here...",
    "file_type": "text"
}

result = extractor.extract(document)
print(f"Extracted: {result}")

# Get usage statistics
stats = extractor._get_llm_client().get_combined_stats()
print(f"Tokens used: {stats['totals']['tokens']}")
```

### Multi-Provider Comparison

```python
providers = [
    {
        "name": "vLLM Llama2",
        "config": LLMConfig(
            provider=LLMProvider.VLLM,
            model_name="meta-llama/Llama-2-7b-chat-hf",
            vllm_server_url="http://localhost:8000"
        )
    },
    {
        "name": "Ollama Mistral",
        "config": LLMConfig(
            provider=LLMProvider.OLLAMA,
            ollama_model="mistral"
        )
    },
    {
        "name": "OpenAI GPT-4",
        "config": LLMConfig(
            provider=LLMProvider.OPENAI,
            model_name="gpt-4-turbo-preview",
            api_key="your-key"
        )
    }
]

# Compare extraction quality across providers
for provider in providers:
    extractor = FewShotExtractor(
        llm_config=provider["config"],
        data_path="data/contracts"
    )
    
    result = extractor.extract(document)
    print(f"{provider['name']}: {result}")
```

## 🚀 Getting Started with vLLM

1. **Install vLLM**: `pip install vllm`

2. **Start server**:
   ```bash
   python -m vllm.entrypoints.openai.api_server \
       --model meta-llama/Llama-2-7b-chat-hf \
       --port 8000
   ```

3. **Configure DocuVerse**:
   ```python
   llm_config = LLMConfig(
       provider=LLMProvider.VLLM,
       model_name="meta-llama/Llama-2-7b-chat-hf",
       vllm_server_url="http://localhost:8000"
   )
   ```

4. **Use in extraction**:
   ```python
   extractor = FewShotExtractor(llm_config=llm_config, data_path="data/contracts")
   result = extractor.extract(document)
   ```

## 🔍 Troubleshooting

### vLLM Issues
- **GPU Memory**: Reduce `vllm_gpu_memory_utilization`
- **Model Not Found**: Check HuggingFace model ID
- **Connection Error**: Verify server is running on correct port

### Ollama Issues
- **Model Not Found**: Run `ollama pull <model-name>`
- **Server Not Running**: Start with `ollama serve`

### HuggingFace Issues
- **CUDA OOM**: Try smaller model or quantization
- **Access Denied**: Get HuggingFace token for gated models

## 💡 Tips

1. **Start with smaller models** (7B) before trying larger ones
2. **Use quantization** (4bit/8bit) to reduce memory usage
3. **Monitor GPU usage** with `nvidia-smi`
4. **Try multiple models** to find best quality/speed balance
5. **Use fallback providers** for reliability

This setup allows you to run powerful open-source models locally while maintaining compatibility with cloud providers!
