# 🚀 LLM Setup Guide for DocuVerse

This guide shows you how to set up and use different LLM providers with DocuVerse, including **vLLM for open-source models**.

## 🎯 Quick Start

### Option 1: Automated Setup (Recommended)
```bash
# Run the automated setup script
python setup_vllm.py

# Follow the prompts to install and configure vLLM
# This will download a 7B model (~13GB) and set up everything
```

### Option 2: Manual Setup
See the detailed instructions below for each provider.

## 🔓 Open-Source Models with vLLM

vLLM is perfect for running models like the recently open-sourced GPT variants, Llama, Mistral, and more.

### 1. Install vLLM
```bash
# Install vLLM and dependencies
pip install vllm torch transformers accelerate

# For CUDA support (recommended)
pip install vllm[cuda]
```

### 2. Start vLLM Server
```bash
# Basic setup with Llama 2 7B
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --host 0.0.0.0 \
    --port 8000

# Advanced setup with optimizations
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096 \
    --tensor-parallel-size 1
```

### 3. Configure DocuVerse
```python
from src.docuverse.core.config import LLMConfig, LLMProvider

# vLLM configuration
config = LLMConfig(
    provider=LLMProvider.VLLM,
    model_name="meta-llama/Llama-2-7b-chat-hf",
    vllm_server_url="http://localhost:8000",
    temperature=0.1,
    max_tokens=2048,
    vllm_gpu_memory_utilization=0.9
)
```

### 4. Test the Setup
```python
# Run the comprehensive test
python llm_setup_example.py

# Or run a simple test
python test_vllm.py
```

## 🤖 Other Local Providers

### Ollama (Easy CPU/GPU inference)
```bash
# Install Ollama (download from https://ollama.com/)
# Or use the installer

# Pull a model
ollama pull llama2

# Start Ollama (usually auto-starts)
ollama serve
```

```python
# Configure Ollama
config = LLMConfig(
    provider=LLMProvider.OLLAMA,
    ollama_base_url="http://localhost:11434",
    ollama_model="llama2",
    temperature=0.1
)
```

### HuggingFace Transformers (Direct model loading)
```python
config = LLMConfig(
    provider=LLMProvider.HUGGINGFACE,
    hf_model_id="microsoft/DialoGPT-medium",
    hf_device="auto",
    hf_quantization="4bit",  # Reduces memory usage
    temperature=0.1
)
```

## ☁️ Cloud Providers

### OpenAI
```python
config = LLMConfig(
    provider=LLMProvider.OPENAI,
    model_name="gpt-4-turbo-preview",
    api_key="your-openai-key",
    temperature=0.1
)
```

### Anthropic Claude
```python
config = LLMConfig(
    provider=LLMProvider.ANTHROPIC,
    model_name="claude-3-sonnet-20240229",
    api_key="your-anthropic-key",
    temperature=0.1
)
```

### Azure OpenAI
```python
config = LLMConfig(
    provider=LLMProvider.AZURE_OPENAI,
    model_name="gpt-4",
    azure_endpoint="https://your-resource.openai.azure.com/",
    api_key="your-azure-key",
    azure_deployment="your-deployment-name"
)
```

## 🎯 Popular Open-Source Models

| Model | Size | Best For | vLLM Command |
|-------|------|----------|--------------|
| **Llama 2 Chat** | 7B | General purpose | `--model meta-llama/Llama-2-7b-chat-hf` |
| **Mistral Instruct** | 7B | Fast inference | `--model mistralai/Mistral-7B-Instruct-v0.2` |
| **Code Llama** | 7B | Code tasks | `--model codellama/CodeLlama-7b-Instruct-hf` |
| **Zephyr** | 7B | High quality | `--model HuggingFaceH4/zephyr-7b-beta` |
| **OpenChat** | 7B | Conversations | `--model openchat/openchat-3.5-1210` |

## 🔄 Fallback Configuration

Configure automatic fallbacks for reliability:

```python
config = LLMConfig(
    # Primary provider (local)
    provider=LLMProvider.VLLM,
    model_name="meta-llama/Llama-2-7b-chat-hf",
    vllm_server_url="http://localhost:8000",
    
    # Fallback provider (cloud)
    fallback_provider=LLMProvider.OPENAI,
    fallback_model="gpt-3.5-turbo",
    api_key="your-openai-key",
    
    # Retry settings
    max_retries=2,
    retry_delay=1.0
)
```

## 📊 Performance Comparison

### Typical Inference Speeds (tokens/second):
- **vLLM + GPU**: 50-100 tokens/sec
- **Ollama + GPU**: 20-50 tokens/sec
- **HuggingFace + GPU**: 10-30 tokens/sec
- **OpenAI API**: 20-40 tokens/sec
- **CPU inference**: 1-5 tokens/sec

### Memory Requirements:
- **7B model**: 14-16GB GPU memory
- **7B with 4-bit quantization**: 4-6GB GPU memory
- **CPU inference**: 8-16GB RAM

## 🚀 Example Usage

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from docuverse.core.config import LLMConfig, LLMProvider
from docuverse.extractors.few_shot import FewShotExtractor

# Document to extract from
document = {
    "content": '''
    SERVICE AGREEMENT
    Contract ID: SA-2024-001
    Payment Terms: Net 30 days
    Total Value: $75,000 annually
    ''',
    "file_type": "text"
}

# Configure vLLM
config = LLMConfig(
    provider=LLMProvider.VLLM,
    model_name="meta-llama/Llama-2-7b-chat-hf",
    vllm_server_url="http://localhost:8000",
    temperature=0.1
)

# Initialize extractor with training data
extractor = FewShotExtractor(
    llm_config=config,
    data_path="data/contracts",  # Your *.labels.json files
    max_examples=3
)

# Extract information
result = extractor.extract(document)
print("Extracted data:", result)
```

## 🛠️ Troubleshooting

### Common Issues:

**1. Out of Memory (OOM) Errors**
```bash
# Reduce GPU memory usage
--gpu-memory-utilization 0.7

# Use smaller model
--model meta-llama/Llama-2-7b-chat-hf  # instead of 13B

# Enable CPU offloading
--device cpu
```

**2. Model Download Issues**
```bash
# Login to HuggingFace
huggingface-cli login

# Set cache directory
export HF_HOME=/path/to/large/storage
```

**3. Connection Issues**
```bash
# Check server status
curl http://localhost:8000/v1/models

# Check server logs
python -m vllm.entrypoints.openai.api_server --model ... --log-level DEBUG
```

**4. Slow Inference**
```bash
# Enable tensor parallelism (multi-GPU)
--tensor-parallel-size 2

# Use Flash Attention
pip install flash-attn

# Check GPU utilization
nvidia-smi
```

## 🎯 Production Tips

1. **Model Selection**: Start with 7B models, upgrade to 13B+ if needed
2. **Quantization**: Use 4-bit for memory savings with minimal quality loss
3. **Caching**: Enable KV-cache for faster repeated inference
4. **Monitoring**: Track GPU memory, inference speed, and error rates
5. **Fallbacks**: Always configure cloud fallbacks for reliability

## 📁 File Structure

After setup, you'll have:
```
docuverse/
├── setup_vllm.py          # Automated setup script
├── llm_setup_example.py   # Comprehensive usage examples
├── test_vllm.py          # Simple test script
├── start_vllm.sh         # Server start script (Linux/Mac)
├── start_vllm.bat        # Server start script (Windows)
├── vllm_config.json      # vLLM configuration
└── src/docuverse/
    ├── core/
    │   └── config.py       # Enhanced LLM configuration
    └── utils/
        └── llm_client.py   # Multi-provider client system
```

## 🔗 Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [HuggingFace Models](https://huggingface.co/models)
- [Ollama Models](https://ollama.com/library)
- [DocuVerse Research Guide](RESEARCH_GUIDE.md)

---

🎉 **Ready to go!** Run `python setup_vllm.py` to get started with open-source models in minutes.
