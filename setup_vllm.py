#!/usr/bin/env python3
"""
Quick setup script for vLLM with DocuVerse.
This script helps you get started with open-source models quickly.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_requirements():
    """Check if basic requirements are met."""
    print("🔍 Checking requirements...")
    
    requirements = {
        "python": (3, 8),
        "gpu": None,  # Will check nvidia-smi
        "disk_space": 10,  # GB minimum
    }
    
    # Check Python version
    python_version = sys.version_info
    if python_version < requirements["python"]:
        print(f"❌ Python {requirements['python'][0]}.{requirements['python'][1]}+ required")
        return False
    print(f"✅ Python {python_version.major}.{python_version.minor}")
    
    # Check GPU (optional but recommended)
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected")
            requirements["gpu"] = True
        else:
            print("⚠️  No NVIDIA GPU detected (CPU inference will be slower)")
            requirements["gpu"] = False
    except FileNotFoundError:
        print("⚠️  nvidia-smi not found (CPU inference will be slower)")
        requirements["gpu"] = False
    
    # Check disk space
    try:
        if os.name == 'nt':  # Windows
            import shutil
            free_gb = shutil.disk_usage('.').free / (1024**3)
        else:  # Unix-like systems
            statvfs = os.statvfs('.')
            free_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
        
        if free_gb < requirements["disk_space"]:
            print(f"⚠️  Low disk space: {free_gb:.1f}GB (need {requirements['disk_space']}GB+)")
        else:
            print(f"✅ Disk space: {free_gb:.1f}GB available")
    except Exception as e:
        print(f"⚠️  Could not check disk space: {e}")
    
    return True

def install_vllm():
    """Install vLLM and dependencies."""
    print("\n📦 Installing dependencies...")
    
    try:
        # Install basic dependencies first
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "requests", "torch", "transformers", "accelerate", "openai"
        ], check=True)
        print("✅ Basic dependencies installed")
        
        # Check if we're on Windows
        if os.name == 'nt':
            print("\n⚠️  Windows detected - vLLM installation can be complex")
            print("   vLLM has limited Windows support and requires specific build tools")
            
            choice = input("\n🤔 Try installing vLLM anyway? (y/N): ").lower().strip()
            if choice not in ['y', 'yes']:
                print("\n💡 Alternative: Use Ollama or HuggingFace Transformers instead")
                print("   • Ollama: Download from https://ollama.com/")
                print("   • HuggingFace: Already installed with transformers")
                return "skip_vllm"
        
        # Try to install vLLM
        print("📦 Installing vLLM (this may take several minutes)...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "vllm"
            ], check=True, timeout=600)  # 10 minute timeout
            print("✅ vLLM installed successfully")
            return True
        except subprocess.TimeoutExpired:
            print("⏰ vLLM installation timed out")
            return "timeout"
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install vLLM: {e}")
            
            if os.name == 'nt':
                print("\n💡 Windows vLLM Installation Alternatives:")
                print("   1. Use WSL2 (Windows Subsystem for Linux)")
                print("   2. Use Docker with vLLM image")
                print("   3. Use Ollama (easier on Windows)")
                print("   4. Use HuggingFace Transformers directly")
                return "windows_alternatives"
            else:
                print("💡 Try installing manually:")
                print("   pip install vllm")
                return False
                
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install basic dependencies: {e}")
        return False

def download_model(model_name="microsoft/DialoGPT-large"):
    """Download a model using HuggingFace."""
    print(f"\n⬇️  Downloading model: {model_name}")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print("📥 Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print("📥 Downloading model (this may take a while)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        
        print(f"✅ Model {model_name} downloaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        print("💡 You may need to log in to HuggingFace:")
        print("   huggingface-cli login")
        return False

def create_vllm_config():
    """Create a vLLM configuration file."""
    print("\n📝 Creating vLLM configuration...")
    
    config = {
        "model": "microsoft/DialoGPT-large",
        "host": "0.0.0.0",
        "port": 8000,
        "gpu_memory_utilization": 0.7,
        "max_model_len": 2048,
        "tensor_parallel_size": 1,
        "disable_log_stats": False
    }
    
    config_path = Path("vllm_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Configuration saved to {config_path}")
    return config

def create_ollama_config():
    """Create an Ollama configuration as alternative to vLLM."""
    print("\n📝 Creating Ollama configuration (vLLM alternative)...")
    
    config = {
        "provider": "ollama",
        "model": "llama2",
        "base_url": "http://localhost:11434",
        "temperature": 0.1,
        "max_tokens": 2048
    }
    
    config_path = Path("ollama_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Ollama configuration saved to {config_path}")
    return config

def create_huggingface_config():
    """Create a HuggingFace configuration as alternative to vLLM."""
    print("\n📝 Creating HuggingFace configuration (vLLM alternative)...")
    
    config = {
        "provider": "huggingface",
        "model_id": "microsoft/DialoGPT-large",
        "device": "auto",
        "quantization": "4bit",
        "temperature": 0.1,
        "max_tokens": 512
    }
    
    config_path = Path("huggingface_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ HuggingFace configuration saved to {config_path}")
    return config

def create_start_script(config):
    """Create a script to start vLLM server."""
    print("\n📜 Creating start script...")
    
    script_content = f"""#!/bin/bash
# Start vLLM server
echo "🚀 Starting vLLM server..."

python -m vllm.entrypoints.openai.api_server \\
    --model {config['model']} \\
    --host {config['host']} \\
    --port {config['port']} \\
    --gpu-memory-utilization {config['gpu_memory_utilization']} \\
    --max-model-len {config['max_model_len']} \\
    --tensor-parallel-size {config['tensor_parallel_size']}

echo "Server stopped"
"""
    
    # Windows batch script
    bat_content = f"""@echo off
echo 🚀 Starting vLLM server...

python -m vllm.entrypoints.openai.api_server ^
    --model {config['model']} ^
    --host {config['host']} ^
    --port {config['port']} ^
    --gpu-memory-utilization {config['gpu_memory_utilization']} ^
    --max-model-len {config['max_model_len']} ^
    --tensor-parallel-size {config['tensor_parallel_size']}

echo Server stopped
pause
"""
    
    # Create both scripts
    with open("start_vllm.sh", "w") as f:
        f.write(script_content)
    
    with open("start_vllm.bat", "w") as f:
        f.write(bat_content)
    
    # Make shell script executable
    try:
        os.chmod("start_vllm.sh", 0o755)
    except:
        pass
    
    print("✅ Start scripts created:")
    print("   • start_vllm.sh (Linux/Mac)")
    print("   • start_vllm.bat (Windows)")

def create_test_script():
    """Create a test script for the vLLM setup."""
    print("\n🧪 Creating test script...")
    
    test_content = '''#!/usr/bin/env python3
"""Test script for vLLM setup."""

import requests
import json

def test_vllm_server():
    """Test if vLLM server is running and responding."""
    print("🧪 Testing vLLM server...")
    
    url = "http://localhost:8000/v1/chat/completions"
    
    payload = {
        "model": "microsoft/DialoGPT-large",
        "messages": [
            {
                "role": "user",
                "content": "Extract the contract value from this text: The total contract value is $50,000 per year."
            }
        ],
        "max_tokens": 100,
        "temperature": 0.1
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        
        print("✅ vLLM server is working!")
        print(f"Response: {result['choices'][0]['message']['content']}")
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to vLLM server")
        print("💡 Make sure the server is running: ./start_vllm.sh")
        return False
        
    except Exception as e:
        print(f"❌ Error testing server: {e}")
        return False

if __name__ == "__main__":
    test_vllm_server()
'''
    
    with open("test_vllm.py", "w") as f:
        f.write(test_content)
    
    print("✅ Test script created: test_vllm.py")

def show_next_steps(config):
    """Show next steps to the user."""
    print("\n🎉 Setup Complete!")
    print("=" * 50)
    
    print("\n📋 Next Steps:")
    print("1. Start the vLLM server:")
    if os.name == 'nt':  # Windows
        print("   start_vllm.bat")
    else:  # Linux/Mac
        print("   ./start_vllm.sh")
    
    print("\n2. Wait for the server to load (may take a few minutes)")
    
    print("\n3. Test the connection:")
    print("   python test_vllm.py")
    
    print("\n4. Use with DocuVerse:")
    print("   python llm_setup_example.py")
    
    print(f"\n🌐 Server Details:")
    print(f"   URL: http://localhost:{config['port']}")
    print(f"   Model: {config['model']}")
    print(f"   API Format: OpenAI Compatible")
    
    print("\n💡 Troubleshooting:")
    print("   • Check GPU memory: nvidia-smi")
    print("   • Monitor server logs for errors")
    print("   • Try smaller models if OOM errors occur")
    print("   • Use CPU fallback: --device cpu")
    
    print("\n🔧 Alternative Models:")
    print("   • mistralai/Mistral-7B-Instruct-v0.2 (fast)")
    print("   • microsoft/DialoGPT-large (smaller)")
    print("   • codellama/CodeLlama-7b-Instruct-hf (code)")

def create_ollama_scripts():
    """Create scripts to download and run Ollama models."""
    print("\n📜 Creating Ollama scripts...")
    
    # Ollama setup script for Windows
    setup_content = '''@echo off
echo 🤖 Ollama Setup for DocuVerse
echo ================================

echo 📥 Downloading and installing Ollama models...

REM Pull the model (this will download ~4GB)
ollama pull llama2

echo ✅ Ollama setup complete!
echo.
echo 🚀 Starting Ollama server...
ollama serve
'''
    
    # Ollama test script
    test_content = '''#!/usr/bin/env python3
"""Test script for Ollama setup."""

import requests
import json

def test_ollama_server():
    """Test if Ollama server is running and responding."""
    print("🧪 Testing Ollama server...")
    
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": "llama2",
        "prompt": "Extract the contract value from this text: The total contract value is $50,000 per year.",
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        
        print("✅ Ollama server is working!")
        print(f"Response: {result.get('response', 'No response')}")
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama server")
        print("💡 Make sure Ollama is running: ollama serve")
        return False
        
    except Exception as e:
        print(f"❌ Error testing server: {e}")
        return False

if __name__ == "__main__":
    test_ollama_server()
'''
    
    # Create setup script
    with open("setup_ollama.bat", "w", encoding='utf-8') as f:
        f.write(setup_content)
    
    # Create test script
    with open("test_ollama.py", "w", encoding='utf-8') as f:
        f.write(test_content)
    
    print("✅ Ollama scripts created:")
    print("   • setup_ollama.bat (Windows setup)")
    print("   • test_ollama.py (Test connection)")

def create_huggingface_scripts():
    """Create scripts for HuggingFace setup."""
    print("\n📜 Creating HuggingFace scripts...")
    
    test_content = '''#!/usr/bin/env python3
"""Test script for HuggingFace setup."""

def test_huggingface():
    """Test HuggingFace model loading."""
    print("🧪 Testing HuggingFace model loading...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        model_name = "microsoft/DialoGPT-large"
        
        print(f"📥 Loading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print(f"📥 Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        
        # Test generation
        test_prompt = "Extract the contract value: The total contract value is $50,000 per year."
        inputs = tokenizer.encode(test_prompt, return_tensors="pt")
        
        print("🔮 Generating response...")
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + 50,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ HuggingFace model is working!")
        print(f"Response: {response}")
        return True
        
    except Exception as e:
        print(f"❌ Error testing HuggingFace: {e}")
        return False

if __name__ == "__main__":
    test_huggingface()
'''
    
    with open("test_huggingface.py", "w", encoding='utf-8') as f:
        f.write(test_content)
    
    print("✅ HuggingFace test script created: test_huggingface.py")

def show_ollama_steps():
    """Show Ollama-specific next steps."""
    print("\n🎉 Ollama Setup Complete!")
    print("=" * 50)
    
    print("\n📋 Next Steps:")
    print("1. Install Ollama:")
    print("   • Download from: https://ollama.com/")
    print("   • Or run: winget install Ollama.Ollama")
    
    print("\n2. Run the setup script:")
    print("   setup_ollama.bat")
    
    print("\n3. Test the connection:")
    print("   python test_ollama.py")
    
    print("\n4. Use with DocuVerse:")
    print("   python llm_setup_example.py")
    
    print(f"\n🌐 Ollama Details:")
    print(f"   URL: http://localhost:11434")
    print(f"   Model: llama2")
    print(f"   API Format: Ollama REST API")
    
    print("\n💡 Ollama Commands:")
    print("   • List models: ollama list")
    print("   • Pull model: ollama pull llama2")
    print("   • Start server: ollama serve")
    print("   • Chat: ollama run llama2")

def show_huggingface_steps():
    """Show HuggingFace-specific next steps."""
    print("\n🎉 HuggingFace Setup Complete!")
    print("=" * 50)
    
    print("\n📋 Next Steps:")
    print("1. Test the model loading:")
    print("   python test_huggingface.py")
    
    print("\n2. Use with DocuVerse:")
    print("   python llm_setup_example.py")
    
    print(f"\n🤗 HuggingFace Details:")
    print(f"   Model: microsoft/DialoGPT-large")
    print(f"   Mode: Direct model loading")
    print(f"   Device: Auto (GPU if available)")

def show_multiple_provider_steps():
    """Show steps for multiple providers."""
    print("\n🎉 Multi-Provider Setup Complete!")
    print("=" * 50)
    
    print("\n📋 You now have multiple LLM options:")
    print("1. 🤖 Ollama (for easy local inference)")
    print("   • Install: https://ollama.com/")
    print("   • Setup: setup_ollama.bat")
    print("   • Test: python test_ollama.py")
    
    print("\n2. 🤗 HuggingFace (for direct model access)")
    print("   • Test: python test_huggingface.py")
    
    print("\n3. 🚀 Use with DocuVerse:")
    print("   python llm_setup_example.py")
    
    print("\n💡 Choose the provider that works best for your setup!")

def main():
    """Main setup function."""
    print("🚀 DocuVerse LLM Setup")
    print("=" * 30)
    
    if not check_requirements():
        print("\n❌ Requirements not met. Please fix issues and try again.")
        return
    
    print("\n📌 This script will set up local LLM providers:")
    print("   • Install basic dependencies (requests, transformers, etc.)")
    print("   • Configure Ollama for easy local inference")
    print("   • Set up HuggingFace as backup option")
    print("   • Create test scripts and configuration files")
    print("   • Skip vLLM (complex Windows installation)")
    
    choice = input("\n🤔 Continue with Ollama + HuggingFace setup? (Y/n): ").lower().strip()
    if choice in ['n', 'no']:
        print("Setup cancelled.")
        return
    
    # Install basic dependencies only
    print("\n📦 Installing basic dependencies...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "requests", "torch", "transformers", "accelerate", "openai"
        ], check=True)
        print("✅ Basic dependencies installed")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return
    
    # Set up Ollama and HuggingFace configurations
    print("\n🔄 Setting up local LLM providers...")
    
    print("\n🤖 Creating Ollama configuration (Recommended)...")
    ollama_config = create_ollama_config()
    create_ollama_scripts()
    
    print("\n🤗 Creating HuggingFace configuration (Backup)...")
    hf_config = create_huggingface_config()
    create_huggingface_scripts()
    
    # Create a unified test script
    create_unified_test_script()
    
    show_multiple_provider_steps()

def create_unified_test_script():
    """Create a unified test script for all providers."""
    print("\n🧪 Creating unified test script...")
    
    test_content = '''#!/usr/bin/env python3
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
    print("\\n🤖 Testing Ollama...")
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
    print("\\n🤗 Testing HuggingFace...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
        print("✅ HuggingFace models accessible")
        results["huggingface"] = True
    except Exception as e:
        print(f"❌ HuggingFace not available: {e}")
        results["huggingface"] = False
    
    # Test DocuVerse integration
    print("\\n📚 Testing DocuVerse integration...")
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
    print("\\n� Summary:")
    print("=" * 20)
    for provider, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {provider.title()}: {'Ready' if status else 'Not available'}")
    
    if any(results.values()):
        print("\\n🎉 At least one provider is working! You're ready to use DocuVerse.")
    else:
        print("\\n⚠️  No providers available. Check installation steps.")

if __name__ == "__main__":
    test_all_providers()
'''
    
    with open("test_all_providers.py", "w", encoding='utf-8') as f:
        f.write(test_content)
    
    print("✅ Unified test script created: test_all_providers.py")

if __name__ == "__main__":
    main()
