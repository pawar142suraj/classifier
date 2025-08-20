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
    print("\n📦 Installing vLLM...")
    
    try:
        # Install basic dependencies first
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "requests", "torch", "transformers", "accelerate"
        ], check=True)
        print("✅ Basic dependencies installed")
        
        # Install vLLM (might take longer)
        print("📦 Installing vLLM (this may take a few minutes)...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "vllm"
        ], check=True)
        print("✅ vLLM installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install vLLM: {e}")
        print("💡 Try installing manually:")
        print("   pip install requests torch transformers accelerate")
        print("   pip install vllm")
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

def main():
    """Main setup function."""
    print("🚀 DocuVerse vLLM Setup")
    print("=" * 30)
    
    if not check_requirements():
        print("\n❌ Requirements not met. Please fix issues and try again.")
        return
    
    print("\n📌 This script will:")
    print("   • Install vLLM and dependencies")
    print("   • Download a 3B parameter model (~6GB)")
    print("   • Create configuration and start scripts")
    print("   • Set up testing utilities")
    
    choice = input("\n🤔 Continue? (y/N): ").lower().strip()
    if choice not in ['y', 'yes']:
        print("Setup cancelled.")
        return
    
    # Install vLLM
    if not install_vllm():
        print("❌ Setup failed during installation.")
        return
    
    # Create configuration
    config = create_vllm_config()
    
    # Create start scripts
    create_start_script(config)
    
    # Create test script
    create_test_script()
    
    # Show next steps
    show_next_steps(config)

if __name__ == "__main__":
    main()
