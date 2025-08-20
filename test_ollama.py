#!/usr/bin/env python3
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
