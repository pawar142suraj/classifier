#!/usr/bin/env python3
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
