"""
LLM client supporting multiple providers including vLLM for open-source models.
"""

import json
import time
import logging
import requests
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod

from ..core.config import LLMConfig, LLMProvider

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Base class for LLM clients."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.request_count = 0
        self.total_tokens = 0
        self.total_cost = 0.0
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text from prompt."""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "requests": self.request_count,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "provider": self.config.provider
        }


class OpenAIClient(BaseLLMClient):
    """OpenAI API client."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import openai
            self.client = openai.OpenAI(
                api_key=config.api_key,
                organization=config.organization,
                base_url=config.api_base
            )
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using OpenAI API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty
            )
            
            self.request_count += 1
            if response.usage:
                self.total_tokens += response.usage.total_tokens
                # Rough cost calculation (update with current pricing)
                self.total_cost += self._calculate_openai_cost(response.usage.total_tokens)
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def _calculate_openai_cost(self, tokens: int) -> float:
        """Rough cost calculation for OpenAI models."""
        # These are approximate rates - update with current pricing
        cost_per_1k_tokens = {
            "gpt-4": 0.03,
            "gpt-4-turbo": 0.01,
            "gpt-3.5-turbo": 0.002
        }
        
        model_key = "gpt-4" if "gpt-4" in self.config.model_name else "gpt-3.5-turbo"
        rate = cost_per_1k_tokens.get(model_key, 0.01)
        return (tokens / 1000) * rate


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude API client."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=config.api_key)
        except ImportError:
            raise ImportError("Anthropic package not installed. Run: pip install anthropic")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using Anthropic API."""
        try:
            message = self.client.messages.create(
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=system_prompt or "",
                messages=[{"role": "user", "content": prompt}]
            )
            
            self.request_count += 1
            if hasattr(message, 'usage'):
                self.total_tokens += message.usage.input_tokens + message.usage.output_tokens
            
            return message.content[0].text
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise


class vLLMClient(BaseLLMClient):
    """vLLM server client for open-source models."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.server_url = config.vllm_server_url.rstrip('/')
        self._check_server_health()
    
    def _check_server_health(self):
        """Check if vLLM server is running."""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.status_code != 200:
                raise ConnectionError(f"vLLM server not healthy: {response.status_code}")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Cannot connect to vLLM server at {self.server_url}: {e}")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using vLLM server."""
        # Combine system and user prompts
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
        
        payload = {
            "prompt": full_prompt,
            "model": self.config.model_name,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "repetition_penalty": self.config.repetition_penalty,
            "stop": ["User:", "Human:"],  # Stop sequences
            "stream": False
        }
        
        try:
            response = requests.post(
                f"{self.server_url}/v1/completions",
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            self.request_count += 1
            
            if "usage" in result:
                self.total_tokens += result["usage"].get("total_tokens", 0)
            
            return result["choices"][0]["text"].strip()
            
        except Exception as e:
            logger.error(f"vLLM API error: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        try:
            response = requests.get(f"{self.server_url}/v1/models")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {}


class OllamaClient(BaseLLMClient):
    """Ollama client for local models."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = config.ollama_base_url.rstrip('/')
        self._check_server_health()
    
    def _check_server_health(self):
        """Check if Ollama server is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                raise ConnectionError(f"Ollama server not available: {response.status_code}")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Cannot connect to Ollama server at {self.base_url}: {e}")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using Ollama."""
        payload = {
            "model": self.config.model_name,  # Use model_name instead of ollama_model
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "repeat_penalty": self.config.repetition_penalty
            }
        }
        
        # Add system prompt if provided
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            self.request_count += 1
            
            return result.get("response", "")
            
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise


class HuggingFaceClient(BaseLLMClient):
    """HuggingFace Transformers client."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.hf_model_id,
                token=config.hf_token
            )
            
            # Load model with appropriate device and quantization
            model_kwargs = {}
            if config.hf_quantization == "4bit":
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            elif config.hf_quantization == "8bit":
                model_kwargs["load_in_8bit"] = True
            
            self.model = AutoModelForCausalLM.from_pretrained(
                config.hf_model_id,
                token=config.hf_token,
                device_map=config.hf_device,
                **model_kwargs
            )
            
        except ImportError:
            raise ImportError("Transformers package not installed. Run: pip install transformers torch")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using HuggingFace model."""
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        try:
            inputs = self.tokenizer.encode(full_prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    repetition_penalty=self.config.repetition_penalty,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode only the new tokens
            new_tokens = outputs[0][len(inputs[0]):]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            self.request_count += 1
            self.total_tokens += len(outputs[0])
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"HuggingFace generation error: {e}")
            raise


class LLMClientFactory:
    """Factory for creating LLM clients based on provider."""
    
    @staticmethod
    def create_client(config: LLMConfig) -> BaseLLMClient:
        """Create appropriate LLM client based on configuration."""
        
        if config.provider == LLMProvider.OPENAI:
            return OpenAIClient(config)
        
        elif config.provider == LLMProvider.ANTHROPIC:
            return AnthropicClient(config)
        
        elif config.provider == LLMProvider.VLLM:
            return vLLMClient(config)
        
        elif config.provider == LLMProvider.OLLAMA:
            return OllamaClient(config)
        
        elif config.provider == LLMProvider.HUGGINGFACE:
            return HuggingFaceClient(config)
        
        else:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")


class LLMManager:
    """High-level LLM manager with error handling and fallbacks."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.primary_client = LLMClientFactory.create_client(config)
        self.fallback_client = None
        
        # Create fallback client if configured
        if config.fallback_provider and config.fallback_model:
            fallback_config = config.copy()
            fallback_config.provider = config.fallback_provider
            fallback_config.model_name = config.fallback_model
            self.fallback_client = LLMClientFactory.create_client(fallback_config)
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text with automatic fallback on errors."""
        
        # Try primary client
        for attempt in range(self.config.max_retries):
            try:
                response = self.primary_client.generate(prompt, system_prompt)
                if response.strip():
                    return response
                
            except Exception as e:
                logger.warning(f"Primary LLM attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    logger.error("All primary LLM attempts failed")
        
        # Try fallback client if available
        if self.fallback_client:
            logger.info("Attempting fallback LLM")
            try:
                return self.fallback_client.generate(prompt, system_prompt)
            except Exception as e:
                logger.error(f"Fallback LLM also failed: {e}")
        
        raise RuntimeError("All LLM generation attempts failed")
    
    def get_combined_stats(self) -> Dict[str, Any]:
        """Get combined statistics from all clients."""
        stats = {
            "primary": self.primary_client.get_stats()
        }
        
        if self.fallback_client:
            stats["fallback"] = self.fallback_client.get_stats()
        
        # Calculate totals
        total_requests = stats["primary"]["requests"]
        total_tokens = stats["primary"]["total_tokens"]
        total_cost = stats["primary"]["total_cost"]
        
        if self.fallback_client:
            total_requests += stats["fallback"]["requests"]
            total_tokens += stats["fallback"]["total_tokens"]
            total_cost += stats["fallback"]["total_cost"]
        
        stats["totals"] = {
            "requests": total_requests,
            "tokens": total_tokens,
            "cost": total_cost
        }
        
        return stats
