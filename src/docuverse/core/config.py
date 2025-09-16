"""
Core configuration classes for DocuVerse extraction methods.
"""

from typing import List, Dict, Any, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field, validator


class ExtractionMethod(str, Enum):
    """Available extraction methods."""
    FEW_SHOT = "few_shot"
    VECTOR_RAG = "vector_rag"
    GRAPH_RAG = "graph_rag"
    REASONING_COT = "reasoning_cot"
    REASONING_REACT = "reasoning_react"
    DYNAMIC_GRAPH_RAG = "dynamic_graph_rag"
    CLASSIFICATION = "classification"


class EvaluationMetric(str, Enum):
    """Available evaluation metrics."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    ROUGE_L = "rouge_l"
    BERT_SCORE = "bert_score"
    EXTRACTION_TIME = "extraction_time"
    TOKEN_USAGE = "token_usage"


class ChunkingStrategy(str, Enum):
    """Document chunking strategies."""
    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    SLIDING_WINDOW = "sliding_window"
    HIERARCHICAL = "hierarchical"


class VectorRAGConfig(BaseModel):
    """Configuration for Vector RAG method."""
    chunk_size: int = Field(default=512, description="Size of document chunks")
    chunk_overlap: int = Field(default=50, description="Overlap between chunks")
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    retrieval_k: int = Field(default=5, description="Number of chunks to retrieve")
    rerank_top_k: int = Field(default=3, description="Number of chunks after reranking")
    use_hybrid_search: bool = Field(default=True, description="Use BM25 + semantic search")
    bm25_weight: float = Field(default=0.3, description="Weight for BM25 in hybrid search")


class GraphRAGConfig(BaseModel):
    """Configuration for Graph RAG method."""
    entity_extraction_model: str = Field(default="en_core_web_sm")
    relation_extraction_model: Optional[str] = None
    graph_db_uri: Optional[str] = Field(default="bolt://localhost:7687")
    graph_db_user: str = Field(default="neo4j", description="Neo4j database username")
    graph_db_password: str = Field(default="Password", description="Neo4j database password")
    max_subgraph_size: int = Field(default=50, description="Maximum nodes in subgraph")
    entity_similarity_threshold: float = Field(default=0.8)
    use_lightweight_kg: bool = Field(default=True)


class ReasoningConfig(BaseModel):
    """Configuration for reasoning-enhanced methods."""
    use_cot: bool = Field(default=True, description="Use Chain of Thought reasoning")
    use_react: bool = Field(default=False, description="Use ReAct reasoning")
    max_reasoning_steps: int = Field(default=5)
    verification_enabled: bool = Field(default=True)
    auto_repair_enabled: bool = Field(default=True)
    uncertainty_threshold: float = Field(default=0.7)


class DynamicGraphRAGConfig(BaseModel):
    """Configuration for novel Dynamic Graph RAG method."""
    base_graph_config: GraphRAGConfig = Field(default_factory=GraphRAGConfig)
    expansion_strategy: str = Field(default="uncertainty_based")
    max_expansion_depth: int = Field(default=3)
    uncertainty_threshold: float = Field(default=0.6)
    fallback_to_cypher: bool = Field(default=True)
    adaptive_chunk_size: bool = Field(default=True)


class LLMProvider(str, Enum):
    """Available LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    VLLM = "vllm"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    AZURE_OPENAI = "azure_openai"
    LOCAL = "local"


class LLMConfig(BaseModel):
    """Enhanced LLM configuration supporting multiple providers including vLLM."""
    
    # Core settings
    provider: LLMProvider = Field(default=LLMProvider.OPENAI, description="LLM provider")
    model_name: str = Field(default="gpt-4-turbo-preview", description="Model name/identifier")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Generation temperature")
    max_tokens: int = Field(default=4096, description="Maximum tokens to generate")
    timeout: int = Field(default=60, description="Request timeout in seconds")
    
    # Authentication
    api_key: Optional[str] = Field(default=None, description="API key for cloud providers")
    api_base: Optional[str] = Field(default=None, description="Custom API base URL")
    organization: Optional[str] = Field(default=None, description="Organization ID (OpenAI)")
    
    # vLLM specific settings
    vllm_server_url: Optional[str] = Field(default="http://localhost:8000", description="vLLM server URL")
    vllm_model_path: Optional[str] = Field(default=None, description="Path to local model for vLLM")
    vllm_gpu_memory_utilization: float = Field(default=0.9, ge=0.1, le=1.0, description="GPU memory utilization")
    vllm_max_model_len: Optional[int] = Field(default=None, description="Maximum model sequence length")
    vllm_tensor_parallel_size: int = Field(default=1, description="Number of GPUs for tensor parallelism")
    
    # Ollama specific settings
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama server URL")
    ollama_model: str = Field(default="llama2", description="Ollama model name")
    
    # HuggingFace specific settings
    hf_model_id: Optional[str] = Field(default=None, description="HuggingFace model ID")
    hf_token: Optional[str] = Field(default=None, description="HuggingFace API token")
    hf_device: str = Field(default="auto", description="Device for HuggingFace models")
    hf_quantization: Optional[str] = Field(default=None, description="Quantization method (4bit, 8bit)")
    
    # Azure OpenAI specific settings
    azure_endpoint: Optional[str] = Field(default=None, description="Azure OpenAI endpoint")
    azure_deployment: Optional[str] = Field(default=None, description="Azure deployment name")
    azure_api_version: str = Field(default="2024-02-15-preview", description="Azure API version")
    
    # Advanced generation settings
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Top-p (nucleus) sampling")
    top_k: int = Field(default=50, ge=1, description="Top-k sampling")
    repetition_penalty: float = Field(default=1.0, ge=0.0, description="Repetition penalty")
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Presence penalty")
    
    # Error handling and reliability
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay: float = Field(default=1.0, description="Delay between retries in seconds")
    fallback_provider: Optional[LLMProvider] = Field(default=None, description="Fallback provider if primary fails")
    fallback_model: Optional[str] = Field(default=None, description="Fallback model name")
    
    # Performance and cost control
    request_cache: bool = Field(default=True, description="Enable request caching")
    batch_size: int = Field(default=1, description="Batch size for batch processing")
    rate_limit_rpm: Optional[int] = Field(default=None, description="Rate limit (requests per minute)")
    cost_tracking: bool = Field(default=True, description="Track token usage and costs")
    
    @validator('vllm_server_url')
    def validate_vllm_url(cls, v, values):
        """Validate vLLM server URL format."""
        if values.get('provider') == LLMProvider.VLLM and v:
            if not v.startswith(('http://', 'https://')):
                raise ValueError("vLLM server URL must start with http:// or https://")
        return v
    
    @validator('model_name')
    def validate_model_name(cls, v, values):
        """Provide model suggestions based on provider."""
        provider = values.get('provider')
        if provider == LLMProvider.VLLM and not v:
            raise ValueError("model_name is required for vLLM provider")
        return v
    
    def get_provider_config(self) -> Dict[str, Any]:
        """Get provider-specific configuration."""
        base_config = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout
        }
        
        if self.provider == LLMProvider.OPENAI:
            return {
                **base_config,
                "api_key": self.api_key,
                "organization": self.organization,
                "api_base": self.api_base
            }
        
        elif self.provider == LLMProvider.ANTHROPIC:
            return {
                **base_config,
                "api_key": self.api_key
            }
        
        elif self.provider == LLMProvider.VLLM:
            return {
                **base_config,
                "server_url": self.vllm_server_url,
                "model_path": self.vllm_model_path,
                "gpu_memory_utilization": self.vllm_gpu_memory_utilization,
                "max_model_len": self.vllm_max_model_len,
                "tensor_parallel_size": self.vllm_tensor_parallel_size
            }
        
        elif self.provider == LLMProvider.OLLAMA:
            return {
                **base_config,
                "base_url": self.ollama_base_url,
                "model": self.ollama_model
            }
        
        elif self.provider == LLMProvider.HUGGINGFACE:
            return {
                **base_config,
                "model_id": self.hf_model_id,
                "token": self.hf_token,
                "device": self.hf_device,
                "quantization": self.hf_quantization
            }
        
        elif self.provider == LLMProvider.AZURE_OPENAI:
            return {
                **base_config,
                "api_key": self.api_key,
                "azure_endpoint": self.azure_endpoint,
                "azure_deployment": self.azure_deployment,
                "api_version": self.azure_api_version
            }
        
        return base_config


class ExtractionConfig(BaseModel):
    """Main configuration class for document extraction."""
    
    # Methods to evaluate
    methods: List[ExtractionMethod] = Field(
        default=[ExtractionMethod.FEW_SHOT],
        description="Extraction methods to run and compare"
    )
    
    # Schema and validation
    schema_path: Optional[str] = Field(
        default=None,
        description="Path to JSON schema for output validation"
    )
    
    # Evaluation configuration
    evaluation_metrics: List[EvaluationMetric] = Field(
        default=[EvaluationMetric.ACCURACY, EvaluationMetric.F1],
        description="Metrics to compute during evaluation"
    )
    
    # LLM configuration
    llm_config: LLMConfig = Field(default_factory=LLMConfig)
    
    # Method-specific configurations
    vector_rag_config: VectorRAGConfig = Field(default_factory=VectorRAGConfig)
    graph_rag_config: GraphRAGConfig = Field(default_factory=GraphRAGConfig)
    reasoning_config: ReasoningConfig = Field(default_factory=ReasoningConfig)
    dynamic_graph_rag_config: DynamicGraphRAGConfig = Field(default_factory=DynamicGraphRAGConfig)
    
    # General extraction settings
    few_shot_examples: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Few-shot examples for baseline method"
    )
    
    max_retries: int = Field(default=3, description="Maximum retries for failed extractions")
    parallel_processing: bool = Field(default=True, description="Enable parallel processing")
    max_workers: int = Field(default=4, description="Maximum parallel workers")
    
    # Output settings
    output_format: str = Field(default="json", description="Output format (json, yaml)")
    include_metadata: bool = Field(default=True, description="Include extraction metadata")
    save_intermediate_results: bool = Field(default=False, description="Save intermediate processing steps")
    
    @validator('methods')
    def validate_methods(cls, v):
        """Ensure at least one method is specified."""
        if not v:
            raise ValueError("At least one extraction method must be specified")
        return v
    
    @validator('max_workers')
    def validate_max_workers(cls, v, values):
        """Ensure max_workers is reasonable when parallel processing is enabled."""
        if values.get('parallel_processing', True) and v < 1:
            raise ValueError("max_workers must be at least 1 when parallel processing is enabled")
        return v
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True
