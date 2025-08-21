"""
Evaluation Configuration Module

This module provides configuration helpers for setting up unified evaluations
across all DocuVerse extraction methods.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from unified_evaluation_framework import (
    EvaluationConfig, 
    ExtractorConfig, 
    TestCase, 
    create_test_cases_from_data_folder
)

import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from docuverse.core.config import (
    LLMConfig, 
    EvaluationMetric, 
    ExtractionMethod,
    LLMProvider,
    ExtractionConfig
)


class EvaluationConfigBuilder:
    """Builder class for creating evaluation configurations."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset builder to default state."""
        self._llm_config = None
        self._metrics = []
        self._test_cases = []
        self._extractors = []
        self._output_dir = Path("tests/reports")
        self._timeout = 300.0
        self._detailed_logging = True
        return self
    
    def with_llm_config(self, **kwargs) -> 'EvaluationConfigBuilder':
        """Set LLM configuration."""
        self._llm_config = LLMConfig(**kwargs)
        return self
    
    def with_ollama_config(
        self, 
        model: str = "llama3.2:latest",
        base_url: str = "http://localhost:11434",
        **kwargs
    ) -> 'EvaluationConfigBuilder':
        """Set Ollama LLM configuration."""
        self._llm_config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model=model,
            ollama_base_url=base_url,
            **kwargs
        )
        return self
    
    def with_openai_config(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        **kwargs
    ) -> 'EvaluationConfigBuilder':
        """Set OpenAI LLM configuration."""
        self._llm_config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model=model,
            openai_api_key=api_key,
            **kwargs
        )
        return self
    
    def with_huggingface_config(
        self,
        model: str = "microsoft/DialoGPT-medium",
        **kwargs
    ) -> 'EvaluationConfigBuilder':
        """Set HuggingFace LLM configuration."""
        self._llm_config = LLMConfig(
            provider=LLMProvider.HUGGINGFACE,
            model=model,
            **kwargs
        )
        return self
    
    def with_metrics(self, *metrics: EvaluationMetric) -> 'EvaluationConfigBuilder':
        """Add evaluation metrics."""
        self._metrics.extend(metrics)
        return self
    
    def with_all_metrics(self) -> 'EvaluationConfigBuilder':
        """Add all available evaluation metrics."""
        self._metrics = [
            EvaluationMetric.ACCURACY,
            EvaluationMetric.PRECISION,
            EvaluationMetric.RECALL,
            EvaluationMetric.F1,
            EvaluationMetric.SEMANTIC_SIMILARITY,
            EvaluationMetric.ROUGE_L,
            EvaluationMetric.BERT_SCORE
        ]
        return self
    
    def with_basic_metrics(self) -> 'EvaluationConfigBuilder':
        """Add basic evaluation metrics."""
        self._metrics = [
            EvaluationMetric.ACCURACY,
            EvaluationMetric.PRECISION,
            EvaluationMetric.RECALL,
            EvaluationMetric.F1
        ]
        return self
    
    def with_semantic_metrics(self) -> 'EvaluationConfigBuilder':
        """Add semantic evaluation metrics."""
        self._metrics = [
            EvaluationMetric.SEMANTIC_SIMILARITY,
            EvaluationMetric.ROUGE_L,
            EvaluationMetric.BERT_SCORE
        ]
        return self
    
    def with_test_cases(self, *test_cases: TestCase) -> 'EvaluationConfigBuilder':
        """Add test cases."""
        self._test_cases.extend(test_cases)
        return self
    
    def with_data_folder(self, data_folder: Path) -> 'EvaluationConfigBuilder':
        """Load test cases from data folder."""
        test_cases = create_test_cases_from_data_folder(data_folder)
        self._test_cases.extend(test_cases)
        return self
    
    def with_few_shot_extractor(
        self,
        schema_path: Optional[Path] = None,
        examples_folder: Optional[Path] = None,
        name: str = "FewShot",
        enabled: bool = True,
        **config
    ) -> 'EvaluationConfigBuilder':
        """Add FewShot extractor configuration."""
        extractor_config = {
            "schema_path": str(schema_path) if schema_path else None,
            "examples_folder": str(examples_folder) if examples_folder else None,
            **config
        }
        
        self._extractors.append(ExtractorConfig(
            method=ExtractionMethod.FEW_SHOT,
            config=extractor_config,
            name=name,
            enabled=enabled
        ))
        return self
    
    def with_vector_rag_extractor(
        self,
        name: str = "VectorRAG",
        enabled: bool = True,
        **config
    ) -> 'EvaluationConfigBuilder':
        """Add VectorRAG extractor configuration."""
        self._extractors.append(ExtractorConfig(
            method=ExtractionMethod.VECTOR_RAG,
            config=config,
            name=name,
            enabled=enabled
        ))
        return self
    
    def with_graph_rag_extractor(
        self,
        name: str = "GraphRAG",
        enabled: bool = True,
        **config
    ) -> 'EvaluationConfigBuilder':
        """Add GraphRAG extractor configuration."""
        self._extractors.append(ExtractorConfig(
            method=ExtractionMethod.GRAPH_RAG,
            config=config,
            name=name,
            enabled=enabled
        ))
        return self
    
    def with_classification_extractor(
        self,
        name: str = "Classification",
        enabled: bool = True,
        **config
    ) -> 'EvaluationConfigBuilder':
        """Add Classification extractor configuration."""
        self._extractors.append(ExtractorConfig(
            method=ExtractionMethod.CLASSIFICATION,
            config=config,
            name=name,
            enabled=enabled
        ))
        return self
    
    def with_output_dir(self, output_dir: Path) -> 'EvaluationConfigBuilder':
        """Set output directory."""
        self._output_dir = Path(output_dir)
        return self
    
    def with_timeout(self, timeout_seconds: float) -> 'EvaluationConfigBuilder':
        """Set evaluation timeout."""
        self._timeout = timeout_seconds
        return self
    
    def with_detailed_logging(self, enabled: bool = True) -> 'EvaluationConfigBuilder':
        """Enable/disable detailed logging."""
        self._detailed_logging = enabled
        return self
    
    def build(self) -> EvaluationConfig:
        """Build the evaluation configuration."""
        if not self._llm_config:
            raise ValueError("LLM configuration is required")
        
        if not self._metrics:
            # Default to basic metrics
            self._metrics = [
                EvaluationMetric.ACCURACY,
                EvaluationMetric.F1
            ]
        
        if not self._test_cases:
            raise ValueError("At least one test case is required")
        
        if not self._extractors:
            raise ValueError("At least one extractor configuration is required")
        
        return EvaluationConfig(
            metrics=self._metrics,
            test_cases=self._test_cases,
            extractors=self._extractors,
            llm_config=self._llm_config,
            output_dir=self._output_dir,
            timeout_seconds=self._timeout,
            enable_detailed_logging=self._detailed_logging
        )


def create_default_config(
    data_folder: Path,
    schema_path: Path,
    examples_folder: Optional[Path] = None,
    output_dir: Optional[Path] = None
) -> EvaluationConfig:
    """Create a default evaluation configuration."""
    
    if output_dir is None:
        output_dir = Path("tests/reports")
    
    if examples_folder is None:
        examples_folder = data_folder / "labels"
    
    return (EvaluationConfigBuilder()
            .with_ollama_config()
            .with_all_metrics()
            .with_data_folder(data_folder)
            .with_few_shot_extractor(
                schema_path=schema_path,
                examples_folder=examples_folder
            )
            .with_output_dir(output_dir)
            .build())


def create_comprehensive_config(
    data_folder: Path,
    schema_path: Path,
    examples_folder: Optional[Path] = None,
    output_dir: Optional[Path] = None
) -> EvaluationConfig:
    """Create a comprehensive evaluation configuration with all extractors."""
    
    if output_dir is None:
        output_dir = Path("tests/reports")
    
    if examples_folder is None:
        examples_folder = data_folder / "labels"
    
    return (EvaluationConfigBuilder()
            .with_ollama_config()
            .with_all_metrics()
            .with_data_folder(data_folder)
            .with_few_shot_extractor(
                schema_path=schema_path,
                examples_folder=examples_folder,
                name="FewShot_Unified"
            )
            .with_vector_rag_extractor(
                enabled=False  # Enable when implemented
            )
            .with_graph_rag_extractor(
                enabled=False  # Enable when implemented
            )
            .with_classification_extractor(
                enabled=False  # Enable when implemented
            )
            .with_output_dir(output_dir)
            .build())


def create_quick_test_config(
    data_folder: Path,
    schema_path: Path
) -> EvaluationConfig:
    """Create a quick test configuration for development."""
    
    return (EvaluationConfigBuilder()
            .with_ollama_config()
            .with_basic_metrics()
            .with_data_folder(data_folder)
            .with_few_shot_extractor(
                schema_path=schema_path,
                examples_folder=data_folder / "labels"
            )
            .with_output_dir(Path("tests/reports/quick"))
            .with_timeout(60.0)
            .build())


def save_config(config: EvaluationConfig, config_path: Path):
    """Save evaluation configuration to JSON file."""
    
    # Convert to serializable format
    config_dict = {
        "llm_config": {
            "provider": config.llm_config.provider.value,
            "model": config.llm_config.model,
            "temperature": config.llm_config.temperature,
            "max_tokens": config.llm_config.max_tokens,
            "top_p": config.llm_config.top_p,
        },
        "metrics": [metric.value for metric in config.metrics],
        "extractors": [
            {
                "method": extractor.method.value,
                "name": extractor.name,
                "enabled": extractor.enabled,
                "config": extractor.config
            }
            for extractor in config.extractors
        ],
        "output_dir": str(config.output_dir),
        "timeout_seconds": config.timeout_seconds,
        "enable_detailed_logging": config.enable_detailed_logging,
        "test_cases_count": len(config.test_cases)
    }
    
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)


def load_config_template(config_path: Path) -> Dict[str, Any]:
    """Load evaluation configuration template from JSON file."""
    
    with open(config_path, 'r') as f:
        return json.load(f)


# Predefined configuration templates
TEMPLATES = {
    "default": {
        "description": "Default configuration with FewShot extractor and all metrics",
        "builder_func": create_default_config
    },
    "comprehensive": {
        "description": "Comprehensive evaluation with all extractors (when available)",
        "builder_func": create_comprehensive_config
    },
    "quick": {
        "description": "Quick test configuration for development",
        "builder_func": create_quick_test_config
    }
}
