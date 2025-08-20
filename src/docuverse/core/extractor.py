"""
Main document extractor class that orchestrates different extraction methods.
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import logging

from .config import ExtractionConfig, ExtractionMethod
from ..extractors.few_shot import FewShotExtractor
from ..extractors.vector_rag import VectorRAGExtractor  
from ..extractors.graph_rag import GraphRAGExtractor
from ..extractors.reasoning import ReasoningExtractor
from ..extractors.dynamic_graph_rag import DynamicGraphRAGExtractor
from ..extractors.classification import ClassificationExtractor
from ..utils.document_loader import DocumentLoader
from ..utils.schema_validator import SchemaValidator
from ..evaluation.metrics import MetricsCalculator

logger = logging.getLogger(__name__)


class ExtractionResult:
    """Container for extraction results."""
    
    def __init__(
        self,
        method: str,
        extracted_data: Dict[str, Any],
        metadata: Dict[str, Any],
        execution_time: float,
        success: bool = True,
        error: Optional[str] = None
    ):
        self.method = method
        self.extracted_data = extracted_data
        self.metadata = metadata
        self.execution_time = execution_time
        self.success = success
        self.error = error
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "method": self.method,
            "extracted_data": self.extracted_data,
            "metadata": self.metadata,
            "execution_time": self.execution_time,
            "success": self.success,
            "error": self.error
        }


class DocumentExtractor:
    """
    Main document extractor that can run multiple extraction methods
    and compare their performance.
    """
    
    def __init__(self, config: ExtractionConfig):
        """Initialize the document extractor."""
        self.config = config
        self.document_loader = DocumentLoader()
        self.schema_validator = SchemaValidator(config.schema_path) if config.schema_path else None
        self.metrics_calculator = MetricsCalculator()
        
        # Initialize extractors for each method
        self.extractors = self._initialize_extractors()
        
        logger.info(f"Initialized DocumentExtractor with methods: {config.methods}")
    
    def _initialize_extractors(self) -> Dict[str, Any]:
        """Initialize extractor instances for each configured method."""
        extractors = {}
        
        for method in self.config.methods:
            if method == ExtractionMethod.FEW_SHOT:
                extractors[method] = FewShotExtractor(
                    llm_config=self.config.llm_config,
                    examples=self.config.few_shot_examples
                )
            elif method == ExtractionMethod.VECTOR_RAG:
                extractors[method] = VectorRAGExtractor(
                    llm_config=self.config.llm_config,
                    rag_config=self.config.vector_rag_config
                )
            elif method == ExtractionMethod.GRAPH_RAG:
                extractors[method] = GraphRAGExtractor(
                    llm_config=self.config.llm_config,
                    graph_config=self.config.graph_rag_config
                )
            elif method in [ExtractionMethod.REASONING_COT, ExtractionMethod.REASONING_REACT]:
                extractors[method] = ReasoningExtractor(
                    llm_config=self.config.llm_config,
                    reasoning_config=self.config.reasoning_config,
                    method_type=method
                )
            elif method == ExtractionMethod.DYNAMIC_GRAPH_RAG:
                extractors[method] = DynamicGraphRAGExtractor(
                    llm_config=self.config.llm_config,
                    dynamic_config=self.config.dynamic_graph_rag_config
                )
            elif method == ExtractionMethod.CLASSIFICATION:
                extractors[method] = ClassificationExtractor(
                    llm_config=self.config.llm_config,
                    schema_path=self.config.schema_path
                )
        
        return extractors
    
    def extract(
        self,
        document_path: Union[str, Path],
        methods: Optional[List[str]] = None
    ) -> Dict[str, ExtractionResult]:
        """
        Extract information from a document using specified methods.
        
        Args:
            document_path: Path to the document to process
            methods: Specific methods to run (if None, runs all configured methods)
            
        Returns:
            Dictionary mapping method names to extraction results
        """
        # Load document
        document = self.document_loader.load(document_path)
        
        # Determine which methods to run
        methods_to_run = methods or [m.value for m in self.config.methods]
        
        results = {}
        
        if self.config.parallel_processing and len(methods_to_run) > 1:
            # Run methods in parallel
            results = self._extract_parallel(document, methods_to_run)
        else:
            # Run methods sequentially
            results = self._extract_sequential(document, methods_to_run)
        
        logger.info(f"Completed extraction for {len(results)} methods")
        return results
    
    def _extract_sequential(
        self,
        document: Dict[str, Any],
        methods: List[str]
    ) -> Dict[str, ExtractionResult]:
        """Extract using methods sequentially."""
        results = {}
        
        for method in methods:
            if method not in self.extractors:
                logger.warning(f"Method {method} not configured, skipping")
                continue
                
            result = self._run_single_extraction(method, document)
            results[method] = result
        
        return results
    
    def _extract_parallel(
        self,
        document: Dict[str, Any],
        methods: List[str]
    ) -> Dict[str, ExtractionResult]:
        """Extract using methods in parallel."""
        import concurrent.futures
        
        results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all extraction tasks
            future_to_method = {
                executor.submit(self._run_single_extraction, method, document): method
                for method in methods if method in self.extractors
            }
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_method):
                method = future_to_method[future]
                try:
                    result = future.result()
                    results[method] = result
                except Exception as e:
                    logger.error(f"Method {method} failed: {str(e)}")
                    results[method] = ExtractionResult(
                        method=method,
                        extracted_data={},
                        metadata={},
                        execution_time=0.0,
                        success=False,
                        error=str(e)
                    )
        
        return results
    
    def _run_single_extraction(
        self,
        method: str,
        document: Dict[str, Any]
    ) -> ExtractionResult:
        """Run extraction for a single method with error handling and timing."""
        start_time = time.time()
        
        try:
            extractor = self.extractors[method]
            
            # Run extraction with retries
            for attempt in range(self.config.max_retries):
                try:
                    extracted_data = extractor.extract(document)
                    
                    # Validate against schema if provided
                    if self.schema_validator:
                        validation_result = self.schema_validator.validate(extracted_data)
                        if not validation_result.is_valid:
                            logger.warning(f"Schema validation failed for {method}: {validation_result.errors}")
                    
                    execution_time = time.time() - start_time
                    
                    # Collect metadata
                    metadata = {
                        "attempt": attempt + 1,
                        "tokens_used": getattr(extractor, 'last_token_usage', None),
                        "chunks_processed": getattr(extractor, 'last_chunks_processed', None),
                        "confidence_score": getattr(extractor, 'last_confidence', None)
                    }
                    
                    return ExtractionResult(
                        method=method,
                        extracted_data=extracted_data,
                        metadata=metadata,
                        execution_time=execution_time,
                        success=True
                    )
                    
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed for {method}: {str(e)}")
                    if attempt == self.config.max_retries - 1:
                        raise e
                    time.sleep(1)  # Brief delay before retry
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"All attempts failed for {method}: {str(e)}")
            
            return ExtractionResult(
                method=method,
                extracted_data={},
                metadata={"error_details": str(e)},
                execution_time=execution_time,
                success=False,
                error=str(e)
            )
    
    def extract_and_evaluate(
        self,
        document_path: Union[str, Path],
        ground_truth: Optional[Union[str, Path, Dict[str, Any]]] = None,
        methods: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Extract information and evaluate against ground truth.
        
        Args:
            document_path: Path to document to process
            ground_truth: Ground truth data (path to file or dict)
            methods: Specific methods to run
            
        Returns:
            Dictionary containing extraction results and evaluation metrics
        """
        # Run extraction
        extraction_results = self.extract(document_path, methods)
        
        # Load ground truth if provided
        if ground_truth is not None:
            if isinstance(ground_truth, (str, Path)):
                with open(ground_truth, 'r') as f:
                    ground_truth_data = json.load(f)
            else:
                ground_truth_data = ground_truth
            
            # Calculate metrics for each method
            evaluation_results = {}
            for method, result in extraction_results.items():
                if result.success:
                    metrics = self.metrics_calculator.calculate_metrics(
                        predicted=result.extracted_data,
                        ground_truth=ground_truth_data,
                        metrics=self.config.evaluation_metrics
                    )
                    evaluation_results[method] = metrics
                else:
                    evaluation_results[method] = {"error": result.error}
            
            return {
                "extraction_results": {k: v.to_dict() for k, v in extraction_results.items()},
                "evaluation_results": evaluation_results,
                "ground_truth": ground_truth_data
            }
        else:
            return {
                "extraction_results": {k: v.to_dict() for k, v in extraction_results.items()}
            }
    
    def benchmark(
        self,
        dataset_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Run benchmark evaluation on a dataset.
        
        Args:
            dataset_path: Path to benchmark dataset
            output_path: Optional path to save results
            
        Returns:
            Comprehensive benchmark results
        """
        # This will be implemented to handle benchmark datasets
        # with multiple documents and ground truth annotations
        pass
