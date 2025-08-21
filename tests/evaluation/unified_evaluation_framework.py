"""
Enhanced Evaluation Framework for DocuVerse Extractors

This module provides a unified evaluation framework that can be used across
all extraction methods (few_shot, vector_rag, graph_rag, etc.) using the
existing evaluation metrics from src/docuverse/evaluation.

Key Features:
- Unified interface for all extractor types
- Comprehensive metrics using src evaluation modules
- Performance benchmarking
- Cross-method comparison
- Detailed reporting with visualizations
- Extensible for new extractors and metrics
"""

import json
import time
import logging
from typing import Dict, Any, List, Optional, Union, Tuple, Type
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from docuverse.core.config import LLMConfig, EvaluationMetric, ExtractionMethod
from docuverse.evaluation.evaluator import Evaluator, EvaluationResult
from docuverse.evaluation.metrics import MetricsCalculator
from docuverse.extractors.base import BaseExtractor
from docuverse.extractors.few_shot import FewShotExtractor
# Import other extractors as needed
# from docuverse.extractors.vector_rag import VectorRAGExtractor
# from docuverse.extractors.graph_rag import GraphRAGExtractor

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """Represents a single test case for evaluation."""
    name: str
    document: Dict[str, Any]
    ground_truth: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    expected_confidence: Optional[float] = None
    max_extraction_time: Optional[float] = None


@dataclass
class ExtractorConfig:
    """Configuration for an extractor to be tested."""
    method: ExtractionMethod
    config: Dict[str, Any]
    name: Optional[str] = None
    enabled: bool = True


@dataclass
class EvaluationConfig:
    """Configuration for the evaluation suite."""
    metrics: List[EvaluationMetric]
    test_cases: List[TestCase]
    extractors: List[ExtractorConfig]
    llm_config: LLMConfig
    output_dir: Path
    timeout_seconds: float = 300.0
    enable_detailed_logging: bool = True


class ExtractorFactory:
    """Factory for creating extractors based on method type."""
    
    @staticmethod
    def create_extractor(
        method: ExtractionMethod,
        llm_config: LLMConfig,
        config: Dict[str, Any]
    ) -> BaseExtractor:
        """Create extractor instance based on method type."""
        
        if method == ExtractionMethod.FEW_SHOT:
            return FewShotExtractor(
                llm_config=llm_config,
                **config
            )
        elif method == ExtractionMethod.VECTOR_RAG:
            # Placeholder - implement when VectorRAGExtractor is available
            raise NotImplementedError("VectorRAG extractor not yet implemented")
        elif method == ExtractionMethod.GRAPH_RAG:
            # Placeholder - implement when GraphRAGExtractor is available
            raise NotImplementedError("GraphRAG extractor not yet implemented")
        elif method == ExtractionMethod.CLASSIFICATION:
            # Placeholder - implement when ClassificationExtractor is available
            raise NotImplementedError("Classification extractor not yet implemented")
        else:
            raise ValueError(f"Unsupported extraction method: {method}")


@dataclass
class ExtractionPerformance:
    """Performance metrics for a single extraction."""
    extraction_time: float
    confidence: float
    success: bool
    error: Optional[str] = None
    token_usage: Optional[int] = None
    memory_usage: Optional[float] = None


@dataclass
class ExtractorResult:
    """Complete result from an extractor evaluation."""
    extractor_name: str
    method: ExtractionMethod
    test_case_name: str
    extracted_data: Dict[str, Any]
    ground_truth: Optional[Dict[str, Any]]
    performance: ExtractionPerformance
    metrics: Dict[str, float]
    validation_errors: List[str]


class UnifiedEvaluationFramework:
    """
    Unified evaluation framework for all DocuVerse extractors.
    
    This framework provides:
    - Standardized evaluation across all extractor types
    - Comprehensive metrics calculation
    - Performance benchmarking
    - Cross-method comparison
    - Detailed reporting
    """
    
    def __init__(self, config: EvaluationConfig):
        """Initialize the evaluation framework."""
        self.config = config
        self.evaluator = Evaluator()
        self.metrics_calculator = MetricsCalculator()
        self.results: List[ExtractorResult] = []
        
        # Setup output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        if config.enable_detailed_logging:
            self._setup_logging()
        
        logger.info(f"Initialized UnifiedEvaluationFramework with {len(config.extractors)} extractors")
    
    def _setup_logging(self):
        """Setup detailed logging for evaluation."""
        log_file = self.config.output_dir / "evaluation.log"
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        handler = logging.FileHandler(log_file)
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    
    def run_evaluation(self) -> Dict[str, Any]:
        """
        Run complete evaluation suite across all extractors and test cases.
        
        Returns:
            Comprehensive evaluation results
        """
        logger.info("Starting unified evaluation suite")
        start_time = time.time()
        
        # Initialize extractors
        extractors = self._initialize_extractors()
        
        # Run evaluations
        for extractor_config in self.config.extractors:
            if not extractor_config.enabled:
                logger.info(f"Skipping disabled extractor: {extractor_config.name}")
                continue
            
            try:
                extractor = extractors[extractor_config.method]
                self._evaluate_extractor(extractor, extractor_config)
            except Exception as e:
                logger.error(f"Failed to evaluate {extractor_config.name}: {e}")
                # Add failed result
                self._add_failed_result(extractor_config, str(e))
        
        # Calculate summary statistics
        summary = self._calculate_summary()
        
        # Generate comparison analysis
        comparison = self._generate_comparison_analysis()
        
        total_time = time.time() - start_time
        
        evaluation_report = {
            "evaluation_config": asdict(self.config),
            "summary": summary,
            "comparison": comparison,
            "individual_results": [asdict(result) for result in self.results],
            "evaluation_metadata": {
                "total_evaluation_time": total_time,
                "timestamp": time.time(),
                "framework_version": "1.0.0"
            }
        }
        
        # Save detailed results
        self._save_results(evaluation_report)
        
        logger.info(f"Evaluation completed in {total_time:.2f} seconds")
        return evaluation_report
    
    def _initialize_extractors(self) -> Dict[ExtractionMethod, BaseExtractor]:
        """Initialize all configured extractors."""
        extractors = {}
        
        for extractor_config in self.config.extractors:
            if not extractor_config.enabled:
                continue
            
            try:
                extractor = ExtractorFactory.create_extractor(
                    extractor_config.method,
                    self.config.llm_config,
                    extractor_config.config
                )
                extractors[extractor_config.method] = extractor
                logger.info(f"Initialized {extractor_config.method} extractor")
            except Exception as e:
                logger.error(f"Failed to initialize {extractor_config.method}: {e}")
        
        return extractors
    
    def _evaluate_extractor(self, extractor: BaseExtractor, config: ExtractorConfig):
        """Evaluate a single extractor against all test cases."""
        extractor_name = config.name or config.method.value
        logger.info(f"Evaluating {extractor_name}")
        
        for test_case in self.config.test_cases:
            try:
                result = self._evaluate_single_case(extractor, config, test_case)
                self.results.append(result)
            except Exception as e:
                logger.error(f"Failed to evaluate {extractor_name} on {test_case.name}: {e}")
                self._add_failed_case_result(config, test_case, str(e))
    
    def _evaluate_single_case(
        self, 
        extractor: BaseExtractor, 
        config: ExtractorConfig, 
        test_case: TestCase
    ) -> ExtractorResult:
        """Evaluate a single test case with an extractor."""
        
        extractor_name = config.name or config.method.value
        logger.debug(f"Running {extractor_name} on {test_case.name}")
        
        # Measure performance
        start_time = time.time()
        
        try:
            # Perform extraction
            extracted_data = extractor.extract(test_case.document)
            end_time = time.time()
            
            # Get performance metrics
            performance = ExtractionPerformance(
                extraction_time=end_time - start_time,
                confidence=getattr(extractor, 'last_confidence', 0.0),
                success=True,
                token_usage=getattr(extractor, 'last_token_usage', None)
            )
            
            # Validate extraction
            validation_errors = []
            if hasattr(extractor, 'validate_schema_compliance'):
                validation = extractor.validate_schema_compliance(extracted_data)
                if not validation.get('is_valid', True):
                    validation_errors.extend(
                        validation.get('missing_fields', []) +
                        validation.get('invalid_enums', []) +
                        validation.get('structure_errors', [])
                    )
            
            # Calculate metrics if ground truth available
            metrics = {}
            if test_case.ground_truth:
                metrics = self.metrics_calculator.calculate_metrics(
                    extracted_data,
                    test_case.ground_truth,
                    self.config.metrics
                )
            
            return ExtractorResult(
                extractor_name=extractor_name,
                method=config.method,
                test_case_name=test_case.name,
                extracted_data=extracted_data,
                ground_truth=test_case.ground_truth,
                performance=performance,
                metrics=metrics,
                validation_errors=validation_errors
            )
            
        except Exception as e:
            end_time = time.time()
            
            performance = ExtractionPerformance(
                extraction_time=end_time - start_time,
                confidence=0.0,
                success=False,
                error=str(e)
            )
            
            return ExtractorResult(
                extractor_name=extractor_name,
                method=config.method,
                test_case_name=test_case.name,
                extracted_data={},
                ground_truth=test_case.ground_truth,
                performance=performance,
                metrics={},
                validation_errors=[f"Extraction failed: {e}"]
            )
    
    def _add_failed_result(self, config: ExtractorConfig, error: str):
        """Add a failed result for an extractor that couldn't be initialized."""
        for test_case in self.config.test_cases:
            self._add_failed_case_result(config, test_case, error)
    
    def _add_failed_case_result(self, config: ExtractorConfig, test_case: TestCase, error: str):
        """Add a failed result for a specific test case."""
        extractor_name = config.name or config.method.value
        
        performance = ExtractionPerformance(
            extraction_time=0.0,
            confidence=0.0,
            success=False,
            error=error
        )
        
        result = ExtractorResult(
            extractor_name=extractor_name,
            method=config.method,
            test_case_name=test_case.name,
            extracted_data={},
            ground_truth=test_case.ground_truth,
            performance=performance,
            metrics={},
            validation_errors=[f"Extractor initialization failed: {error}"]
        )
        
        self.results.append(result)
    
    def _calculate_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics across all results."""
        if not self.results:
            return {}
        
        # Group results by extractor
        extractor_results = defaultdict(list)
        for result in self.results:
            extractor_results[result.extractor_name].append(result)
        
        summary = {}
        
        for extractor_name, results in extractor_results.items():
            successful_results = [r for r in results if r.performance.success]
            
            extractor_summary = {
                "total_tests": len(results),
                "successful_tests": len(successful_results),
                "success_rate": len(successful_results) / len(results) if results else 0.0,
                "average_extraction_time": 0.0,
                "average_confidence": 0.0,
                "metrics_summary": {}
            }
            
            if successful_results:
                # Performance averages
                extractor_summary["average_extraction_time"] = sum(
                    r.performance.extraction_time for r in successful_results
                ) / len(successful_results)
                
                extractor_summary["average_confidence"] = sum(
                    r.performance.confidence for r in successful_results
                ) / len(successful_results)
                
                # Metrics averages
                all_metrics = defaultdict(list)
                for result in successful_results:
                    for metric, value in result.metrics.items():
                        all_metrics[metric].append(value)
                
                for metric, values in all_metrics.items():
                    extractor_summary["metrics_summary"][metric] = {
                        "mean": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                        "count": len(values)
                    }
            
            summary[extractor_name] = extractor_summary
        
        return summary
    
    def _generate_comparison_analysis(self) -> Dict[str, Any]:
        """Generate cross-extractor comparison analysis."""
        if not self.results:
            return {}
        
        # Collect results by test case for fair comparison
        test_case_results = defaultdict(list)
        for result in self.results:
            test_case_results[result.test_case_name].append(result)
        
        comparison = {
            "metric_rankings": {},
            "performance_rankings": {},
            "overall_ranking": [],
            "head_to_head": {}
        }
        
        # Calculate metric rankings
        metric_scores = defaultdict(lambda: defaultdict(list))
        
        for test_case_name, results in test_case_results.items():
            for result in results:
                if result.performance.success:
                    for metric, value in result.metrics.items():
                        metric_scores[metric][result.extractor_name].append(value)
        
        # Average scores per metric per extractor
        for metric, extractor_scores in metric_scores.items():
            ranking = []
            for extractor, scores in extractor_scores.items():
                avg_score = sum(scores) / len(scores) if scores else 0.0
                ranking.append((extractor, avg_score))
            
            ranking.sort(key=lambda x: x[1], reverse=True)
            comparison["metric_rankings"][metric] = ranking
        
        return comparison
    
    def _save_results(self, evaluation_report: Dict[str, Any]):
        """Save evaluation results in multiple formats."""
        
        # Save JSON results
        json_path = self.config.output_dir / "evaluation_results.json"
        with open(json_path, 'w') as f:
            json.dump(evaluation_report, f, indent=2, default=str)
        
        # Generate and save markdown report
        markdown_report = self._generate_markdown_report(evaluation_report)
        markdown_path = self.config.output_dir / "evaluation_report.md"
        with open(markdown_path, 'w') as f:
            f.write(markdown_report)
        
        # Generate and save HTML report using existing evaluator
        html_path = self.config.output_dir / "evaluation_report.html"
        try:
            html_report = self.evaluator.generate_report(
                evaluation_report, 
                output_path=html_path, 
                format="html"
            )
        except Exception as e:
            logger.warning(f"Failed to generate HTML report: {e}")
        
        logger.info(f"Results saved to {self.config.output_dir}")
    
    def _generate_markdown_report(self, evaluation_report: Dict[str, Any]) -> str:
        """Generate detailed markdown report."""
        
        summary = evaluation_report.get("summary", {})
        comparison = evaluation_report.get("comparison", {})
        
        report_lines = [
            "# DocuVerse Unified Extraction Evaluation Report",
            f"",
            f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Framework Version**: {evaluation_report['evaluation_metadata']['framework_version']}",
            f"**Total Evaluation Time**: {evaluation_report['evaluation_metadata']['total_evaluation_time']:.2f}s",
            f"",
            "## Executive Summary",
            f"",
        ]
        
        # Overall statistics
        total_tests = sum(s.get("total_tests", 0) for s in summary.values())
        total_successful = sum(s.get("successful_tests", 0) for s in summary.values())
        overall_success_rate = total_successful / total_tests if total_tests > 0 else 0.0
        
        report_lines.extend([
            f"- **Total Tests Run**: {total_tests}",
            f"- **Successful Extractions**: {total_successful}",
            f"- **Overall Success Rate**: {overall_success_rate:.1%}",
            f"- **Extractors Evaluated**: {len(summary)}",
            f"",
            "## Extractor Performance Summary",
            f"",
            "| Extractor | Tests | Success Rate | Avg Time | Avg Confidence |",
            "| --------- | ----- | ------------ | -------- | -------------- |",
        ])
        
        for extractor_name, extractor_summary in summary.items():
            success_rate = extractor_summary.get("success_rate", 0.0)
            avg_time = extractor_summary.get("average_extraction_time", 0.0)
            avg_confidence = extractor_summary.get("average_confidence", 0.0)
            total_tests = extractor_summary.get("total_tests", 0)
            
            report_lines.append(
                f"| {extractor_name} | {total_tests} | {success_rate:.1%} | {avg_time:.3f}s | {avg_confidence:.3f} |"
            )
        
        # Metric rankings
        report_lines.extend([
            f"",
            "## Metric Rankings",
            f"",
        ])
        
        metric_rankings = comparison.get("metric_rankings", {})
        for metric, rankings in metric_rankings.items():
            report_lines.extend([
                f"### {metric.title()}",
                f"",
                "| Rank | Extractor | Score |",
                "| ---- | --------- | ----- |",
            ])
            
            for i, (extractor, score) in enumerate(rankings, 1):
                report_lines.append(f"| {i} | {extractor} | {score:.3f} |")
            
            report_lines.append("")
        
        # Detailed results
        report_lines.extend([
            "## Detailed Results by Extractor",
            ""
        ])
        
        for extractor_name, extractor_summary in summary.items():
            report_lines.extend([
                f"### {extractor_name}",
                "",
                f"- **Total Tests**: {extractor_summary.get('total_tests', 0)}",
                f"- **Successful Tests**: {extractor_summary.get('successful_tests', 0)}",
                f"- **Success Rate**: {extractor_summary.get('success_rate', 0.0):.1%}",
                f"- **Average Extraction Time**: {extractor_summary.get('average_extraction_time', 0.0):.3f}s",
                f"- **Average Confidence**: {extractor_summary.get('average_confidence', 0.0):.3f}",
                ""
            ])
            
            # Metrics summary
            metrics_summary = extractor_summary.get("metrics_summary", {})
            if metrics_summary:
                report_lines.extend([
                    "#### Metrics Summary",
                    "",
                    "| Metric | Mean | Min | Max | Count |",
                    "| ------ | ---- | --- | --- | ----- |",
                ])
                
                for metric, stats in metrics_summary.items():
                    report_lines.append(
                        f"| {metric} | {stats['mean']:.3f} | {stats['min']:.3f} | {stats['max']:.3f} | {stats['count']} |"
                    )
                
                report_lines.append("")
        
        return "\n".join(report_lines)


def create_test_cases_from_data_folder(data_folder: Path) -> List[TestCase]:
    """Create test cases from data folder structure."""
    test_cases = []
    
    # Look for documents and corresponding labels
    if (data_folder / "labels").exists():
        labels_folder = data_folder / "labels"
        
        for label_file in labels_folder.glob("*.json"):
            # Load ground truth
            with open(label_file, 'r') as f:
                ground_truth = json.load(f)
            
            # Find corresponding document
            base_name = label_file.stem.replace('_label', '').replace('.label', '')
            
            document_file = None
            for ext in ['.txt', '.md', '.pdf', '.docx']:
                potential_doc = data_folder / f"{base_name}{ext}"
                if potential_doc.exists():
                    document_file = potential_doc
                    break
            
            if document_file:
                with open(document_file, 'r') as f:
                    content = f.read()
            else:
                # Generate synthetic content from labels
                content = f"Test document for {base_name}"
            
            test_case = TestCase(
                name=base_name,
                document={
                    "content": content,
                    "metadata": {
                        "filename": document_file.name if document_file else f"{base_name}.txt",
                        "source": "data_folder"
                    }
                },
                ground_truth=ground_truth,
                metadata={"label_file": str(label_file)}
            )
            
            test_cases.append(test_case)
    
    return test_cases
