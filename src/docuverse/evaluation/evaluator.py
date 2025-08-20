"""
Evaluation utilities and metrics calculation.
"""

import json
import time
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import logging

from ..core.config import EvaluationMetric

logger = logging.getLogger(__name__)


class EvaluationResult:
    """Container for evaluation results."""
    
    def __init__(self, metrics: Dict[str, float], method_comparison: Optional[Dict] = None):
        self.metrics = metrics
        self.method_comparison = method_comparison or {}
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metrics": self.metrics,
            "method_comparison": self.method_comparison,
            "timestamp": self.timestamp
        }


class Evaluator:
    """
    Main evaluator class for comparing extraction methods and generating reports.
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        self.results_history = []
        logger.info("Initialized Evaluator")
    
    def evaluate_single_result(
        self,
        extraction_result: Dict[str, Any],
        ground_truth: Dict[str, Any],
        metrics: List[EvaluationMetric]
    ) -> EvaluationResult:
        """
        Evaluate a single extraction result against ground truth.
        
        Args:
            extraction_result: The extracted data
            ground_truth: The ground truth data
            metrics: List of metrics to calculate
            
        Returns:
            EvaluationResult containing calculated metrics
        """
        calculated_metrics = {}
        
        for metric in metrics:
            if metric == EvaluationMetric.ACCURACY:
                calculated_metrics["accuracy"] = self._calculate_accuracy(
                    extraction_result, ground_truth
                )
            elif metric == EvaluationMetric.F1:
                f1, precision, recall = self._calculate_f1_precision_recall(
                    extraction_result, ground_truth
                )
                calculated_metrics["f1"] = f1
                calculated_metrics["precision"] = precision
                calculated_metrics["recall"] = recall
            elif metric == EvaluationMetric.SEMANTIC_SIMILARITY:
                calculated_metrics["semantic_similarity"] = self._calculate_semantic_similarity(
                    extraction_result, ground_truth
                )
            # Add more metrics as needed
        
        return EvaluationResult(calculated_metrics)
    
    def compare_methods(
        self,
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare multiple extraction methods.
        
        Args:
            results: Dictionary of results from extract_and_evaluate
            
        Returns:
            Comparison analysis
        """
        if "evaluation_results" not in results:
            raise ValueError("Results must contain evaluation_results")
        
        evaluation_results = results["evaluation_results"]
        method_names = list(evaluation_results.keys())
        
        # Calculate rankings for each metric
        rankings = {}
        all_metrics = set()
        
        # Collect all metrics
        for method_result in evaluation_results.values():
            if isinstance(method_result, dict) and "error" not in method_result:
                all_metrics.update(method_result.keys())
        
        # Rank methods for each metric
        for metric in all_metrics:
            metric_values = {}
            for method, method_result in evaluation_results.items():
                if isinstance(method_result, dict) and metric in method_result:
                    metric_values[method] = method_result[metric]
            
            # Sort by value (higher is better for most metrics)
            sorted_methods = sorted(
                metric_values.items(),
                key=lambda x: x[1],
                reverse=True
            )
            rankings[metric] = sorted_methods
        
        # Calculate overall ranking (simple average of ranks)
        overall_scores = {}
        for method in method_names:
            total_rank = 0
            valid_metrics = 0
            
            for metric, ranking in rankings.items():
                for i, (ranked_method, _) in enumerate(ranking):
                    if ranked_method == method:
                        total_rank += i + 1  # 1-based ranking
                        valid_metrics += 1
                        break
            
            if valid_metrics > 0:
                overall_scores[method] = total_rank / valid_metrics
        
        overall_ranking = sorted(
            overall_scores.items(),
            key=lambda x: x[1]  # Lower average rank is better
        )
        
        return {
            "metric_rankings": rankings,
            "overall_ranking": overall_ranking,
            "method_count": len(method_names),
            "metrics_evaluated": list(all_metrics)
        }
    
    def generate_report(
        self,
        results: Dict[str, Any],
        output_path: Optional[Union[str, Path]] = None,
        format: str = "html"
    ) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            results: Results from extract_and_evaluate
            output_path: Optional path to save the report
            format: Report format (html, json, markdown)
            
        Returns:
            Report content as string
        """
        if format == "html":
            report_content = self._generate_html_report(results)
        elif format == "json":
            report_content = json.dumps(results, indent=2)
        elif format == "markdown":
            report_content = self._generate_markdown_report(results)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            logger.info(f"Report saved to {output_path}")
        
        return report_content
    
    def _calculate_accuracy(
        self,
        predicted: Dict[str, Any],
        ground_truth: Dict[str, Any]
    ) -> float:
        """Calculate field-level accuracy."""
        if not predicted or not ground_truth:
            return 0.0
        
        total_fields = 0
        correct_fields = 0
        
        for key in ground_truth.keys():
            total_fields += 1
            if key in predicted and predicted[key] == ground_truth[key]:
                correct_fields += 1
        
        return correct_fields / total_fields if total_fields > 0 else 0.0
    
    def _calculate_f1_precision_recall(
        self,
        predicted: Dict[str, Any],
        ground_truth: Dict[str, Any]
    ) -> tuple:
        """Calculate F1, precision, and recall."""
        # Simplified implementation - in practice, this would be more sophisticated
        # based on the specific task (entity extraction, field extraction, etc.)
        
        predicted_fields = set(predicted.keys()) if predicted else set()
        ground_truth_fields = set(ground_truth.keys()) if ground_truth else set()
        
        if not ground_truth_fields:
            return 0.0, 0.0, 0.0
        
        true_positives = len(predicted_fields & ground_truth_fields)
        false_positives = len(predicted_fields - ground_truth_fields)
        false_negatives = len(ground_truth_fields - predicted_fields)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return f1, precision, recall
    
    def _calculate_semantic_similarity(
        self,
        predicted: Dict[str, Any],
        ground_truth: Dict[str, Any]
    ) -> float:
        """Calculate semantic similarity between predicted and ground truth."""
        # Placeholder implementation - would use sentence transformers or similar
        # for actual semantic similarity calculation
        return 0.85  # Placeholder value
    
    def _generate_html_report(self, results: Dict[str, Any]) -> str:
        """Generate HTML report."""
        comparison = self.compare_methods(results)
        
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>DocuVerse Extraction Evaluation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
                .method-section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
                .metrics-table { width: 100%; border-collapse: collapse; margin: 10px 0; }
                .metrics-table th, .metrics-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                .metrics-table th { background-color: #f2f2f2; }
                .success { color: #27ae60; }
                .error { color: #e74c3c; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>DocuVerse Extraction Evaluation Report</h1>
                <p>Generated on: {timestamp}</p>
            </div>
            
            <h2>Overall Method Ranking</h2>
            <table class="metrics-table">
                <tr><th>Rank</th><th>Method</th><th>Average Score</th></tr>
                {overall_ranking_rows}
            </table>
            
            <h2>Detailed Results by Method</h2>
            {method_details}
            
        </body>
        </html>
        """
        
        # Generate ranking rows
        ranking_rows = ""
        for i, (method, score) in enumerate(comparison["overall_ranking"], 1):
            ranking_rows += f"<tr><td>{i}</td><td>{method}</td><td>{score:.3f}</td></tr>"
        
        # Generate method details
        method_details = ""
        if "evaluation_results" in results:
            for method, eval_result in results["evaluation_results"].items():
                if isinstance(eval_result, dict) and "error" not in eval_result:
                    metrics_rows = ""
                    for metric, value in eval_result.items():
                        metrics_rows += f"<tr><td>{metric}</td><td>{value:.3f}</td></tr>"
                    
                    method_details += f"""
                    <div class="method-section">
                        <h3 class="success">{method}</h3>
                        <table class="metrics-table">
                            <tr><th>Metric</th><th>Value</th></tr>
                            {metrics_rows}
                        </table>
                    </div>
                    """
                else:
                    error_msg = eval_result.get("error", "Unknown error") if isinstance(eval_result, dict) else "Unknown error"
                    method_details += f"""
                    <div class="method-section">
                        <h3 class="error">{method} (Failed)</h3>
                        <p class="error">Error: {error_msg}</p>
                    </div>
                    """
        
        return html_template.format(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            overall_ranking_rows=ranking_rows,
            method_details=method_details
        )
    
    def _generate_markdown_report(self, results: Dict[str, Any]) -> str:
        """Generate Markdown report."""
        comparison = self.compare_methods(results)
        
        report = f"""# DocuVerse Extraction Evaluation Report

Generated on: {time.strftime("%Y-%m-%d %H:%M:%S")}

## Overall Method Ranking

| Rank | Method | Average Score |
|------|--------|---------------|
"""
        
        for i, (method, score) in enumerate(comparison["overall_ranking"], 1):
            report += f"| {i} | {method} | {score:.3f} |\n"
        
        report += "\n## Detailed Results by Method\n\n"
        
        if "evaluation_results" in results:
            for method, eval_result in results["evaluation_results"].items():
                report += f"### {method}\n\n"
                
                if isinstance(eval_result, dict) and "error" not in eval_result:
                    report += "| Metric | Value |\n|--------|-------|\n"
                    for metric, value in eval_result.items():
                        report += f"| {metric} | {value:.3f} |\n"
                else:
                    error_msg = eval_result.get("error", "Unknown error") if isinstance(eval_result, dict) else "Unknown error"
                    report += f"**Error:** {error_msg}\n"
                
                report += "\n"
        
        return report
