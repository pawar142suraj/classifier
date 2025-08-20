"""
Metrics calculation for evaluation.
"""

from typing import Dict, Any, List
import logging

from ..core.config import EvaluationMetric

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Utility for calculating various evaluation metrics.
    """
    
    def __init__(self):
        """Initialize metrics calculator."""
        logger.info("Initialized MetricsCalculator")
    
    def calculate_metrics(
        self,
        predicted: Dict[str, Any],
        ground_truth: Dict[str, Any],
        metrics: List[EvaluationMetric]
    ) -> Dict[str, float]:
        """
        Calculate specified metrics for predicted vs ground truth data.
        
        Args:
            predicted: Predicted extraction results
            ground_truth: Ground truth data
            metrics: List of metrics to calculate
            
        Returns:
            Dictionary of metric names to values
        """
        results = {}
        
        for metric in metrics:
            if metric == EvaluationMetric.ACCURACY:
                results["accuracy"] = self._calculate_accuracy(predicted, ground_truth)
            elif metric == EvaluationMetric.PRECISION:
                results["precision"] = self._calculate_precision(predicted, ground_truth)
            elif metric == EvaluationMetric.RECALL:
                results["recall"] = self._calculate_recall(predicted, ground_truth)
            elif metric == EvaluationMetric.F1:
                f1, precision, recall = self._calculate_f1_precision_recall(predicted, ground_truth)
                results["f1"] = f1
                results["precision"] = precision
                results["recall"] = recall
            elif metric == EvaluationMetric.SEMANTIC_SIMILARITY:
                results["semantic_similarity"] = self._calculate_semantic_similarity(predicted, ground_truth)
            elif metric == EvaluationMetric.ROUGE_L:
                results["rouge_l"] = self._calculate_rouge_l(predicted, ground_truth)
            elif metric == EvaluationMetric.BERT_SCORE:
                results["bert_score"] = self._calculate_bert_score(predicted, ground_truth)
        
        return results
    
    def _calculate_accuracy(self, predicted: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
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
    
    def _calculate_precision(self, predicted: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
        """Calculate precision."""
        if not predicted:
            return 0.0
        
        predicted_fields = set(predicted.keys())
        ground_truth_fields = set(ground_truth.keys()) if ground_truth else set()
        
        true_positives = len(predicted_fields & ground_truth_fields)
        false_positives = len(predicted_fields - ground_truth_fields)
        
        return true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    
    def _calculate_recall(self, predicted: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
        """Calculate recall."""
        if not ground_truth:
            return 0.0
        
        predicted_fields = set(predicted.keys()) if predicted else set()
        ground_truth_fields = set(ground_truth.keys())
        
        true_positives = len(predicted_fields & ground_truth_fields)
        false_negatives = len(ground_truth_fields - predicted_fields)
        
        return true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    
    def _calculate_f1_precision_recall(self, predicted: Dict[str, Any], ground_truth: Dict[str, Any]) -> tuple:
        """Calculate F1, precision, and recall."""
        precision = self._calculate_precision(predicted, ground_truth)
        recall = self._calculate_recall(predicted, ground_truth)
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return f1, precision, recall
    
    def _calculate_semantic_similarity(self, predicted: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
        """Calculate semantic similarity between predicted and ground truth."""
        # Placeholder implementation
        # In a full implementation, this would use sentence transformers
        # to compute semantic similarity between text values
        
        if not predicted or not ground_truth:
            return 0.0
        
        # Simple text similarity based on common words
        predicted_text = " ".join(str(v) for v in predicted.values() if v)
        ground_truth_text = " ".join(str(v) for v in ground_truth.values() if v)
        
        if not predicted_text or not ground_truth_text:
            return 0.0
        
        predicted_words = set(predicted_text.lower().split())
        ground_truth_words = set(ground_truth_text.lower().split())
        
        intersection = predicted_words & ground_truth_words
        union = predicted_words | ground_truth_words
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_rouge_l(self, predicted: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
        """Calculate ROUGE-L score."""
        # Placeholder implementation
        # In a full implementation, this would use the rouge-score library
        return 0.75  # Placeholder value
    
    def _calculate_bert_score(self, predicted: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
        """Calculate BERTScore."""
        # Placeholder implementation
        # In a full implementation, this would use the bert-score library
        return 0.80  # Placeholder value
