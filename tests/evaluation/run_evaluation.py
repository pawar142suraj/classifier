"""
Sample Evaluation Runner

This script demonstrates how to use the unified evaluation framework
to test all DocuVerse extraction methods.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from unified_evaluation_framework import UnifiedEvaluationFramework
from evaluation_config import EvaluationConfigBuilder, create_default_config
from docuverse.core.config import EvaluationMetric, ExtractionMethod

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def run_comprehensive_evaluation():
    """Run comprehensive evaluation across all available extractors."""
    
    logger.info("Starting comprehensive DocuVerse evaluation")
    
    # Define paths
    base_path = Path(__file__).parent.parent.parent
    data_folder = base_path / "data"
    schema_path = base_path / "schemas" / "contracts_schema_hybrid.json"
    output_dir = Path("tests/reports/comprehensive")
    
    try:
        # Build comprehensive evaluation configuration
        config = (EvaluationConfigBuilder()
                 .with_ollama_config(
                     model="llama3.2:latest",
                     base_url="http://localhost:11434",
                     temperature=0.0,
                     max_tokens=4000
                 )
                 .with_all_metrics()
                 .with_data_folder(data_folder)
                 .with_few_shot_extractor(
                     schema_path=schema_path,
                     examples_folder=data_folder / "labels",
                     name="FewShot_Unified",
                     confidence_threshold=0.3,
                     max_attempts=3
                 )
                 # Add more extractors when available
                 # .with_vector_rag_extractor(
                 #     name="VectorRAG_Basic",
                 #     enabled=False
                 # )
                 # .with_graph_rag_extractor(
                 #     name="GraphRAG_Basic", 
                 #     enabled=False
                 # )
                 .with_output_dir(output_dir)
                 .with_timeout(300.0)
                 .with_detailed_logging(True)
                 .build())
        
        logger.info(f"Configuration created with {len(config.extractors)} extractors and {len(config.test_cases)} test cases")
        
        # Run evaluation
        framework = UnifiedEvaluationFramework(config)
        results = framework.run_evaluation()
        
        # Print summary
        print_evaluation_summary(results)
        
        logger.info(f"Evaluation completed. Results saved to {output_dir}")
        return results
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


def run_quick_evaluation():
    """Run quick evaluation for development testing."""
    
    logger.info("Starting quick DocuVerse evaluation")
    
    # Define paths
    base_path = Path(__file__).parent.parent.parent
    data_folder = base_path / "data"
    schema_path = base_path / "schemas" / "contracts_schema_hybrid.json"
    output_dir = Path("tests/reports/quick")
    
    try:
        # Build quick evaluation configuration
        config = (EvaluationConfigBuilder()
                 .with_ollama_config(temperature=0.0)
                 .with_basic_metrics()
                 .with_data_folder(data_folder)
                 .with_few_shot_extractor(
                     schema_path=schema_path,
                     examples_folder=data_folder / "labels",
                     name="FewShot_Quick"
                 )
                 .with_output_dir(output_dir)
                 .with_timeout(60.0)
                 .build())
        
        # Run evaluation
        framework = UnifiedEvaluationFramework(config)
        results = framework.run_evaluation()
        
        # Print summary
        print_evaluation_summary(results)
        
        logger.info(f"Quick evaluation completed. Results saved to {output_dir}")
        return results
        
    except Exception as e:
        logger.error(f"Quick evaluation failed: {e}")
        raise


def run_metric_comparison():
    """Run evaluation focusing on metric comparison."""
    
    logger.info("Starting metric comparison evaluation")
    
    # Define paths
    base_path = Path(__file__).parent.parent.parent
    data_folder = base_path / "data"
    schema_path = base_path / "schemas" / "contracts_schema_hybrid.json"
    output_dir = Path("tests/reports/metrics")
    
    try:
        # Build configuration with focus on different metrics
        config = (EvaluationConfigBuilder()
                 .with_ollama_config(temperature=0.0)
                 .with_metrics(
                     EvaluationMetric.ACCURACY,
                     EvaluationMetric.F1,
                     EvaluationMetric.SEMANTIC_SIMILARITY,
                     EvaluationMetric.ROUGE_L
                 )
                 .with_data_folder(data_folder)
                 .with_few_shot_extractor(
                     schema_path=schema_path,
                     examples_folder=data_folder / "labels",
                     name="FewShot_HighTemp",
                     temperature_override=0.7
                 )
                 .with_few_shot_extractor(
                     schema_path=schema_path,
                     examples_folder=data_folder / "labels",
                     name="FewShot_LowTemp",
                     temperature_override=0.1
                 )
                 .with_output_dir(output_dir)
                 .build())
        
        # Run evaluation
        framework = UnifiedEvaluationFramework(config)
        results = framework.run_evaluation()
        
        # Print summary
        print_evaluation_summary(results)
        
        logger.info(f"Metric comparison completed. Results saved to {output_dir}")
        return results
        
    except Exception as e:
        logger.error(f"Metric comparison failed: {e}")
        raise


def print_evaluation_summary(results):
    """Print a formatted summary of evaluation results."""
    
    print("\n" + "="*80)
    print("DOCUVERSE EVALUATION SUMMARY")
    print("="*80)
    
    summary = results.get("summary", {})
    metadata = results.get("evaluation_metadata", {})
    
    print(f"Evaluation Time: {metadata.get('total_evaluation_time', 0):.2f}s")
    print(f"Framework Version: {metadata.get('framework_version', 'Unknown')}")
    print()
    
    if not summary:
        print("No results available.")
        return
    
    # Overall statistics
    total_tests = sum(s.get("total_tests", 0) for s in summary.values())
    total_successful = sum(s.get("successful_tests", 0) for s in summary.values())
    overall_success_rate = total_successful / total_tests if total_tests > 0 else 0.0
    
    print(f"Overall Statistics:")
    print(f"  Total Tests: {total_tests}")
    print(f"  Successful: {total_successful}")
    print(f"  Success Rate: {overall_success_rate:.1%}")
    print()
    
    # Per-extractor results
    print("Extractor Performance:")
    print("-" * 80)
    
    for extractor_name, extractor_summary in summary.items():
        success_rate = extractor_summary.get("success_rate", 0.0)
        avg_time = extractor_summary.get("average_extraction_time", 0.0)
        avg_confidence = extractor_summary.get("average_confidence", 0.0)
        test_count = extractor_summary.get("total_tests", 0)
        
        print(f"{extractor_name}:")
        print(f"  Tests: {test_count}")
        print(f"  Success Rate: {success_rate:.1%}")
        print(f"  Avg Time: {avg_time:.3f}s")
        print(f"  Avg Confidence: {avg_confidence:.3f}")
        
        # Show top metrics
        metrics_summary = extractor_summary.get("metrics_summary", {})
        if metrics_summary:
            print(f"  Top Metrics:")
            for metric, stats in list(metrics_summary.items())[:3]:
                print(f"    {metric}: {stats['mean']:.3f}")
        
        print()
    
    # Rankings
    comparison = results.get("comparison", {})
    metric_rankings = comparison.get("metric_rankings", {})
    
    if metric_rankings:
        print("Top Performers by Metric:")
        print("-" * 80)
        
        for metric, rankings in list(metric_rankings.items())[:3]:
            if rankings:
                winner = rankings[0]
                print(f"{metric}: {winner[0]} ({winner[1]:.3f})")
        print()
    
    print("="*80)
    print()


def run_custom_evaluation():
    """Example of creating a custom evaluation configuration."""
    
    logger.info("Starting custom evaluation example")
    
    # This demonstrates how to create highly customized evaluations
    base_path = Path(__file__).parent.parent.parent
    data_folder = base_path / "data"
    schema_path = base_path / "schemas" / "contracts_schema_hybrid.json"
    
    # Custom test case
    from unified_evaluation_framework import TestCase
    
    custom_test_case = TestCase(
        name="custom_contract",
        document={
            "content": """
            SERVICE AGREEMENT
            
            This Service Agreement is entered into between TechCorp Ltd. and DataSoft Inc.
            
            Payment Terms: Payment due within 30 days of invoice.
            Warranty: 12-month limited warranty on all software deliverables.
            Customer: TechCorp Ltd., 123 Tech Street, Silicon Valley
            """,
            "metadata": {"type": "service_agreement"}
        },
        ground_truth={
            "payment_terms": {
                "value": "30 days",
                "classification": "standard"
            },
            "warranty": {
                "value": "12-month limited warranty on all software deliverables",
                "classification": "limited"
            },
            "customer_name": "TechCorp Ltd."
        },
        expected_confidence=0.8
    )
    
    try:
        # Build custom configuration
        config = (EvaluationConfigBuilder()
                 .with_ollama_config(
                     temperature=0.2,
                     max_tokens=3000
                 )
                 .with_metrics(
                     EvaluationMetric.ACCURACY,
                     EvaluationMetric.F1,
                     EvaluationMetric.SEMANTIC_SIMILARITY
                 )
                 .with_test_cases(custom_test_case)
                 .with_data_folder(data_folder)  # Also load from data folder
                 .with_few_shot_extractor(
                     schema_path=schema_path,
                     examples_folder=data_folder / "labels",
                     name="FewShot_Custom",
                     confidence_threshold=0.5
                 )
                 .with_output_dir(Path("tests/reports/custom"))
                 .build())
        
        # Run evaluation
        framework = UnifiedEvaluationFramework(config)
        results = framework.run_evaluation()
        
        print_evaluation_summary(results)
        
        logger.info("Custom evaluation completed")
        return results
        
    except Exception as e:
        logger.error(f"Custom evaluation failed: {e}")
        raise


if __name__ == "__main__":
    """Run evaluation based on command line argument."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Run DocuVerse evaluation")
    parser.add_argument(
        "type", 
        choices=["comprehensive", "quick", "metrics", "custom"],
        help="Type of evaluation to run"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Custom output directory"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        if args.type == "comprehensive":
            results = run_comprehensive_evaluation()
        elif args.type == "quick":
            results = run_quick_evaluation()
        elif args.type == "metrics":
            results = run_metric_comparison()
        elif args.type == "custom":
            results = run_custom_evaluation()
        else:
            raise ValueError(f"Unknown evaluation type: {args.type}")
        
        print(f"Evaluation completed successfully!")
        
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Evaluation failed: {e}")
        sys.exit(1)
