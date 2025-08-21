#!/usr/bin/env python3
"""
Comprehensive test and evaluation of the unified FewShotExtractor.

This test suite includes:
- Basic functionality testing
- Performance evaluation with metrics
- Accuracy analysis against ground truth
- Example effectiveness assessment
- Schema compliance validation
- Error analysis and edge cases
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from docuverse.core.config import LLMConfig
from docuverse.extractors.few_shot import FewShotExtractor


class FewShotEvaluator:
    """Evaluator for FewShotExtractor performance and accuracy."""
    
    def __init__(self, extractor: FewShotExtractor):
        self.extractor = extractor
        self.test_results = []
        self.metrics = defaultdict(list)
    
    def evaluate_extraction_accuracy(self, result: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate extraction accuracy against ground truth.
        
        Args:
            result: Extracted data from the model
            ground_truth: Expected ground truth data
            
        Returns:
            Dictionary with accuracy metrics
        """
        metrics = {
            "field_accuracy": 0.0,
            "classification_accuracy": 0.0,
            "extraction_accuracy": 0.0,
            "overall_accuracy": 0.0
        }
        
        result_fields = result.get("fields", {})
        gt_fields = ground_truth.get("fields", {})
        
        if not gt_fields:
            return metrics
        
        total_fields = len(gt_fields)
        correct_fields = 0
        correct_classifications = 0
        correct_extractions = 0
        classification_count = 0
        extraction_count = 0
        
        for field_name, gt_value in gt_fields.items():
            if field_name in result_fields:
                result_value = result_fields[field_name]
                
                # Check if field structure is correct
                if isinstance(gt_value, dict) and isinstance(result_value, dict):
                    field_correct = True
                    
                    # Check extracted content
                    if "extracted_content" in gt_value and "extracted_content" in result_value:
                        extraction_count += 1
                        gt_content = gt_value["extracted_content"].strip().lower()
                        result_content = result_value["extracted_content"].strip().lower()
                        
                        # Use fuzzy matching for extracted content
                        if self._fuzzy_match(gt_content, result_content):
                            correct_extractions += 1
                        else:
                            field_correct = False
                    
                    # Check classification
                    if "classification" in gt_value and "classification" in result_value:
                        classification_count += 1
                        if gt_value["classification"] == result_value["classification"]:
                            correct_classifications += 1
                        else:
                            field_correct = False
                    
                    if field_correct:
                        correct_fields += 1
                
                elif gt_value == result_value:
                    correct_fields += 1
        
        # Calculate metrics
        metrics["field_accuracy"] = correct_fields / total_fields if total_fields > 0 else 0.0
        metrics["classification_accuracy"] = correct_classifications / classification_count if classification_count > 0 else 0.0
        metrics["extraction_accuracy"] = correct_extractions / extraction_count if extraction_count > 0 else 0.0
        metrics["overall_accuracy"] = (correct_fields / total_fields) if total_fields > 0 else 0.0
        
        return metrics
    
    def _fuzzy_match(self, text1: str, text2: str, threshold: float = 0.8) -> bool:
        """Simple fuzzy matching for extracted content."""
        if not text1 or not text2:
            return text1 == text2
        
        # Simple Jaccard similarity
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 and not words2:
            return True
        if not words1 or not words2:
            return False
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return (intersection / union) >= threshold
    
    def evaluate_performance(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate extraction performance including timing and confidence.
        
        Args:
            document: Document to extract from
            
        Returns:
            Performance metrics dictionary
        """
        start_time = time.time()
        
        try:
            result = self.extractor.extract(document)
            end_time = time.time()
            
            performance = {
                "extraction_time": end_time - start_time,
                "confidence": self.extractor.last_confidence,
                "success": True,
                "error": None,
                "result": result
            }
            
            # Validate schema compliance
            validation = self.extractor.validate_schema_compliance(result)
            performance["schema_valid"] = validation["is_valid"]
            performance["validation_errors"] = validation.get("missing_fields", []) + \
                                             validation.get("invalid_enums", []) + \
                                             validation.get("structure_errors", [])
            
            return performance
            
        except Exception as e:
            end_time = time.time()
            return {
                "extraction_time": end_time - start_time,
                "confidence": 0.0,
                "success": False,
                "error": str(e),
                "result": {},
                "schema_valid": False,
                "validation_errors": [f"Extraction failed: {e}"]
            }
    
    def run_evaluation_suite(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run comprehensive evaluation on multiple test cases.
        
        Args:
            test_cases: List of test cases with document and ground_truth
            
        Returns:
            Comprehensive evaluation report
        """
        print("🔬 Running Few-Shot Extractor Evaluation Suite")
        print("=" * 60)
        
        results = []
        all_metrics = defaultdict(list)
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📋 Test Case {i}/{len(test_cases)}: {test_case.get('name', f'case_{i}')}")
            
            # Run performance evaluation
            performance = self.evaluate_performance(test_case["document"])
            
            # Run accuracy evaluation if ground truth available
            accuracy = {}
            if "ground_truth" in test_case and performance["success"]:
                accuracy = self.evaluate_extraction_accuracy(
                    performance["result"], 
                    test_case["ground_truth"]
                )
            
            # Combine results
            test_result = {
                "test_case": test_case.get("name", f"case_{i}"),
                "performance": performance,
                "accuracy": accuracy
            }
            
            results.append(test_result)
            
            # Collect metrics
            for key, value in performance.items():
                if isinstance(value, (int, float)):
                    all_metrics[f"performance_{key}"].append(value)
            
            for key, value in accuracy.items():
                if isinstance(value, (int, float)):
                    all_metrics[f"accuracy_{key}"].append(value)
            
            # Print results
            print(f"  ⏱️  Extraction time: {performance['extraction_time']:.3f}s")
            print(f"  🎯 Confidence: {performance['confidence']:.3f}")
            print(f"  ✅ Success: {'Yes' if performance['success'] else 'No'}")
            print(f"  📏 Schema valid: {'Yes' if performance['schema_valid'] else 'No'}")
            
            if accuracy:
                print(f"  📊 Field accuracy: {accuracy['field_accuracy']:.3f}")
                print(f"  🔍 Classification accuracy: {accuracy['classification_accuracy']:.3f}")
                print(f"  📝 Extraction accuracy: {accuracy['extraction_accuracy']:.3f}")
            
            if performance.get("validation_errors"):
                print(f"  ⚠️  Validation errors: {len(performance['validation_errors'])}")
        
        # Calculate summary statistics
        summary = self._calculate_summary_stats(all_metrics, results)
        
        return {
            "test_results": results,
            "summary": summary,
            "extractor_info": self._get_extractor_info()
        }
    
    def _calculate_summary_stats(self, all_metrics: Dict[str, List], results: List) -> Dict[str, Any]:
        """Calculate summary statistics from all test results."""
        summary = {}
        
        # Calculate averages and ranges for numeric metrics
        for metric_name, values in all_metrics.items():
            if values:
                summary[metric_name] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }
        
        # Calculate success rates
        total_tests = len(results)
        successful_extractions = sum(1 for r in results if r["performance"]["success"])
        valid_schemas = sum(1 for r in results if r["performance"]["schema_valid"])
        
        summary["success_rate"] = successful_extractions / total_tests if total_tests > 0 else 0.0
        summary["schema_validity_rate"] = valid_schemas / total_tests if total_tests > 0 else 0.0
        summary["total_tests"] = total_tests
        
        return summary
    
    def _get_extractor_info(self) -> Dict[str, Any]:
        """Get information about the extractor configuration."""
        return {
            "examples_count": len(self.extractor.examples),
            "schema_fields": len(self.extractor.schema.get("field", {})) if self.extractor.schema else 0,
            "example_sources": [ex.get("metadata", {}).get("base_name", "unknown") 
                              for ex in self.extractor.examples],
            "field_analysis": self.extractor.get_field_analysis(),
            "example_summary": self.extractor.get_example_summary()
        }
    
    def generate_report(self, evaluation_results: Dict[str, Any], output_path: str = None) -> str:
        """Generate a comprehensive evaluation report."""
        report_lines = [
            "# Few-Shot Extractor Evaluation Report",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary Statistics",
            ""
        ]
        
        summary = evaluation_results["summary"]
        
        # Overall performance
        report_lines.extend([
            f"- **Total Tests**: {summary['total_tests']}",
            f"- **Success Rate**: {summary['success_rate']:.1%}",
            f"- **Schema Validity Rate**: {summary['schema_validity_rate']:.1%}",
            ""
        ])
        
        # Performance metrics
        if "performance_extraction_time" in summary:
            time_stats = summary["performance_extraction_time"]
            report_lines.extend([
                "### Performance Metrics",
                f"- **Average Extraction Time**: {time_stats['mean']:.3f}s",
                f"- **Min/Max Time**: {time_stats['min']:.3f}s / {time_stats['max']:.3f}s",
                ""
            ])
        
        if "performance_confidence" in summary:
            conf_stats = summary["performance_confidence"]
            report_lines.extend([
                f"- **Average Confidence**: {conf_stats['mean']:.3f}",
                f"- **Min/Max Confidence**: {conf_stats['min']:.3f} / {conf_stats['max']:.3f}",
                ""
            ])
        
        # Accuracy metrics
        accuracy_metrics = ["accuracy_field_accuracy", "accuracy_classification_accuracy", "accuracy_extraction_accuracy"]
        has_accuracy = any(metric in summary for metric in accuracy_metrics)
        
        if has_accuracy:
            report_lines.extend(["### Accuracy Metrics", ""])
            
            for metric in accuracy_metrics:
                if metric in summary:
                    stats = summary[metric]
                    metric_name = metric.replace("accuracy_", "").replace("_", " ").title()
                    report_lines.append(f"- **{metric_name}**: {stats['mean']:.1%} (min: {stats['min']:.1%}, max: {stats['max']:.1%})")
            
            report_lines.append("")
        
        # Extractor information
        extractor_info = evaluation_results["extractor_info"]
        report_lines.extend([
            "## Extractor Configuration",
            "",
            f"- **Examples Count**: {extractor_info['examples_count']}",
            f"- **Schema Fields**: {extractor_info['schema_fields']}",
            f"- **Example Sources**: {', '.join(extractor_info['example_sources'])}",
            ""
        ])
        
        # Field analysis
        field_analysis = extractor_info["field_analysis"]
        if field_analysis["hybrid_fields"]:
            report_lines.extend([
                "### Hybrid Fields (Extract + Classify)",
                ""
            ])
            for field in field_analysis["hybrid_fields"]:
                report_lines.append(f"- **{field['name']}**: {field['enum_options']}")
            report_lines.append("")
        
        if field_analysis["extraction_fields"]:
            report_lines.extend([
                "### Pure Extraction Fields",
                ""
            ])
            for field in field_analysis["extraction_fields"]:
                report_lines.append(f"- **{field['name']}**: {field['description']}")
            report_lines.append("")
        
        # Individual test results
        report_lines.extend([
            "## Individual Test Results",
            ""
        ])
        
        for result in evaluation_results["test_results"]:
            test_name = result["test_case"]
            perf = result["performance"]
            acc = result["accuracy"]
            
            report_lines.extend([
                f"### {test_name}",
                "",
                f"- **Extraction Time**: {perf['extraction_time']:.3f}s",
                f"- **Confidence**: {perf['confidence']:.3f}",
                f"- **Success**: {'✅' if perf['success'] else '❌'}",
                f"- **Schema Valid**: {'✅' if perf['schema_valid'] else '❌'}",
            ])
            
            if acc:
                report_lines.extend([
                    f"- **Field Accuracy**: {acc['field_accuracy']:.1%}",
                    f"- **Classification Accuracy**: {acc['classification_accuracy']:.1%}",
                    f"- **Extraction Accuracy**: {acc['extraction_accuracy']:.1%}",
                ])
            
            if perf.get("validation_errors"):
                report_lines.append(f"- **Validation Errors**: {len(perf['validation_errors'])}")
            
            report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"📄 Report saved to: {output_path}")
        
        return report_text


def test_few_shot_extractor_with_evaluation():
    """Main test function with comprehensive evaluation."""
    
    print("🧪 Few-Shot Extractor Test & Evaluation Suite")
    print("=" * 60)
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    schema_path = project_root / "schemas" / "contracts_schema_hybrid.json"
    data_path = project_root / "data"
    
    # Load schema
    print("\n1. 📋 Loading Schema...")
    try:
        with open(schema_path, 'r') as f:
            schema = json.load(f)
        print(f"   ✅ Loaded schema with {len(schema.get('field', {}))} fields")
    except Exception as e:
        print(f"   ❌ Schema loading failed: {e}")
        return
    
    # Configure LLM
    print("\n2. 🤖 Configuring LLM...")
    llm_config = LLMConfig(
        provider="ollama",
        model_name="llama3.2:latest",
        ollama_base_url="http://localhost:11434",
        temperature=0.1
    )
    print(f"   ✅ LLM configured: {llm_config.provider} - {llm_config.model_name}")
    
    # Initialize extractor
    print("\n3. 🚀 Initializing Extractor...")
    try:
        extractor = FewShotExtractor(
            llm_config=llm_config,
            schema=schema,
            auto_load_labels=True
        )
        print(f"   ✅ Extractor initialized with {len(extractor.examples)} examples")
    except Exception as e:
        print(f"   ❌ Extractor initialization failed: {e}")
        return
    
    # Prepare test cases
    print("\n4. 📄 Preparing Test Cases...")
    test_cases = []
    
    # Test case 1: Real contract document
    contract_path = data_path / "contract1.txt"
    labels_path = data_path / "labels" / "contract1_label.json"
    
    if contract_path.exists():
        with open(contract_path, 'r') as f:
            contract_content = f.read()
        
        ground_truth = None
        if labels_path.exists():
            with open(labels_path, 'r') as f:
                ground_truth = json.load(f)
        
        test_cases.append({
            "name": "Contract1_RealData",
            "document": {
                "content": contract_content,
                "metadata": {"filename": "contract1.txt", "source": "real_data"}
            },
            "ground_truth": ground_truth
        })
        
        print(f"   ✅ Added real contract test case (length: {len(contract_content)} chars)")
    
    # Test case 2: Synthetic contract document
    synthetic_contract = """
    PROFESSIONAL SERVICES AGREEMENT
    
    This agreement is between TechSolutions LLC and Jane Smith.
    
    PAYMENT TERMS:
    Payment is due once a year on December 31st. All payments must be made via wire transfer.
    
    WARRANTY INFORMATION:
    This contract includes a non-standard warranty period of 2 years covering all services.
    
    CLIENT DETAILS:
    Client Name: Jane Smith
    Client Company: Smith Enterprises
    Contact Email: jane@smithenterprises.com
    
    This agreement is effective immediately upon signing.
    """
    
    synthetic_ground_truth = {
        "fields": {
            "payment_terms": {
                "extracted_content": "Payment is due once a year on December 31st",
                "classification": "yearly"
            },
            "warranty": {
                "extracted_content": "non-standard warranty period of 2 years",
                "classification": "non_standard"
            },
            "customer_name": {
                "extracted_content": "Jane Smith"
            }
        }
    }
    
    test_cases.append({
        "name": "Contract2_Synthetic",
        "document": {
            "content": synthetic_contract,
            "metadata": {"filename": "synthetic_contract.txt", "source": "synthetic"}
        },
        "ground_truth": synthetic_ground_truth
    })
    
    print(f"   ✅ Added synthetic contract test case")
    
    # Test case 3: Edge case - minimal contract
    minimal_contract = """
    Contract between Company and John Doe.
    One-time payment required.
    Standard terms apply.
    """
    
    minimal_ground_truth = {
        "fields": {
            "payment_terms": {
                "extracted_content": "One-time payment required",
                "classification": "one-time"
            },
            "warranty": {
                "extracted_content": "Standard terms apply",
                "classification": "standard"
            },
            "customer_name": {
                "extracted_content": "John Doe"
            }
        }
    }
    
    test_cases.append({
        "name": "Contract3_Minimal",
        "document": {
            "content": minimal_contract,
            "metadata": {"filename": "minimal_contract.txt", "source": "edge_case"}
        },
        "ground_truth": minimal_ground_truth
    })
    
    print(f"   ✅ Added minimal contract test case")
    print(f"   📊 Total test cases: {len(test_cases)}")
    
    # Run evaluation
    print("\n5. 🔬 Running Evaluation...")
    evaluator = FewShotEvaluator(extractor)
    evaluation_results = evaluator.run_evaluation_suite(test_cases)
    
    # Generate report
    print("\n6. 📄 Generating Report...")
    report_path = Path(__file__).parent / "few_shot_evaluation_report.md"
    report = evaluator.generate_report(evaluation_results, str(report_path))
    
    # Print summary
    print("\n" + "=" * 60)
    print("🎉 EVALUATION COMPLETE!")
    print("=" * 60)
    
    summary = evaluation_results["summary"]
    print(f"📊 Overall Results:")
    print(f"   • Total tests: {summary['total_tests']}")
    print(f"   • Success rate: {summary['success_rate']:.1%}")
    print(f"   • Schema validity: {summary['schema_validity_rate']:.1%}")
    
    if "performance_extraction_time" in summary:
        time_stats = summary["performance_extraction_time"]
        print(f"   • Avg extraction time: {time_stats['mean']:.3f}s")
    
    if "accuracy_field_accuracy" in summary:
        acc_stats = summary["accuracy_field_accuracy"]
        print(f"   • Avg field accuracy: {acc_stats['mean']:.1%}")
    
    print(f"\n📄 Detailed report: {report_path}")
    
    return evaluation_results


if __name__ == "__main__":
    test_few_shot_extractor_with_evaluation()
