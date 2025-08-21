#!/usr/bin/env python3
"""
Comprehensive Test Runner for DocuVerse

This script orchestrates all types of tests including unit tests, integration tests,
evaluation tests, and benchmarks for the DocuVerse extraction framework.
"""

import sys
import os
import unittest
import argparse
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class DocuVerseTestRunner:
    """Comprehensive test runner for DocuVerse framework."""
    
    def __init__(self, base_path: Path):
        """Initialize test runner."""
        self.base_path = Path(base_path)
        self.tests_path = self.base_path / "tests"
        self.src_path = self.base_path / "src"
        
        # Test directories
        self.unit_tests_path = self.tests_path / "unit"
        self.integration_tests_path = self.tests_path / "integration"
        self.evaluation_tests_path = self.tests_path / "evaluation"
        self.benchmark_tests_path = self.tests_path / "benchmarks"
        self.reports_path = self.tests_path / "reports"
        
        # Ensure all directories exist
        for path in [self.unit_tests_path, self.integration_tests_path, 
                     self.evaluation_tests_path, self.benchmark_tests_path, 
                     self.reports_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.results = {}
    
    def run_unit_tests(self, pattern: str = "test_*.py") -> Dict[str, Any]:
        """Run unit tests."""
        logger.info("Running unit tests...")
        
        # Discover and run unit tests
        loader = unittest.TestLoader()
        start_dir = str(self.unit_tests_path)
        
        try:
            suite = loader.discover(start_dir, pattern=pattern)
            runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
            
            start_time = time.time()
            result = runner.run(suite)
            end_time = time.time()
            
            unit_results = {
                "type": "unit_tests",
                "tests_run": result.testsRun,
                "failures": len(result.failures),
                "errors": len(result.errors),
                "skipped": len(result.skipped) if hasattr(result, 'skipped') else 0,
                "success_rate": (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0.0,
                "duration": end_time - start_time,
                "details": {
                    "failures": [{"test": str(test), "error": error} for test, error in result.failures],
                    "errors": [{"test": str(test), "error": error} for test, error in result.errors]
                }
            }
            
            self.results["unit_tests"] = unit_results
            logger.info(f"Unit tests completed: {unit_results['tests_run']} tests, "
                       f"{unit_results['success_rate']:.1%} success rate")
            
            return unit_results
            
        except Exception as e:
            logger.error(f"Failed to run unit tests: {e}")
            error_result = {
                "type": "unit_tests",
                "error": str(e),
                "tests_run": 0,
                "success_rate": 0.0
            }
            self.results["unit_tests"] = error_result
            return error_result
    
    def run_integration_tests(self, pattern: str = "test_*.py") -> Dict[str, Any]:
        """Run integration tests."""
        logger.info("Running integration tests...")
        
        # Add integration tests to path
        sys.path.insert(0, str(self.integration_tests_path))
        
        try:
            loader = unittest.TestLoader()
            start_dir = str(self.integration_tests_path)
            
            suite = loader.discover(start_dir, pattern=pattern)
            runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
            
            start_time = time.time()
            result = runner.run(suite)
            end_time = time.time()
            
            integration_results = {
                "type": "integration_tests",
                "tests_run": result.testsRun,
                "failures": len(result.failures),
                "errors": len(result.errors),
                "skipped": len(result.skipped) if hasattr(result, 'skipped') else 0,
                "success_rate": (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0.0,
                "duration": end_time - start_time,
                "details": {
                    "failures": [{"test": str(test), "error": error} for test, error in result.failures],
                    "errors": [{"test": str(test), "error": error} for test, error in result.errors]
                }
            }
            
            self.results["integration_tests"] = integration_results
            logger.info(f"Integration tests completed: {integration_results['tests_run']} tests, "
                       f"{integration_results['success_rate']:.1%} success rate")
            
            return integration_results
            
        except Exception as e:
            logger.error(f"Failed to run integration tests: {e}")
            error_result = {
                "type": "integration_tests",
                "error": str(e),
                "tests_run": 0,
                "success_rate": 0.0
            }
            self.results["integration_tests"] = error_result
            return error_result
        finally:
            # Remove from path
            if str(self.integration_tests_path) in sys.path:
                sys.path.remove(str(self.integration_tests_path))
    
    def run_evaluation_tests(self, evaluation_type: str = "quick") -> Dict[str, Any]:
        """Run evaluation tests."""
        logger.info(f"Running {evaluation_type} evaluation tests...")
        
        try:
            # Add evaluation tests to path
            sys.path.insert(0, str(self.evaluation_tests_path))
            
            # Import evaluation runner
            from run_evaluation import (
                run_quick_evaluation, 
                run_comprehensive_evaluation,
                run_metric_comparison
            )
            
            start_time = time.time()
            
            if evaluation_type == "quick":
                results = run_quick_evaluation()
            elif evaluation_type == "comprehensive":
                results = run_comprehensive_evaluation()
            elif evaluation_type == "metrics":
                results = run_metric_comparison()
            else:
                raise ValueError(f"Unknown evaluation type: {evaluation_type}")
            
            end_time = time.time()
            
            # Extract summary information
            summary = results.get("summary", {})
            total_tests = sum(s.get("total_tests", 0) for s in summary.values())
            total_successful = sum(s.get("successful_tests", 0) for s in summary.values())
            overall_success_rate = total_successful / total_tests if total_tests > 0 else 0.0
            
            evaluation_results = {
                "type": f"evaluation_{evaluation_type}",
                "extractors_tested": len(summary),
                "total_tests": total_tests,
                "successful_tests": total_successful,
                "success_rate": overall_success_rate,
                "duration": end_time - start_time,
                "evaluation_metadata": results.get("evaluation_metadata", {}),
                "summary": summary
            }
            
            self.results[f"evaluation_{evaluation_type}"] = evaluation_results
            logger.info(f"Evaluation tests completed: {evaluation_results['total_tests']} tests, "
                       f"{evaluation_results['success_rate']:.1%} success rate")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Failed to run evaluation tests: {e}")
            error_result = {
                "type": f"evaluation_{evaluation_type}",
                "error": str(e),
                "extractors_tested": 0,
                "success_rate": 0.0
            }
            self.results[f"evaluation_{evaluation_type}"] = error_result
            return error_result
        finally:
            # Remove from path
            if str(self.evaluation_tests_path) in sys.path:
                sys.path.remove(str(self.evaluation_tests_path))
    
    def run_benchmarks(self, quick: bool = True) -> Dict[str, Any]:
        """Run performance benchmarks."""
        logger.info("Running performance benchmarks...")
        
        try:
            # Add benchmark tests to path
            sys.path.insert(0, str(self.benchmark_tests_path))
            
            from performance_benchmark import PerformanceBenchmark
            
            # Initialize benchmark
            benchmark = PerformanceBenchmark(
                output_dir=self.reports_path / "benchmarks",
                timeout_seconds=60.0 if quick else 300.0
            )
            
            # Create test cases for benchmarking
            test_cases = self._create_benchmark_test_cases(quick=quick)
            
            start_time = time.time()
            
            # Benchmark available extractors
            try:
                from docuverse.extractors.few_shot import FewShotExtractor
                from docuverse.core.config import LLMConfig, LLMProvider
                
                # Create LLM config
                llm_config = LLMConfig(
                    provider=LLMProvider.OLLAMA,
                    model="llama3.2:latest",
                    ollama_base_url="http://localhost:11434",
                    temperature=0.0
                )
                
                # Create FewShot extractor
                few_shot_extractor = FewShotExtractor(
                    llm_config=llm_config,
                    schema_path=str(self.base_path / "schemas" / "contracts_schema_hybrid.json"),
                    examples_folder=str(self.base_path / "data" / "labels")
                )
                
                # Benchmark FewShot
                result = benchmark.benchmark_extractor(
                    extractor=few_shot_extractor,
                    test_cases=test_cases,
                    method_name="FewShot",
                    extraction_method="FEW_SHOT",
                    config={"schema_path": "contracts_schema_hybrid.json"}
                )
                
            except Exception as e:
                logger.warning(f"Could not benchmark FewShot extractor: {e}")
            
            end_time = time.time()
            
            # Generate comparison
            comparison = benchmark.compare_methods()
            
            # Save benchmark results
            benchmark.save_results("benchmark_results.json")
            
            # Save benchmark report
            report = benchmark.generate_report()
            report_path = self.reports_path / "benchmarks" / "benchmark_report.md"
            with open(report_path, 'w') as f:
                f.write(report)
            
            benchmark_results = {
                "type": "benchmarks",
                "methods_benchmarked": len(benchmark.results),
                "test_cases_per_method": len(test_cases),
                "duration": end_time - start_time,
                "comparison": comparison,
                "results_summary": [
                    {
                        "method": result.method_name,
                        "success_rate": result.performance_metrics.success_rate,
                        "avg_time": result.performance_metrics.mean_extraction_time,
                        "throughput": result.performance_metrics.extractions_per_second
                    }
                    for result in benchmark.results
                ]
            }
            
            self.results["benchmarks"] = benchmark_results
            logger.info(f"Benchmarks completed: {benchmark_results['methods_benchmarked']} methods tested")
            
            return benchmark_results
            
        except Exception as e:
            logger.error(f"Failed to run benchmarks: {e}")
            error_result = {
                "type": "benchmarks",
                "error": str(e),
                "methods_benchmarked": 0
            }
            self.results["benchmarks"] = error_result
            return error_result
        finally:
            # Remove from path
            if str(self.benchmark_tests_path) in sys.path:
                sys.path.remove(str(self.benchmark_tests_path))
    
    def _create_benchmark_test_cases(self, quick: bool = True) -> List[Dict[str, Any]]:
        """Create test cases for benchmarking."""
        
        # Load actual test cases if available
        data_folder = self.base_path / "data"
        labels_folder = data_folder / "labels"
        
        test_cases = []
        
        if labels_folder.exists():
            # Load from labels folder
            for label_file in labels_folder.glob("*.json"):
                base_name = label_file.stem.replace('_label', '').replace('.label', '')
                
                # Find corresponding document
                document_file = None
                for ext in ['.txt', '.md']:
                    potential_doc = data_folder / f"{base_name}{ext}"
                    if potential_doc.exists():
                        document_file = potential_doc
                        break
                
                if document_file:
                    with open(document_file, 'r') as f:
                        content = f.read()
                    
                    test_cases.append({
                        "content": content,
                        "metadata": {"filename": document_file.name}
                    })
        
        # Add synthetic test cases if needed
        if not test_cases or len(test_cases) < 3:
            synthetic_cases = [
                {
                    "content": "Contract between Company A and Company B. Payment due in 30 days. 1-year warranty.",
                    "metadata": {"filename": "synthetic_1.txt"}
                },
                {
                    "content": "Service Agreement. Customer: TechCorp. Payment: 45 days. Warranty: Limited 6 months.",
                    "metadata": {"filename": "synthetic_2.txt"}
                },
                {
                    "content": "Purchase Order. Buyer: DataSoft Inc. Payment terms: Net 15. Full warranty coverage.",
                    "metadata": {"filename": "synthetic_3.txt"}
                }
            ]
            test_cases.extend(synthetic_cases)
        
        # Limit test cases for quick benchmarks
        if quick and len(test_cases) > 5:
            test_cases = test_cases[:5]
        
        return test_cases
    
    def run_all_tests(
        self, 
        include_benchmarks: bool = False,
        evaluation_type: str = "quick"
    ) -> Dict[str, Any]:
        """Run all test suites."""
        logger.info("Running complete test suite...")
        
        start_time = time.time()
        
        # Run unit tests
        self.run_unit_tests()
        
        # Run integration tests
        self.run_integration_tests()
        
        # Run evaluation tests
        self.run_evaluation_tests(evaluation_type)
        
        # Run benchmarks if requested
        if include_benchmarks:
            self.run_benchmarks(quick=(evaluation_type == "quick"))
        
        end_time = time.time()
        
        # Calculate overall results
        overall_results = {
            "test_suite_duration": end_time - start_time,
            "timestamp": time.time(),
            "test_results": self.results,
            "summary": self._calculate_overall_summary()
        }
        
        # Save complete results
        self.save_results(overall_results)
        
        # Generate comprehensive report
        self.generate_comprehensive_report(overall_results)
        
        logger.info(f"Complete test suite finished in {end_time - start_time:.2f} seconds")
        
        return overall_results
    
    def _calculate_overall_summary(self) -> Dict[str, Any]:
        """Calculate overall test summary."""
        summary = {
            "total_test_suites": len(self.results),
            "successful_suites": 0,
            "total_tests": 0,
            "total_successful_tests": 0,
            "overall_success_rate": 0.0,
            "test_breakdown": {}
        }
        
        for test_type, result in self.results.items():
            if "error" not in result:
                summary["successful_suites"] += 1
            
            # Count tests based on test type
            if test_type in ["unit_tests", "integration_tests"]:
                tests_run = result.get("tests_run", 0)
                failures = result.get("failures", 0)
                errors = result.get("errors", 0)
                successful = tests_run - failures - errors
                
                summary["total_tests"] += tests_run
                summary["total_successful_tests"] += successful
                
            elif test_type.startswith("evaluation_"):
                total_tests = result.get("total_tests", 0)
                successful_tests = result.get("successful_tests", 0)
                
                summary["total_tests"] += total_tests
                summary["total_successful_tests"] += successful_tests
            
            summary["test_breakdown"][test_type] = {
                "success": "error" not in result,
                "details": result
            }
        
        if summary["total_tests"] > 0:
            summary["overall_success_rate"] = summary["total_successful_tests"] / summary["total_tests"]
        
        return summary
    
    def save_results(self, results: Dict[str, Any], filename: str = "comprehensive_test_results.json"):
        """Save test results to file."""
        import json
        
        output_path = self.reports_path / filename
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Test results saved to {output_path}")
    
    def generate_comprehensive_report(self, results: Dict[str, Any]):
        """Generate comprehensive test report."""
        
        summary = results.get("summary", {})
        
        report_lines = [
            "# DocuVerse Comprehensive Test Report",
            "",
            f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Total Duration**: {results.get('test_suite_duration', 0):.2f}s",
            "",
            "## Executive Summary",
            "",
            f"- **Test Suites Run**: {summary.get('total_test_suites', 0)}",
            f"- **Successful Suites**: {summary.get('successful_suites', 0)}",
            f"- **Total Tests**: {summary.get('total_tests', 0)}",
            f"- **Overall Success Rate**: {summary.get('overall_success_rate', 0):.1%}",
            "",
            "## Test Suite Results",
            "",
        ]
        
        # Add results for each test suite
        test_breakdown = summary.get("test_breakdown", {})
        
        for test_type, test_info in test_breakdown.items():
            details = test_info.get("details", {})
            success = test_info.get("success", False)
            
            status = "✅ PASSED" if success else "❌ FAILED"
            
            report_lines.extend([
                f"### {test_type.title().replace('_', ' ')} {status}",
                "",
            ])
            
            if test_type in ["unit_tests", "integration_tests"]:
                tests_run = details.get("tests_run", 0)
                failures = details.get("failures", 0)
                errors = details.get("errors", 0)
                success_rate = details.get("success_rate", 0)
                duration = details.get("duration", 0)
                
                report_lines.extend([
                    f"- **Tests Run**: {tests_run}",
                    f"- **Failures**: {failures}",
                    f"- **Errors**: {errors}",
                    f"- **Success Rate**: {success_rate:.1%}",
                    f"- **Duration**: {duration:.2f}s",
                    "",
                ])
                
            elif test_type.startswith("evaluation_"):
                extractors = details.get("extractors_tested", 0)
                total_tests = details.get("total_tests", 0)
                success_rate = details.get("success_rate", 0)
                duration = details.get("duration", 0)
                
                report_lines.extend([
                    f"- **Extractors Tested**: {extractors}",
                    f"- **Total Tests**: {total_tests}",
                    f"- **Success Rate**: {success_rate:.1%}",
                    f"- **Duration**: {duration:.2f}s",
                    "",
                ])
                
            elif test_type == "benchmarks":
                methods = details.get("methods_benchmarked", 0)
                test_cases = details.get("test_cases_per_method", 0)
                duration = details.get("duration", 0)
                
                report_lines.extend([
                    f"- **Methods Benchmarked**: {methods}",
                    f"- **Test Cases per Method**: {test_cases}",
                    f"- **Duration**: {duration:.2f}s",
                    "",
                ])
        
        # Save report
        report_path = self.reports_path / "comprehensive_test_report.md"
        with open(report_path, 'w') as f:
            f.write("\n".join(report_lines))
        
        logger.info(f"Comprehensive report saved to {report_path}")


def main():
    """Main test runner entry point."""
    
    parser = argparse.ArgumentParser(description="Run DocuVerse test suite")
    parser.add_argument(
        "test_type",
        choices=["unit", "integration", "evaluation", "benchmarks", "all"],
        help="Type of tests to run"
    )
    parser.add_argument(
        "--evaluation-type",
        choices=["quick", "comprehensive", "metrics"],
        default="quick",
        help="Type of evaluation to run"
    )
    parser.add_argument(
        "--include-benchmarks",
        action="store_true",
        help="Include benchmarks in 'all' tests"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        default=Path(__file__).parent.parent,
        help="Base path for the project"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize test runner
    runner = DocuVerseTestRunner(args.base_path)
    
    try:
        if args.test_type == "unit":
            results = runner.run_unit_tests()
        elif args.test_type == "integration":
            results = runner.run_integration_tests()
        elif args.test_type == "evaluation":
            results = runner.run_evaluation_tests(args.evaluation_type)
        elif args.test_type == "benchmarks":
            results = runner.run_benchmarks(quick=(args.evaluation_type == "quick"))
        elif args.test_type == "all":
            results = runner.run_all_tests(
                include_benchmarks=args.include_benchmarks,
                evaluation_type=args.evaluation_type
            )
        else:
            raise ValueError(f"Unknown test type: {args.test_type}")
        
        # Print summary
        if args.test_type == "all":
            summary = results.get("summary", {})
            print(f"\n🎉 Test suite completed!")
            print(f"   Total tests: {summary.get('total_tests', 0)}")
            print(f"   Success rate: {summary.get('overall_success_rate', 0):.1%}")
            print(f"   Duration: {results.get('test_suite_duration', 0):.2f}s")
        else:
            success_rate = results.get("success_rate", 0)
            print(f"\n✅ {args.test_type.title()} tests completed: {success_rate:.1%} success rate")
        
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Tests failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
