"""
Integration Tests for DocuVerse Unified Evaluation Framework

This module contains integration tests that verify the evaluation framework
works correctly across different extraction methods and configurations.
"""

import unittest
import tempfile
import json
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

# Add evaluation modules to path
sys.path.append(str(Path(__file__).parent.parent / "evaluation"))

from unified_evaluation_framework import (
    UnifiedEvaluationFramework,
    TestCase,
    ExtractorConfig,
    EvaluationConfig,
    ExtractorFactory
)
from evaluation_config import EvaluationConfigBuilder
from docuverse.core.config import LLMConfig, EvaluationMetric, ExtractionMethod, LLMProvider


class TestUnifiedEvaluationFramework(unittest.TestCase):
    """Test the unified evaluation framework."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.data_dir = self.temp_dir / "data"
        self.labels_dir = self.data_dir / "labels"
        self.schemas_dir = self.temp_dir / "schemas"
        self.output_dir = self.temp_dir / "output"
        
        # Create directories
        self.data_dir.mkdir(parents=True)
        self.labels_dir.mkdir(parents=True)
        self.schemas_dir.mkdir(parents=True)
        self.output_dir.mkdir(parents=True)
        
        # Create test data
        self._create_test_data()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def _create_test_data(self):
        """Create test data files."""
        
        # Create test document
        test_doc = """
        SERVICE AGREEMENT
        
        This agreement is between TestCorp and ClientCorp.
        Payment due within 30 days.
        Warranty: 1 year limited warranty.
        """
        
        (self.data_dir / "test_contract.txt").write_text(test_doc)
        
        # Create test label
        test_label = {
            "payment_terms": {
                "value": "30 days",
                "classification": "standard"
            },
            "warranty": {
                "value": "1 year limited warranty",
                "classification": "limited"
            },
            "customer_name": "ClientCorp"
        }
        
        with open(self.labels_dir / "test_contract_label.json", 'w') as f:
            json.dump(test_label, f, indent=2)
        
        # Create test schema
        test_schema = {
            "type": "object",
            "properties": {
                "payment_terms": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"},
                        "classification": {
                            "type": "string",
                            "enum": ["standard", "expedited", "extended"]
                        }
                    },
                    "hybrid": True
                },
                "warranty": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"},
                        "classification": {
                            "type": "string",
                            "enum": ["limited", "full", "none"]
                        }
                    },
                    "hybrid": True
                },
                "customer_name": {
                    "type": "string",
                    "hybrid": False
                }
            }
        }
        
        with open(self.schemas_dir / "test_schema.json", 'w') as f:
            json.dump(test_schema, f, indent=2)
    
    def test_config_builder(self):
        """Test the evaluation config builder."""
        
        config = (EvaluationConfigBuilder()
                 .with_ollama_config()
                 .with_basic_metrics()
                 .with_data_folder(self.data_dir)
                 .with_few_shot_extractor(
                     schema_path=self.schemas_dir / "test_schema.json",
                     examples_folder=self.labels_dir
                 )
                 .with_output_dir(self.output_dir)
                 .build())
        
        self.assertIsInstance(config, EvaluationConfig)
        self.assertEqual(len(config.test_cases), 1)
        self.assertEqual(len(config.extractors), 1)
        self.assertEqual(len(config.metrics), 4)  # Basic metrics
        self.assertEqual(config.extractors[0].method, ExtractionMethod.FEW_SHOT)
    
    def test_test_case_creation_from_data_folder(self):
        """Test automatic test case creation from data folder."""
        
        from unified_evaluation_framework import create_test_cases_from_data_folder
        
        test_cases = create_test_cases_from_data_folder(self.data_dir)
        
        self.assertEqual(len(test_cases), 1)
        self.assertEqual(test_cases[0].name, "test_contract")
        self.assertIn("SERVICE AGREEMENT", test_cases[0].document["content"])
        self.assertIsNotNone(test_cases[0].ground_truth)
        self.assertEqual(test_cases[0].ground_truth["customer_name"], "ClientCorp")
    
    @patch('docuverse.utils.llm_client.LLMClient')
    def test_framework_initialization(self, mock_llm_client):
        """Test framework initialization."""
        
        # Mock LLM client
        mock_client = MagicMock()
        mock_llm_client.return_value = mock_client
        
        config = (EvaluationConfigBuilder()
                 .with_ollama_config()
                 .with_basic_metrics()
                 .with_data_folder(self.data_dir)
                 .with_few_shot_extractor(
                     schema_path=self.schemas_dir / "test_schema.json",
                     examples_folder=self.labels_dir
                 )
                 .with_output_dir(self.output_dir)
                 .build())
        
        framework = UnifiedEvaluationFramework(config)
        
        self.assertIsInstance(framework, UnifiedEvaluationFramework)
        self.assertEqual(framework.config, config)
        self.assertTrue(self.output_dir.exists())
    
    @patch('docuverse.utils.llm_client.LLMClient')
    def test_extractor_factory(self, mock_llm_client):
        """Test extractor factory."""
        
        # Mock LLM client
        mock_client = MagicMock()
        mock_llm_client.return_value = mock_client
        
        llm_config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="test-model"
        )
        
        extractor = ExtractorFactory.create_extractor(
            ExtractionMethod.FEW_SHOT,
            llm_config,
            {
                "schema_path": str(self.schemas_dir / "test_schema.json"),
                "examples_folder": str(self.labels_dir)
            }
        )
        
        self.assertIsNotNone(extractor)
    
    def test_unsupported_extractor_method(self):
        """Test factory with unsupported extraction method."""
        
        llm_config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="test-model"
        )
        
        # VectorRAG not implemented yet
        with self.assertRaises(NotImplementedError):
            ExtractorFactory.create_extractor(
                ExtractionMethod.VECTOR_RAG,
                llm_config,
                {}
            )
    
    @patch('docuverse.utils.llm_client.LLMClient')
    def test_evaluation_run_mock(self, mock_llm_client):
        """Test evaluation run with mocked LLM client."""
        
        # Mock LLM client responses
        mock_client = MagicMock()
        mock_response = {
            "payment_terms": {
                "value": "30 days",
                "classification": "standard"
            },
            "warranty": {
                "value": "1 year limited warranty",
                "classification": "limited"
            },
            "customer_name": "ClientCorp"
        }
        
        mock_client.chat.return_value = json.dumps(mock_response)
        mock_llm_client.return_value = mock_client
        
        config = (EvaluationConfigBuilder()
                 .with_ollama_config()
                 .with_basic_metrics()
                 .with_data_folder(self.data_dir)
                 .with_few_shot_extractor(
                     schema_path=self.schemas_dir / "test_schema.json",
                     examples_folder=self.labels_dir
                 )
                 .with_output_dir(self.output_dir)
                 .build())
        
        framework = UnifiedEvaluationFramework(config)
        results = framework.run_evaluation()
        
        # Check results structure
        self.assertIn("summary", results)
        self.assertIn("comparison", results)
        self.assertIn("individual_results", results)
        self.assertIn("evaluation_metadata", results)
        
        # Check that evaluation ran
        self.assertGreater(len(results["individual_results"]), 0)
        
        # Check that output files were created
        self.assertTrue((self.output_dir / "evaluation_results.json").exists())
        self.assertTrue((self.output_dir / "evaluation_report.md").exists())
    
    def test_config_validation(self):
        """Test configuration validation."""
        
        # Missing LLM config
        with self.assertRaises(ValueError):
            (EvaluationConfigBuilder()
             .with_basic_metrics()
             .with_data_folder(self.data_dir)
             .build())
        
        # Missing test cases
        with self.assertRaises(ValueError):
            (EvaluationConfigBuilder()
             .with_ollama_config()
             .with_basic_metrics()
             .build())
        
        # Missing extractors
        with self.assertRaises(ValueError):
            (EvaluationConfigBuilder()
             .with_ollama_config()
             .with_basic_metrics()
             .with_data_folder(self.data_dir)
             .build())
    
    def test_metric_configurations(self):
        """Test different metric configurations."""
        
        # All metrics
        config = (EvaluationConfigBuilder()
                 .with_ollama_config()
                 .with_all_metrics()
                 .with_data_folder(self.data_dir)
                 .with_few_shot_extractor(
                     schema_path=self.schemas_dir / "test_schema.json"
                 )
                 .with_output_dir(self.output_dir)
                 .build())
        
        self.assertEqual(len(config.metrics), 7)  # All available metrics
        
        # Basic metrics
        config = (EvaluationConfigBuilder()
                 .with_ollama_config()
                 .with_basic_metrics()
                 .with_data_folder(self.data_dir)
                 .with_few_shot_extractor(
                     schema_path=self.schemas_dir / "test_schema.json"
                 )
                 .with_output_dir(self.output_dir)
                 .build())
        
        self.assertEqual(len(config.metrics), 4)  # Basic metrics
        
        # Semantic metrics
        config = (EvaluationConfigBuilder()
                 .with_ollama_config()
                 .with_semantic_metrics()
                 .with_data_folder(self.data_dir)
                 .with_few_shot_extractor(
                     schema_path=self.schemas_dir / "test_schema.json"
                 )
                 .with_output_dir(self.output_dir)
                 .build())
        
        self.assertEqual(len(config.metrics), 3)  # Semantic metrics
    
    def test_multiple_extractors_config(self):
        """Test configuration with multiple extractors."""
        
        config = (EvaluationConfigBuilder()
                 .with_ollama_config()
                 .with_basic_metrics()
                 .with_data_folder(self.data_dir)
                 .with_few_shot_extractor(
                     schema_path=self.schemas_dir / "test_schema.json",
                     name="FewShot_A"
                 )
                 .with_few_shot_extractor(
                     schema_path=self.schemas_dir / "test_schema.json",
                     name="FewShot_B",
                     confidence_threshold=0.8
                 )
                 .with_vector_rag_extractor(enabled=False)
                 .with_output_dir(self.output_dir)
                 .build())
        
        self.assertEqual(len(config.extractors), 3)
        
        enabled_extractors = [e for e in config.extractors if e.enabled]
        self.assertEqual(len(enabled_extractors), 2)  # Only FewShot extractors enabled
    
    def test_custom_test_case(self):
        """Test adding custom test cases."""
        
        custom_test_case = TestCase(
            name="custom_test",
            document={
                "content": "Custom contract content",
                "metadata": {"type": "custom"}
            },
            ground_truth={
                "customer_name": "Custom Corp"
            }
        )
        
        config = (EvaluationConfigBuilder()
                 .with_ollama_config()
                 .with_basic_metrics()
                 .with_test_cases(custom_test_case)
                 .with_few_shot_extractor(
                     schema_path=self.schemas_dir / "test_schema.json"
                 )
                 .with_output_dir(self.output_dir)
                 .build())
        
        self.assertEqual(len(config.test_cases), 1)
        self.assertEqual(config.test_cases[0].name, "custom_test")


class TestCrossMethodEvaluation(unittest.TestCase):
    """Test cross-method evaluation capabilities."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.output_dir = self.temp_dir / "output"
        self.output_dir.mkdir(parents=True)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_comparison_analysis(self):
        """Test comparison analysis between methods."""
        
        # Create mock results for comparison
        from unified_evaluation_framework import ExtractorResult, ExtractionPerformance
        
        results = [
            ExtractorResult(
                extractor_name="Method_A",
                method=ExtractionMethod.FEW_SHOT,
                test_case_name="test1",
                extracted_data={"field1": "value1"},
                ground_truth={"field1": "value1"},
                performance=ExtractionPerformance(
                    extraction_time=1.0,
                    confidence=0.9,
                    success=True
                ),
                metrics={"accuracy": 1.0, "f1_score": 1.0},
                validation_errors=[]
            ),
            ExtractorResult(
                extractor_name="Method_B",
                method=ExtractionMethod.VECTOR_RAG,
                test_case_name="test1",
                extracted_data={"field1": "value2"},
                ground_truth={"field1": "value1"},
                performance=ExtractionPerformance(
                    extraction_time=2.0,
                    confidence=0.7,
                    success=True
                ),
                metrics={"accuracy": 0.0, "f1_score": 0.0},
                validation_errors=[]
            )
        ]
        
        config = EvaluationConfig(
            metrics=[EvaluationMetric.ACCURACY, EvaluationMetric.F1_SCORE],
            test_cases=[],
            extractors=[],
            llm_config=LLMConfig(provider=LLMProvider.OLLAMA, model="test"),
            output_dir=self.output_dir
        )
        
        framework = UnifiedEvaluationFramework(config)
        framework.results = results
        
        # Test summary calculation
        summary = framework._calculate_summary()
        
        self.assertIn("Method_A", summary)
        self.assertIn("Method_B", summary)
        self.assertEqual(summary["Method_A"]["success_rate"], 1.0)
        self.assertEqual(summary["Method_B"]["success_rate"], 1.0)
        
        # Test comparison analysis
        comparison = framework._generate_comparison_analysis()
        
        self.assertIn("metric_rankings", comparison)
        if comparison["metric_rankings"]:
            # Method_A should rank higher in accuracy
            accuracy_ranking = comparison["metric_rankings"].get("accuracy", [])
            if accuracy_ranking:
                self.assertEqual(accuracy_ranking[0][0], "Method_A")


if __name__ == "__main__":
    unittest.main()
