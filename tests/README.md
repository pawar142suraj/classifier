# DocuVerse Evaluation & Testing Framework

This directory contains a comprehensive testing and evaluation framework for the DocuVerse document extraction system. The framework provides unified evaluation capabilities across all extraction methods (few-shot, vector RAG, graph RAG, etc.) using the existing evaluation metrics from `src/docuverse/evaluation/`.

## 🏗️ Directory Structure

```
tests/
├── unit/                    # Unit tests for individual components
├── integration/             # Integration tests for end-to-end workflows
├── evaluation/              # Unified evaluation framework
├── benchmarks/              # Performance benchmarking tools
├── data/                    # Test data and data generation tools
├── reports/                 # Generated evaluation reports and results
└── run_all_tests.py         # Comprehensive test runner
```

## 🎯 Key Features

- **Unified Evaluation**: Single framework that works across all extraction methods
- **Comprehensive Metrics**: Uses existing evaluation metrics from `src/docuverse/evaluation/`
- **Cross-Method Comparison**: Compare performance between different extractors
- **Performance Benchmarking**: Measure speed, memory usage, and throughput
- **Automated Test Data Generation**: Create synthetic test cases for evaluation
- **Rich Reporting**: Generate detailed HTML and Markdown reports

## 🚀 Quick Start

### 1. Generate Test Data
```bash
python3 tests/data/generate_test_data.py
```

### 2. Run Quick Evaluation
```bash
python3 tests/run_all_tests.py evaluation --evaluation-type quick
```

### 3. Run Complete Test Suite
```bash
python3 tests/run_all_tests.py all --include-benchmarks
```

## 📊 Evaluation Framework

### Core Components

- **`unified_evaluation_framework.py`**: Main evaluation engine
- **`evaluation_config.py`**: Configuration builder for easy setup
- **`run_evaluation.py`**: Sample evaluation runners with different configurations

### Supported Extractors

- ✅ **FewShot**: Unified extraction and classification
- 🔄 **VectorRAG**: (Ready for implementation)
- 🔄 **GraphRAG**: (Ready for implementation)
- 🔄 **Classification**: (Ready for implementation)

### Available Metrics

- **Accuracy**: Overall extraction accuracy
- **Precision/Recall/F1**: Classification performance
- **Semantic Similarity**: Content quality using embeddings
- **ROUGE-L**: Text overlap analysis
- **BERTScore**: Semantic similarity using BERT

## 🧪 Test Types

### Unit Tests (`unit/`)
- Component-level testing
- Schema validation
- Configuration testing
- Individual extractor functionality

### Integration Tests (`integration/`)
- End-to-end extraction workflows
- Cross-component interactions
- Framework initialization
- Error handling

### Evaluation Tests (`evaluation/`)
- Method comparison across test cases
- Metrics calculation and analysis
- Configuration validation
- Performance measurement

### Benchmarks (`benchmarks/`)
- Speed and memory performance
- Throughput measurement
- Resource utilization
- Scalability testing

## 🔧 Configuration Examples

### Basic Evaluation
```python
from evaluation_config import EvaluationConfigBuilder

config = (EvaluationConfigBuilder()
         .with_ollama_config()
         .with_basic_metrics()
         .with_data_folder(Path("data"))
         .with_few_shot_extractor(
             schema_path="schemas/contract_schema.json"
         )
         .build())
```

### Comprehensive Evaluation
```python
config = (EvaluationConfigBuilder()
         .with_ollama_config()
         .with_all_metrics()
         .with_data_folder(Path("data"))
         .with_few_shot_extractor(name="FewShot_A")
         .with_vector_rag_extractor(name="VectorRAG_A", enabled=False)
         .with_graph_rag_extractor(name="GraphRAG_A", enabled=False)
         .build())
```

## 📈 Reports and Results

### Generated Reports
- **`evaluation_results.json`**: Complete evaluation data
- **`evaluation_report.md`**: Human-readable summary
- **`evaluation_report.html`**: Interactive HTML report
- **`benchmark_results.json`**: Performance benchmark data

### Report Contents
- **Executive Summary**: Overall performance statistics
- **Method Comparison**: Side-by-side extractor comparison
- **Metric Rankings**: Best performers by metric
- **Performance Analysis**: Speed, memory, and throughput data
- **Recommendations**: Data-driven suggestions for method selection

## 🔄 Extending the Framework

### Adding New Extractors
1. Implement extractor in `src/docuverse/extractors/`
2. Add to `ExtractorFactory` in `unified_evaluation_framework.py`
3. Add configuration method to `EvaluationConfigBuilder`

### Adding New Metrics
1. Add metric to `src/docuverse/evaluation/metrics.py`
2. Update `EvaluationMetric` enum in `src/docuverse/core/config.py`
3. Include in evaluation configurations

### Custom Test Cases
```python
from unified_evaluation_framework import TestCase

custom_test = TestCase(
    name="custom_contract",
    document={"content": "Your document content"},
    ground_truth={"field": "expected_value"},
    expected_confidence=0.8
)
```

## ⚡ Performance Optimization

### Quick Development Testing
```bash
python3 tests/run_all_tests.py evaluation --evaluation-type quick
```

### Comprehensive Production Testing
```bash
python3 tests/run_all_tests.py all --evaluation-type comprehensive --include-benchmarks
```

### Memory-Efficient Testing
- Use smaller test datasets for development
- Enable timeout settings for long-running tests
- Monitor memory usage during benchmarks

## 🛠️ Troubleshooting

### Common Issues

1. **LLM Connection Errors**
   - Verify Ollama server is running: `ollama serve`
   - Check model availability: `ollama list`
   - Validate connection settings in config

2. **Import Errors**
   - Ensure `src/` is in Python path
   - Check all dependencies are installed
   - Verify relative import paths

3. **Test Data Issues**
   - Generate fresh test data: `python3 tests/data/generate_test_data.py`
   - Verify data folder structure
   - Check schema compatibility

### Debug Mode
```bash
python3 tests/run_all_tests.py evaluation --verbose
```

## 📚 Related Documentation

- **[LLM Setup Guide](../docs/LLM_SETUP_GUIDE.md)**: Configure language models
- **[Few-Shot Guide](../docs/FEW_SHOT_GUIDE.md)**: Setup few-shot extraction
- **[Unified Extraction Guide](../docs/UNIFIED_EXTRACTION_GUIDE.md)**: Combined extraction/classification

## 🎉 Getting Started

1. **Install Dependencies**: Follow main project setup
2. **Generate Test Data**: Run data generator
3. **Start Ollama**: `ollama serve` (if using Ollama)
4. **Run Quick Test**: `python3 tests/run_all_tests.py evaluation --evaluation-type quick`
5. **Review Results**: Check `tests/reports/` for generated reports

The evaluation framework is designed to grow with your extraction methods. As you implement new extractors, simply add them to the configuration and they'll be automatically included in all evaluations and comparisons!

---

## 📜 Legacy Test Documentation

The following tests were moved to the new structure:

### Moved to `unit/`
- `test_unified_extractor.py` - Basic functionality tests

### Moved to `evaluation/`
- `test_few_shot_evaluation.py` - Comprehensive evaluation suite

### Moved to `reports/`
- `few_shot_evaluation_report.md` - Generated evaluation report

### Legacy Test Runner
- `run_tests.py` - Basic test discovery (superseded by `run_all_tests.py`)
