# DocuVerse: Advanced Document Information Extraction Research Library

A comprehensive research library for evaluating and comparing different document information extraction methods, from baseline few-shot approaches to novel dynamic graph-RAG techniques.

## Overview

This library provides a systematic framework for testing and comparing various document information extraction approaches:

1. **Few-shot Baseline**: Traditional prompt-based extraction with examples
2. **Vector RAG**: Hybrid retrieval with reranking for long documents
3. **Graph RAG**: Knowledge graph-based extraction with dynamic subgraph generation
4. **Reasoning Enhancement**: Verifier-augmented CoT/ReAct with schema validation
5. **Dynamic Graph-RAG**: Novel adaptive retrieval expansion based on verifier uncertainty
6. **Hybrid Extraction + Classification**: Extract content AND classify it using enum definitions

## Features

- 🔬 **Research-First Design**: Built for systematic evaluation and comparison
- 📊 **Comprehensive Metrics**: Accuracy, precision, recall, F1, semantic similarity
- 📈 **Benchmarking Suite**: Standardized evaluation across different document types
- 🧪 **Ablation Studies**: Component-wise analysis of each method
- 📝 **Paper Generation**: Automated research paper and visualization generation
- 🔍 **Hybrid Extraction**: Combine content extraction with enum-based classification
- 🎯 **Schema Validation**: Pydantic-based type validation and error handling

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd docuverse

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from docuverse import DocumentExtractor, ExtractionConfig
from docuverse.evaluation import Evaluator

# Configure extraction methods to compare
config = ExtractionConfig(
    methods=['few_shot', 'vector_rag', 'graph_rag', 'dynamic_graph_rag'],
    schema_path='schemas/invoice_schema.json',
    evaluation_metrics=['accuracy', 'f1', 'semantic_similarity']
)

# Initialize extractor
extractor = DocumentExtractor(config)

# Run extraction and evaluation
results = extractor.extract_and_evaluate(
    document_path='data/sample_invoice.pdf',
    ground_truth='data/ground_truth.json'
)

# Generate comparison report
evaluator = Evaluator()
evaluator.generate_report(results, output_path='results/comparison_report.html')
```

## Research Methods

### 1. Few-Shot Baseline
- Traditional prompt engineering with examples
- Chunking strategies for long documents
- Token optimization

### 2. Vector RAG
- Hybrid retrieval (BM25 + semantic search)
- Reranking with cross-encoders
- Adaptive chunk sizing

### 3. Graph RAG
- Knowledge graph construction
- Entity-centric subgraph retrieval
- Cypher-like query generation

### 4. Dynamic Graph-RAG (Novel)
- Adaptive retrieval expansion
- Uncertainty-based fallback mechanisms
- Lightweight KG for entity-centric queries

### 5. Hybrid Extraction + Classification
- Extract exact text content from documents
- Classify extracted content using enum definitions
- Single operation for both extraction and categorization
- Support for structured data parsing (amounts, contacts, etc.)

## Hybrid Extraction Examples

```python
# Payment terms: Extract text AND classify
{
  "payment_terms": {
    "extracted_text": "Net 30 days from invoice date",
    "classification": "standard"  # vs "non_standard", "unknown"
  }
}

# Contract value: Extract amount AND classify tier
{
  "contract_value": {
    "extracted_text": "$75,000 annually",
    "amount": 75000,
    "currency": "USD",
    "classification": "medium_value"  # vs "small", "high_value", "enterprise"
  }
}
```

See [`docs/HYBRID_EXTRACTION_GUIDE.md`](docs/HYBRID_EXTRACTION_GUIDE.md) for detailed documentation.

## Project Structure

```
docuverse/
├── src/docuverse/           # Core library code
│   ├── extractors/          # Different extraction methods
│   ├── rag/                 # RAG implementations
│   ├── graph/               # Graph processing
│   ├── reasoning/           # CoT/ReAct implementations
│   ├── evaluation/          # Metrics and benchmarking
│   └── utils/               # Shared utilities
├── schemas/                 # JSON schemas for different document types
├── data/                    # Sample documents and ground truth
├── benchmarks/              # Benchmark suites
├── experiments/             # Research experiments
├── results/                 # Evaluation results
└── docs/                    # Documentation
```

## Contributing

This is a research library designed for systematic evaluation of document extraction methods. Contributions are welcome, especially:

- New extraction methods
- Additional evaluation metrics
- Benchmark datasets
- Performance optimizations

## License

MIT License - see LICENSE file for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{docuverse2025,
  title={DocuVerse: A Comprehensive Framework for Document Information Extraction Research},
  author={[Your Name]},
  year={2025},
  url={https://github.com/yourusername/docuverse}
}
```
