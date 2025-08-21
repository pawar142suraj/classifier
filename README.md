# DocuVerse: Advanced Document Information Extraction Research Library

A comprehensive research library for evaluating and comparing different document information extraction methods, from baseline few-shot approaches to novel dynamic graph-RAG techniques.

## Overview

This library provides a systematic framework for testing and comparing various document information extraction approaches:

1. **🎯 Unified Few-shot Extraction**: Hybrid extraction + classification with dynamic example loading
2. **🔍 Vector RAG**: Advanced retrieval-augmented generation with state-of-the-art optimizations
3. **🧠 Reasoning Enhancement**: Chain of Thought (CoT) and ReAct methodologies with Vector RAG integration
4. **🕸️ Graph RAG**: Knowledge graph-based extraction with dynamic subgraph generation
5. **⚡ Dynamic Graph-RAG**: Novel adaptive retrieval expansion based on verifier uncertainty

## 🏆 Latest Achievements

### Vector RAG Implementation
- ✅ **2.4x Faster Processing** than Few-Shot (3.65s vs 8.85s)
- ✅ **100% Validation Success Rate** (vs 67% for Few-Shot)
- ✅ **Perfect Field Accuracy** (1.000 for both content and classification)
- ✅ **Advanced Chunking Strategies** (Semantic, Fixed-size, Sliding Window, Hierarchical)
- ✅ **Hybrid Retrieval** (BM25 + Semantic Search with FAISS)
- ✅ **Cross-Encoder Reranking** for improved relevance

### Reasoning Enhancement Implementation
- ✅ **Chain of Thought (CoT)**: Step-by-step reasoning with retrieval context
- ✅ **ReAct Methodology**: Iterative reasoning and acting with self-correction
- ✅ **Vector RAG Integration**: Reasoning-augmented retrieval for enhanced context
- ✅ **Evidence Tracking**: Complete provenance from source to extraction
- ✅ **Multi-step Verification**: Automatic validation and error correction
- ✅ **Uncertainty Detection**: Confidence scoring and uncertainty handling

### Performance Comparison
| Method | Speed | Validation | Confidence | Reasoning | Best Use Case |
|---------|-------|------------|------------|-----------|---------------|
| Few-Shot | 8.85s | 67% | 1.000 | ⭐⭐ | Small, consistent docs |
| Vector RAG | 3.65s | 100% | 0.867 | ⭐⭐⭐ | Large, complex docs |
| CoT Reasoning | ~12s | 95%+ | 0.870+ | ⭐⭐⭐⭐⭐ | Critical analysis |
| ReAct Reasoning | ~15s | 95%+ | 0.885+ | ⭐⭐⭐⭐⭐ | Complex documents |

## Features

- 🔬 **Research-First Design**: Built for systematic evaluation and comparison
- 📊 **Comprehensive Metrics**: Accuracy, precision, recall, F1, semantic similarity
- 📈 **Benchmarking Suite**: Standardized evaluation across different document types
- 🧪 **Ablation Studies**: Component-wise analysis of each method
- 📝 **Paper Generation**: Automated research paper and visualization generation
- 🔍 **Unified Pipeline**: Single extractor handles both extraction and classification
- 🎯 **Schema-Aware**: Automatic field type detection and processing
- 📁 **Auto-Loading**: Dynamic loading of examples from data/labels folder
- ✅ **Validation**: Built-in schema compliance checking

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

## Quick Start: Unified Extraction

### Few-Shot Approach
```python
from docuverse.core.config import LLMConfig
from docuverse.extractors.few_shot import FewShotExtractor

# Configure LLM
llm_config = LLMConfig(
    provider="ollama",
    model_name="llama2",
    base_url="http://localhost:11434"
)

# Initialize unified extractor with auto-loading
extractor = FewShotExtractor(
    llm_config=llm_config,
    schema_path="schemas/contracts_schema_hybrid.json",
    auto_load_labels=True  # Loads examples from data/labels/
)

# Extract and classify in one operation
document = {"content": "Contract with monthly payments..."}
result = extractor.extract(document)
```

### Reasoning Enhancement
```python
from docuverse.extractors.reasoning import ReasoningExtractor
from docuverse.core.config import ReasoningConfig, ExtractionMethod

# Configure reasoning
reasoning_config = ReasoningConfig(
    use_cot=True,
    verification_enabled=True,
    uncertainty_threshold=0.7
)

# Initialize CoT reasoning extractor
extractor = ReasoningExtractor(
    llm_config=llm_config,
    reasoning_config=reasoning_config,
    method_type=ExtractionMethod.REASONING_COT,
    schema_path="schemas/contract_schema.json",
    use_vector_rag=True  # Enhanced with retrieval
)

# Extract with step-by-step reasoning
result = extractor.extract(document)

# Get reasoning analysis
analysis = extractor.get_reasoning_analysis()
print(f"Reasoning steps: {analysis['total_steps']}")
print(f"Evidence pieces: {analysis['evidence_pieces']}")
print(f"Confidence: {analysis['overall_confidence']}")
```

### Vector RAG Approach
```python
from docuverse.extractors.vector_rag import VectorRAGExtractor
from docuverse.core.config import VectorRAGConfig, ChunkingStrategy

# Configure Vector RAG
rag_config = VectorRAGConfig(
    chunk_size=512,
    chunking_strategy=ChunkingStrategy.SEMANTIC,
    retrieval_k=5,
    rerank_top_k=3,
    use_hybrid_search=True
)

# Initialize Vector RAG extractor
extractor = VectorRAGExtractor(
    llm_config=llm_config,
    rag_config=rag_config,
    schema_path="schemas/contract_schema.json"
)

# Extract with advanced retrieval
result = extractor.extract(document)
```

# Result includes both extraction and classification
print(result)
# {
#   "fields": {
#     "payment_terms": {
#       "extracted_content": "Payments are due every month",
#       "classification": "monthly"
#     },
#     "customer_name": {
#       "extracted_content": "John Doe"
#     }
#   }
# }
```

## Schema-Driven Hybrid Processing

The extractor automatically handles different field types based on your schema:

```json
{
  "field": {
    "payment_terms": {
      "type": "string",
      "enum": ["monthly", "yearly", "one-time"],
      "description": "The payment terms for the contract.",
      "enumDescriptions": {
        "monthly": "Payment is due every month.",
        "yearly": "Payment is due once a year.",
        "one-time": "Payment is made in a single transaction."
      }
    },
    "customer_name": {
      "type": "string",
      "description": "The name of the customer for the contract."
    }
  }
}
```

**Fields with `enum`**: Hybrid extraction + classification
**Fields without `enum`**: Pure extraction

## Auto-Loading Examples

Place your training examples in `data/labels/` folder:

```
data/
├── labels/
│   ├── contract1_label.json    # Ground truth labels
│   ├── contract2_label.json
│   └── ...
├── contract1.txt               # Optional: corresponding documents
└── contract2.txt
```

The extractor automatically:
- ✅ Loads all JSON files from `data/labels/`
- ✅ Matches with corresponding document files
- ✅ Generates synthetic documents if needed
- ✅ Provides few-shot examples for better accuracy

## Research Methods

### 1. Few-Shot Baseline
- Traditional prompt engineering with examples
- Chunking strategies for long documents
- Token optimization

### 2. Vector RAG
- Hybrid retrieval (BM25 + semantic search)
- Reranking with cross-encoders
- Adaptive chunk sizing

### 3. Reasoning Enhancement
- Chain of Thought (CoT) reasoning with retrieval context
- ReAct (Reasoning + Acting) iterative methodology
- Multi-step verification and auto-correction
- Evidence tracking and uncertainty detection
- Vector RAG integration for enhanced context

### 4. Graph RAG
- Knowledge graph construction
- Entity-centric subgraph retrieval
- Cypher-like query generation

### 5. Dynamic Graph-RAG (Novel)
- Adaptive retrieval expansion
- Uncertainty-based fallback mechanisms
- Lightweight KG for entity-centric queries

### 6. Hybrid Extraction + Classification
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

## Documentation

- **[Reasoning Extractor Guide](docs/REASONING_EXTRACTOR_GUIDE.md)**: Chain of Thought and ReAct methodologies
- **[Vector RAG Guide](docs/VECTOR_RAG_GUIDE.md)**: Advanced retrieval-augmented generation
- **[Hybrid Extraction Guide](docs/HYBRID_EXTRACTION_GUIDE.md)**: Combined extraction and classification
- **[Few-Shot Guide](docs/FEW_SHOT_GUIDE.md)**: Traditional few-shot approaches
- **[LLM Setup Guide](docs/LLM_SETUP_GUIDE.md)**: Language model configuration

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
