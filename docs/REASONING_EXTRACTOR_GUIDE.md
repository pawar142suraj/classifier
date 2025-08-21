# Reasoning Extractor Guide

## Overview

The **Reasoning Extractor** is an advanced document information extraction method that combines sophisticated reasoning methodologies with Vector RAG capabilities. It implements Chain of Thought (CoT) and ReAct (Reasoning + Acting) approaches to provide enhanced analytical capabilities for contract information extraction.

## Key Features

### 🧠 Advanced Reasoning Methods
- **Chain of Thought (CoT)**: Step-by-step reasoning with retrieval-augmented context
- **ReAct Methodology**: Iterative reasoning and acting with self-correction
- **Multi-step Verification**: Automatic validation and error correction
- **Uncertainty Detection**: Confidence scoring and uncertainty handling

### 🔍 Vector RAG Integration
- **Hybrid Retrieval**: Combines BM25 and semantic search
- **Semantic Chunking**: Intelligent document segmentation
- **Context-Aware Reasoning**: Retrieval-augmented reasoning steps
- **Evidence Tracking**: Links extracted information to source text

### 📊 Enhanced Analytics
- **Reasoning Trace**: Complete step-by-step reasoning history
- **Evidence Scoring**: Confidence assessment for each extraction
- **Uncertainty Flags**: Identification of low-confidence extractions
- **Performance Metrics**: Detailed analysis of reasoning process

## Architecture

```
Document Input
     ↓
[Vector RAG Components]
     ↓
[Reasoning Engine]
     ↓
[Verification & Correction]
     ↓
Structured Output + Reasoning Trace
```

### Core Components

1. **ReasoningExtractor**: Main extraction class
2. **Vector RAG Integration**: Optional retrieval augmentation
3. **Reasoning Steps**: Structured thinking process
4. **Evidence Tracker**: Links extractions to source evidence
5. **Verification System**: Multi-step validation and correction

## Usage Examples

### Basic CoT Reasoning

```python
from docuverse.extractors.reasoning import ReasoningExtractor
from docuverse.core.config import LLMConfig, ReasoningConfig, ExtractionMethod

# Configure LLM
llm_config = LLMConfig(
    provider="ollama",
    model_name="llama3.2:latest",
    temperature=0.1
)

# Configure reasoning
reasoning_config = ReasoningConfig(
    use_cot=True,
    verification_enabled=True,
    uncertainty_threshold=0.7,
    max_reasoning_steps=5
)

# Initialize CoT extractor
extractor = ReasoningExtractor(
    llm_config=llm_config,
    reasoning_config=reasoning_config,
    method_type=ExtractionMethod.REASONING_COT,
    schema_path="schemas/contract_schema.json",
    use_vector_rag=True
)

# Extract with reasoning
document = {"content": contract_text, "metadata": {"source": "contract"}}
result = extractor.extract(document)

print(f"Confidence: {result['metadata']['overall_confidence']}")
print(f"Reasoning steps: {result['metadata']['reasoning_steps']}")
```

### ReAct Methodology

```python
# Initialize ReAct extractor
react_extractor = ReasoningExtractor(
    llm_config=llm_config,
    reasoning_config=reasoning_config,
    method_type=ExtractionMethod.REASONING_REACT,
    schema_path="schemas/contract_schema.json",
    use_vector_rag=True
)

# Extract with iterative reasoning
result = react_extractor.extract(document)

# Get detailed reasoning analysis
analysis = react_extractor.get_reasoning_analysis()
print(f"Evidence pieces: {analysis['evidence_pieces']}")
print(f"Step breakdown: {analysis['step_breakdown']}")
```

### Advanced Configuration

```python
from docuverse.core.config import VectorRAGConfig, ChunkingStrategy

# Custom reasoning configuration
reasoning_config = ReasoningConfig(
    use_cot=True,
    use_react=True,
    max_reasoning_steps=8,
    verification_enabled=True,
    auto_repair_enabled=True,
    uncertainty_threshold=0.6
)

# Custom Vector RAG configuration
rag_config = VectorRAGConfig(
    chunk_size=384,  # Smaller chunks for better reasoning
    chunk_overlap=50,
    chunking_strategy=ChunkingStrategy.SEMANTIC,
    retrieval_k=8,  # More context for reasoning
    rerank_top_k=5,
    use_hybrid_search=True,
    bm25_weight=0.4
)

# Initialize with custom configurations
extractor = ReasoningExtractor(
    llm_config=llm_config,
    reasoning_config=reasoning_config,
    method_type=ExtractionMethod.REASONING_COT,
    schema=schema,
    use_vector_rag=True
)

# The extractor will use the reasoning config to optimize Vector RAG
```

## Reasoning Process

### Chain of Thought (CoT)

1. **Document Analysis**: Understand document type and structure
2. **Field Identification**: Map target fields to document sections
3. **Information Extraction**: Extract with evidence and confidence
4. **Structure Validation**: Organize and validate results

```
Step 1 - Document Analysis:
"This appears to be a service contract between two parties. 
The document contains standard contract sections including 
parties, terms, payment details, and legal provisions..."

Step 2 - Field Identification:
"Contract number likely appears in the header section.
Party information is in the opening paragraphs.
Payment terms are discussed in section 3..."

Step 3 - Information Extraction:
"CONTRACT_NUMBER:
- Extracted Value: SC-2024-001
- Evidence: 'Service Contract No. SC-2024-001'
- Confidence: 0.95
- Reasoning: Clear contract identifier in header"

Step 4 - Structure Validation:
"Validating extracted information against schema requirements.
All required fields present. Data types validated.
Final JSON structure confirmed."
```

### ReAct Methodology

For each target field:

```
Thought 1: I need to find the contract number. It's usually in the header or first paragraph.

Action 1: Search for patterns like "Contract No.", "Agreement No.", or similar identifiers.

Observation 1: Found "Service Contract No. SC-2024-001" in the document header.
- Extracted Value: SC-2024-001
- Confidence: 0.95

Thought 2: High confidence extraction. The format matches typical contract numbering.

Action 2: Verify this is the primary contract identifier and not a reference number.

Observation 2: Confirmed as primary identifier. No other contract numbers found.
```

## Output Structure

```json
{
  "fields": {
    "contract_number": "SC-2024-001",
    "contract_title": "Software Development Services Agreement",
    "parties": [
      {
        "name": "TechCorp Inc.",
        "role": "contractor",
        "address": "123 Tech Street, Silicon Valley, CA"
      }
    ],
    "effective_date": "2024-01-15",
    "contract_value": {
      "amount": 150000,
      "currency": "USD"
    }
  },
  "metadata": {
    "extraction_method": "reasoning_cot",
    "reasoning_steps": 12,
    "evidence_pieces": 8,
    "overall_confidence": 0.87,
    "extraction_time": 15.3,
    "vector_rag_enabled": true,
    "verification_enabled": true,
    "reasoning_trace": [
      {
        "step": 1,
        "type": "thought",
        "content": "Analyzing document structure...",
        "confidence": 0.8,
        "uncertainty_flags": []
      }
    ],
    "evidence_summary": [
      {
        "field": "contract_number",
        "value": "SC-2024-001",
        "confidence": 0.95,
        "reasoning": "Clear identifier in header section"
      }
    ]
  }
}
```

## Performance Characteristics

### When to Use Reasoning Extractors

**✅ Ideal for:**
- Complex documents requiring detailed analysis
- High-stakes applications needing evidence tracking
- Documents with ambiguous or unclear information
- When explainable AI is required
- Multi-step validation scenarios

**⚠️ Consider alternatives for:**
- Simple, well-structured documents
- High-volume, speed-critical processing
- When LLM costs are a primary concern
- Basic information extraction tasks

### Performance Comparison

| Method | Speed | Accuracy | Explainability | Evidence |
|--------|-------|----------|----------------|-----------|
| Vector RAG | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| CoT Reasoning | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| ReAct Reasoning | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## Configuration Options

### ReasoningConfig

```python
reasoning_config = ReasoningConfig(
    use_cot=True,              # Enable Chain of Thought
    use_react=False,           # Enable ReAct methodology
    max_reasoning_steps=5,     # Maximum reasoning iterations
    verification_enabled=True, # Enable result verification
    auto_repair_enabled=True,  # Enable automatic error correction
    uncertainty_threshold=0.7  # Confidence threshold for acceptance
)
```

### Vector RAG Integration

```python
# Optimized for reasoning
rag_config = VectorRAGConfig(
    chunk_size=384,           # Smaller chunks for detailed analysis
    retrieval_k=8,            # More context for reasoning
    rerank_top_k=5,           # Focus on most relevant chunks
    use_hybrid_search=True,   # Combine BM25 + semantic search
    bm25_weight=0.4          # Higher keyword weight for reasoning
)
```

## Advanced Features

### Evidence Tracking

Every extraction includes:
- **Source Evidence**: Exact text supporting the extraction
- **Confidence Score**: Quantified extraction certainty
- **Reasoning Chain**: Step-by-step logical process
- **Uncertainty Flags**: Identified potential issues

### Verification System

1. **Schema Validation**: Check against target schema
2. **Cross-Reference**: Verify consistency across fields
3. **Auto-Repair**: Attempt correction of identified errors
4. **Confidence Assessment**: Update confidence based on verification

### Uncertainty Handling

- **Threshold-based Filtering**: Skip low-confidence extractions
- **Alternative Hypotheses**: Consider multiple interpretations
- **Confidence Propagation**: Track uncertainty through reasoning steps
- **Human-in-the-Loop**: Flag uncertain cases for review

## Integration Examples

### With Evaluation Framework

```python
from docuverse.evaluation.evaluator import UnifiedEvaluator
from docuverse.core.config import ExtractionConfig

config = ExtractionConfig(
    methods=[ExtractionMethod.REASONING_COT, ExtractionMethod.REASONING_REACT],
    evaluation_metrics=[EvaluationMetric.ACCURACY, EvaluationMetric.F1],
    reasoning_config=reasoning_config
)

evaluator = UnifiedEvaluator(config)
results = evaluator.evaluate_methods(test_documents)
```

### Batch Processing

```python
import asyncio
from pathlib import Path

async def process_contracts_with_reasoning():
    extractor = ReasoningExtractor(
        llm_config=llm_config,
        reasoning_config=reasoning_config,
        method_type=ExtractionMethod.REASONING_COT,
        use_vector_rag=True
    )
    
    contract_files = Path("contracts/").glob("*.txt")
    results = []
    
    for contract_file in contract_files:
        with open(contract_file) as f:
            document = {"content": f.read(), "metadata": {"source": str(contract_file)}}
        
        result = extractor.extract(document)
        results.append(result)
        
        print(f"Processed {contract_file.name}: "
              f"confidence={result['metadata']['overall_confidence']:.3f}")
    
    return results

results = asyncio.run(process_contracts_with_reasoning())
```

## Troubleshooting

### Common Issues

1. **Slow Performance**: Reduce `max_reasoning_steps` or disable Vector RAG
2. **Low Confidence**: Adjust `uncertainty_threshold` or improve schema
3. **Memory Usage**: Reduce `retrieval_k` or use smaller embedding models
4. **LLM Errors**: Check model compatibility and API configuration

### Optimization Tips

1. **Schema Design**: Provide clear field descriptions and examples
2. **Document Preprocessing**: Clean and structure input documents
3. **Model Selection**: Use reasoning-capable models (e.g., GPT-4, Claude)
4. **Batch Size**: Process documents in appropriate batch sizes
5. **Caching**: Enable embedding cache for repeated processing

## Best Practices

### Schema Design for Reasoning

```json
{
  "properties": {
    "contract_value": {
      "type": "object",
      "description": "Total monetary value of the contract including amount and currency",
      "properties": {
        "amount": {
          "type": "number",
          "description": "Numerical value of the contract (e.g., 150000 for $150,000)"
        },
        "currency": {
          "type": "string",
          "description": "Currency code (e.g., USD, EUR, GBP)"
        }
      }
    }
  }
}
```

### Prompt Engineering

- Use clear, specific field descriptions
- Provide examples in schema when possible
- Include validation rules and constraints
- Specify expected formats and units

### Error Handling

```python
try:
    result = extractor.extract(document)
    
    # Check confidence threshold
    if result['metadata']['overall_confidence'] < 0.6:
        logger.warning("Low confidence extraction - review required")
    
    # Check for uncertainty flags
    for step in result['metadata'].get('reasoning_trace', []):
        if step['uncertainty_flags']:
            logger.info(f"Uncertainty in step {step['step']}: {step['uncertainty_flags']}")
            
except Exception as e:
    logger.error(f"Reasoning extraction failed: {e}")
    # Fallback to simpler method
    result = fallback_extractor.extract(document)
```

## Conclusion

The Reasoning Extractor provides state-of-the-art capabilities for complex document information extraction. By combining advanced reasoning methodologies with Vector RAG capabilities, it offers unparalleled accuracy and explainability for contract analysis and other structured document processing tasks.

The trade-off between processing speed and analytical depth makes it ideal for applications where accuracy and evidence tracking are more important than raw throughput.
