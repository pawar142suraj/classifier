# Unified Extraction and Classification Guide

This guide explains how to use the enhanced `FewShotExtractor` that combines extraction and classification in a single, unified pipeline.

## Overview

The unified approach handles two types of operations in a single extractor:

1. **Hybrid Fields**: Extract exact text AND classify it using predefined enums
2. **Pure Extraction**: Extract exact text content without classification
3. **Auto-Loading**: Dynamically load training examples from `data/labels/` folder

## Key Features

- ✅ **Single Pipeline**: One extractor handles both extraction and classification
- ✅ **Schema-Aware**: Automatically detects field types from JSON schema
- ✅ **Auto-Loading**: Loads examples from `data/labels/` folder automatically
- ✅ **Validation**: Built-in schema compliance checking
- ✅ **Confidence Scoring**: Provides extraction confidence metrics
- ✅ **Flexible**: Works with or without predefined examples

## Schema Structure

### Hybrid Field (Extract + Classify)
```json
{
  "payment_terms": {
    "type": "string",
    "enum": ["monthly", "yearly", "one-time"],
    "description": "The payment terms for the contract.",
    "enumDescriptions": {
      "monthly": "Payment is due every month.",
      "yearly": "Payment is due once a year.", 
      "one-time": "Payment is made in a single transaction."
    }
  }
}
```

**Output**:
```json
{
  "payment_terms": {
    "extracted_content": "Payments are due every month on the 15th",
    "classification": "monthly"
  }
}
```

### Pure Extraction Field
```json
{
  "customer_name": {
    "type": "string",
    "description": "The name of the customer for the contract."
  }
}
```

**Output**:
```json
{
  "customer_name": {
    "extracted_content": "John Doe"
  }
}
```

## Usage Examples

### Basic Usage

```python
from docuverse.core.config import LLMConfig
from docuverse.extractors.few_shot import FewShotExtractor

# Configure LLM
llm_config = LLMConfig(
    provider="ollama",
    model_name="llama2",
    base_url="http://localhost:11434",
    temperature=0.1
)

# Load schema
with open("schemas/contracts_schema_hybrid.json", 'r') as f:
    schema = json.load(f)

# Initialize extractor
extractor = FewShotExtractor(
    llm_config=llm_config,
    schema=schema,
    auto_load_labels=True  # Auto-load from data/labels/
)

# Extract information
document = {
    "content": "Service contract with monthly payments to John Doe...",
    "metadata": {"filename": "contract.txt"}
}

result = extractor.extract(document)
confidence = extractor.last_confidence
```

### Advanced Configuration

```python
# Custom configuration
extractor = FewShotExtractor(
    llm_config=llm_config,
    schema=schema,
    data_labels_path="path/to/custom/labels",  # Custom labels path
    max_examples=10,                           # More examples
    auto_load_labels=True
)

# Add manual examples
extractor.add_example_from_labels(
    labels_path="path/to/new_example.json",
    document_content="Custom document content..."
)

# Reload examples from folder
extractor.reload_examples_from_labels()
```

## Example Data Structure

### Labels File (`data/labels/contract1_label.json`)
```json
{
  "fields": {
    "payment_terms": {
      "extracted_content": "Payments are due every month.",
      "classification": "monthly"
    },
    "warranty": {
      "extracted_content": "Standard warranty is provided for 1 year.",
      "classification": "standard"
    },
    "customer_name": {
      "extracted_content": "John Doe"
    }
  }
}
```

### Document File (`data/contract1.txt`)
```text
SERVICE AGREEMENT CONTRACT

This Service Agreement is entered into between TechCorp Inc. and John Doe.

PAYMENT TERMS:
Payments are due every month on the 15th of each month.

WARRANTY:
Standard warranty is provided for 1 year from the date of service completion.

CUSTOMER INFORMATION:
Customer Name: John Doe
...
```

## Analysis and Validation

### Get Example Summary
```python
summary = extractor.get_example_summary()
print(f"Total examples: {summary['total_examples']}")
print(f"Field coverage: {summary['field_coverage']}")
print(f"Schema info: {summary['schema_info']}")
```

### Validate Results
```python
validation = extractor.validate_schema_compliance(result)
if validation["is_valid"]:
    print("✅ All schema requirements met")
else:
    print("❌ Validation errors:")
    for error in validation["missing_fields"]:
        print(f"  Missing: {error}")
    for error in validation["invalid_enums"]:
        print(f"  Invalid enum: {error}")
```

### Field Analysis
```python
analysis = extractor.get_field_analysis()

print("Hybrid fields:")
for field in analysis["hybrid_fields"]:
    print(f"  {field['name']}: {field['enum_options']}")

print("Pure extraction fields:")
for field in analysis["extraction_fields"]:
    print(f"  {field['name']}: {field['description']}")
```

## Best Practices

### 1. Schema Design
- Use clear, descriptive enum values
- Provide detailed `enumDescriptions`
- Keep field descriptions concise but informative

### 2. Training Data
- Provide diverse examples in `data/labels/`
- Include edge cases and variations
- Use consistent labeling format

### 3. Example Quality
- Ensure extracted_content matches document text exactly
- Verify classifications align with enum definitions
- Include examples for all schema fields

### 4. Validation
- Always validate results against schema
- Check confidence scores for quality assessment
- Review field coverage in training examples

## Error Handling

### Common Issues

1. **Missing Examples**: Extractor works with zero examples but lower accuracy
2. **Schema Mismatch**: Use validation to catch structure errors
3. **Invalid Enums**: Check enum values match schema definitions
4. **Empty Fields**: Extractor returns empty strings for missing content

### Debugging

```python
# Check loaded examples
print(f"Loaded {len(extractor.examples)} examples")
for i, ex in enumerate(extractor.examples):
    metadata = ex.get("metadata", {})
    print(f"  {i+1}. {metadata.get('base_name', 'unknown')}")

# Analyze field coverage
analysis = extractor.get_field_analysis()
for field, coverage in analysis["example_coverage"].items():
    if coverage["coverage_percentage"] < 50:
        print(f"⚠️ Low coverage for {field}: {coverage['coverage_percentage']:.1f}%")
```

## Performance Tips

1. **Limit Examples**: Use `max_examples` to control prompt length
2. **Schema Optimization**: Keep enum lists concise
3. **Document Chunking**: For long documents, consider preprocessing
4. **Caching**: Results can be cached for repeated extractions

## Integration with Other Methods

The unified extractor can be combined with other approaches:

```python
# Use with vector RAG for long documents
from docuverse.extractors.vector_rag import VectorRAGExtractor

# Fallback chain: unified -> vector RAG -> graph RAG
extractors = [
    FewShotExtractor(llm_config, schema=schema),
    VectorRAGExtractor(llm_config, schema=schema),
    # ... other extractors
]

# Try extractors in sequence until confidence threshold met
for extractor in extractors:
    result = extractor.extract(document)
    if extractor.last_confidence > 0.8:
        break
```

This unified approach provides a solid foundation for document extraction while maintaining the flexibility to integrate with more advanced methods as needed.
