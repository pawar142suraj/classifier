# Few-Shot Contract Extraction with Dynamic Ground Truth

## Overview

The enhanced FewShotExtractor in DocuVerse supports dynamic loading of training examples from `*.labels.json` files, making it easy to build and manage few-shot learning datasets for contract information extraction and classification.

## Key Features

### 1. Dynamic Ground Truth Loading
- Automatically loads training examples from `*.labels.json` files
- Supports multiple document formats (`.txt`, `.md`, `.pdf`, `.docx`)
- Validates training data structure and completeness
- Provides detailed training data analysis

### 2. Hybrid Extraction + Classification Training
- Examples include both content extraction and classification
- Supports contract-specific field types (payment terms, contract values, etc.)
- Schema-aware prompt construction
- Evidence-based classification learning

### 3. Training Data Management
- Automatic validation of labels structure
- Field coverage analysis across training set
- Export capabilities for model training
- Template-based example creation

## File Structure

```
data/contracts/
├── contract_001.txt          # Document content
├── contract_001.labels.json  # Ground truth labels
├── contract_002.txt
├── contract_002.labels.json
└── contract_003.txt
    contract_003.labels.json
```

## Labels File Format

Each `*.labels.json` file contains ground truth extraction and classification data:

```json
{
  "payment_terms": {
    "extracted_text": "Net 30 days from invoice date",
    "classification": "standard",
    "standardized_value": "Net 30"
  },
  "contract_value": {
    "extracted_text": "$75,000 annually",
    "amount": 75000,
    "currency": "USD",
    "classification": "medium_value"
  },
  "delivery_terms": {
    "extracted_text": "Standard delivery within 5-7 business days",
    "classification": "standard_shipping",
    "delivery_timeframe": "5-7 business days"
  },
  "compliance_requirements": {
    "extracted_text": "This agreement complies with all industry standards...",
    "classification": "fully_compliant",
    "missing_elements": []
  },
  "contact_information": {
    "extracted_contacts": [
      {
        "name": "John Smith",
        "email": "john.smith@businesscorp.com",
        "phone": "(555) 987-6543",
        "role": "Procurement Manager"
      }
    ],
    "classification": "complete"
  },
  "document_type": "contract",
  "priority_level": "medium"
}
```

## Usage

### Basic Initialization

```python
from docuverse.extractors.few_shot import FewShotExtractor
from docuverse.core.config import LLMConfig

# Configure LLM
llm_config = LLMConfig(
    provider="openai",
    model_name="gpt-4-turbo-preview",
    temperature=0.1,
    api_key="your-api-key"
)

# Initialize with training data directory
extractor = FewShotExtractor(
    llm_config=llm_config,
    data_path="data/contracts",
    max_examples=3,
    schema_path="schemas/hybrid_schema.json"
)
```

### Manual Examples

```python
# Initialize with manual examples
manual_examples = [
    {
        "input": "Contract with Net 30 payment terms and $50,000 value",
        "output": {
            "payment_terms": {
                "extracted_text": "Net 30",
                "classification": "standard"
            },
            "contract_value": {
                "amount": 50000,
                "classification": "medium_value"
            }
        }
    }
]

extractor = FewShotExtractor(
    llm_config=llm_config,
    examples=manual_examples
)
```

### Extract from New Documents

```python
# Prepare document
document = {
    "content": "Your contract text here...",
    "file_type": "text",
    "metadata": {"name": "new_contract.txt"}
}

# Extract with few-shot learning
result = extractor.extract(document)

# Access results
payment_terms = result["payment_terms"]
print(f"Payment Text: {payment_terms['extracted_text']}")
print(f"Classification: {payment_terms['classification']}")
print(f"Confidence: {extractor.last_confidence}")
```

## Training Data Management

### TrainingDataManager

```python
from docuverse.utils.training_data_manager import TrainingDataManager

# Initialize manager
manager = TrainingDataManager("data/contracts")

# Validate existing data
report = manager.validate_training_data()
print(f"Valid examples: {report['valid_pairs']}")
print(f"Field coverage: {report['field_coverage']}")

# Get summary
summary = manager.get_training_summary()
print(f"Total examples: {summary['total_examples']}")
print(f"Document types: {summary['document_types']}")
```

### Creating New Examples

```python
# Create new training example
doc_path, labels_path = manager.create_example_from_template(
    document_content="Your contract content...",
    filename="contract_new",
    payment_terms="Net 15 days",
    contract_value=125000
)

# Add existing files as examples
extractor.add_example_from_files(
    document_path="path/to/contract.txt",
    labels_path="path/to/contract.labels.json"
)
```

## Field Types and Classifications

### Payment Terms
- **Standard**: Net 30, Net 15, Net 60, Due on Receipt
- **Non-Standard**: Custom terms, extended periods >60 days
- **Unknown**: Unclear or unspecified terms

### Contract Values
- **Low Value**: Under $10,000
- **Medium Value**: $10,000 - $100,000
- **High Value**: $100,000 - $1,000,000
- **Enterprise**: Over $1,000,000

### Delivery Terms
- **Standard Shipping**: 5-10 business days
- **Expedited**: 2-4 business days
- **Rush**: Within 24-48 hours
- **Custom Delivery**: Special arrangements

### Compliance Requirements
- **Fully Compliant**: All requirements met
- **Partially Compliant**: Most requirements met
- **Non-Compliant**: Missing critical requirements
- **Requires Review**: Complex compliance situation

### Contact Information
- **Complete**: Full details for key parties
- **Partial**: Some contact information missing
- **Minimal**: Basic information only
- **Missing**: No usable contact information

## Advanced Features

### Example Analysis

```python
# Get example summary
summary = extractor.get_example_summary()
print(f"Examples: {summary['count']}")
print(f"Document types: {summary['document_types']}")
print(f"Fields covered: {summary['fields_covered']}")
```

### Export Training Data

```python
# Export for external training
exported_file = manager.export_training_data("training_export.json")
print(f"Exported to: {exported_file}")
```

### Dynamic Example Updates

```python
# Update examples during runtime
new_examples = load_additional_examples()
extractor.update_examples(new_examples)
```

## Best Practices

### 1. Training Data Quality
- Use diverse examples (standard, non-standard, edge cases)
- Ensure consistent field naming across examples
- Include complete hybrid field structures
- Validate labels before adding to training set

### 2. Example Selection
- Include examples that cover different document types
- Balance standard and non-standard cases
- Ensure good coverage of all target fields
- Limit to most relevant examples (3-5 typically optimal)

### 3. Schema Alignment
- Align training examples with target schema
- Use consistent classification categories
- Include evidence fields for classification decisions
- Maintain structured data formats

### 4. Validation and Testing
- Regularly validate training data structure
- Test extraction accuracy on held-out examples
- Monitor field coverage across training set
- Update examples based on model performance

## Integration with Research Framework

### Evaluation Pipeline

```python
from docuverse.evaluation import Evaluator

# Compare few-shot vs other methods
evaluator = Evaluator()
results = evaluator.compare_methods(
    methods=["few_shot", "vector_rag", "classification"],
    test_documents=test_set,
    ground_truth=ground_truth_labels
)
```

### Ablation Studies

```python
# Test different numbers of examples
for n_examples in [1, 2, 3, 5]:
    extractor = FewShotExtractor(
        llm_config=llm_config,
        data_path="data/contracts",
        max_examples=n_examples
    )
    accuracy = evaluate_on_test_set(extractor, test_set)
    print(f"{n_examples} examples: {accuracy:.2%} accuracy")
```

## Performance Considerations

### Example Selection Strategy
- Use most representative examples for target domain
- Include edge cases that are common in real data
- Balance between diversity and relevance
- Consider computational cost of longer prompts

### Prompt Optimization
- Schema-aware prompt construction
- Clear instructions for hybrid fields
- Consistent example formatting
- Evidence-based classification guidance

### Confidence Calibration
- Confidence based on example similarity
- Field completeness scoring
- Structure validation checks
- Pattern matching with training examples

This enhanced few-shot system provides a robust foundation for contract-focused document extraction research, combining the power of in-context learning with systematic ground truth management and hybrid extraction capabilities.
