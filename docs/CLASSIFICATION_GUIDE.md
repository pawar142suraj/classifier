# Classification with Enum Definitions - DocuVerse

## Overview

The DocuVerse framework now supports sophisticated **classification tasks** where you can define enums with detailed descriptions in your JSON schema, and the system will:

1. **Extract relevant content** from documents
2. **Classify fields** based on enum definitions 
3. **Provide evidence** for classification decisions
4. **Validate and repair** classifications automatically
5. **Calculate confidence scores** for results

## How It Works

### 1. Define Classification Schema

Create a JSON schema with enums and `enumDescriptions`:

```json
{
  "properties": {
    "document_type": {
      "type": "string",
      "enum": ["invoice", "contract", "receipt", "report", "email"],
      "enumDescriptions": {
        "invoice": "A commercial document requesting payment for goods or services, containing vendor details, line items, and amounts",
        "contract": "A legal agreement between parties outlining terms, conditions, and obligations",
        "receipt": "A proof of payment document showing completed transaction details",
        "report": "An analytical or informational document presenting data, findings, or status updates",
        "email": "Electronic correspondence between parties containing messages and attachments"
      },
      "description": "Classification of the document type based on content and structure"
    },
    "priority_level": {
      "type": "string", 
      "enum": ["low", "medium", "high", "urgent"],
      "enumDescriptions": {
        "low": "Routine documents with no time sensitivity, can be processed within a week",
        "medium": "Standard business documents requiring attention within 2-3 business days",
        "high": "Important documents needing prompt attention within 24 hours",
        "urgent": "Critical documents requiring immediate action within hours, containing keywords like 'urgent', 'immediate', 'deadline', 'overdue'"
      }
    }
  }
}
```

### 2. Use ClassificationExtractor

```python
from docuverse.extractors.classification import ClassificationExtractor
from docuverse.core.config import LLMConfig

# Configure LLM
llm_config = LLMConfig(
    provider="openai",
    model_name="gpt-4-turbo-preview",
    api_key="your-api-key"
)

# Initialize classification extractor
extractor = ClassificationExtractor(
    llm_config=llm_config,
    schema_path="schemas/classification_schema.json"
)

# Extract and classify
document = {"content": "URGENT: Payment overdue notice..."}
result = extractor.extract(document)
```

### 3. Get Structured Results

The extractor returns structured results with classifications and evidence:

```json
{
  "document_type": "invoice",
  "priority_level": "urgent", 
  "sentiment": "negative",
  "extracted_content": {
    "key_entities": ["payment", "overdue", "invoice"],
    "classification_evidence": {
      "document_type_indicators": ["invoice", "payment", "billing"],
      "priority_indicators": ["URGENT", "immediate action", "overdue"],
      "sentiment_indicators": ["overdue", "collection actions", "problem"]
    }
  }
}
```

## Key Features

### 🎯 **Intelligent Classification**
- Uses enum descriptions to guide classification decisions
- Extracts relevant text snippets as evidence
- Handles multiple classification fields simultaneously

### 🔧 **Automatic Validation & Repair**
- Validates results against JSON schema
- Auto-repairs common classification errors
- Maps invalid values to valid enum options

### 📊 **Confidence Scoring**
- Calculates confidence based on evidence quality
- Considers classification completeness
- Helps identify uncertain classifications

### 📈 **Batch Processing & Analytics**
- Process multiple documents efficiently
- Generate classification statistics
- Analyze field distributions and confidence metrics

## Example Use Cases

### 1. Document Routing System
```python
# Classify incoming documents for routing
documents = load_documents("incoming/")
results = extractor.classify_batch(documents)

for doc, result in zip(documents, results):
    route_document(doc, result["document_type"], result["priority_level"])
```

### 2. Content Analysis Pipeline
```python
# Analyze document sentiment and compliance
config = ExtractionConfig(
    methods=[ExtractionMethod.CLASSIFICATION],
    schema_path="schemas/content_analysis_schema.json"
)

# Process documents
results = analyze_content_batch(documents)
compliance_report = generate_compliance_report(results)
```

### 3. Multi-Field Classification
```python
# Classify multiple aspects simultaneously
schema_fields = {
    "document_type": "Primary document classification",
    "priority_level": "Urgency assessment", 
    "sentiment": "Emotional tone analysis",
    "compliance_status": "Regulatory compliance check",
    "language": "Document language detection"
}
```

## Advanced Features

### Custom Enum Descriptions
Make your enum descriptions as detailed as needed:

```json
{
  "priority_level": {
    "enum": ["low", "medium", "high", "urgent"],
    "enumDescriptions": {
      "urgent": "Documents requiring immediate action within 2-4 hours. Indicators: 'URGENT', 'IMMEDIATE', 'ASAP', overdue notices, legal deadlines, system outages, customer escalations, or explicit time constraints under 24 hours."
    }
  }
}
```

### Evidence-Based Decisions
The system provides evidence for each classification:

```python
# Access classification evidence
evidence = result["extracted_content"]["classification_evidence"]
priority_evidence = evidence.get("priority_indicators", [])

print(f"Classified as urgent based on: {priority_evidence}")
# Output: ["URGENT", "immediate action required", "within 24 hours"]
```

### Confidence Thresholds
Set confidence thresholds for quality control:

```python
result = extractor.extract(document)
if extractor.last_confidence < 0.7:
    # Route to human review
    send_for_manual_review(document, result)
else:
    # Process automatically
    process_classification(result)
```

## Integration with Research Framework

The classification extractor integrates seamlessly with the research framework:

```python
from experiments.research_framework import ResearchExperiment

# Compare classification methods
experiment = ResearchExperiment("classification_comparison")

extractors = {
    "few_shot_classification": few_shot_extractor,
    "enum_based_classification": classification_extractor,
    "hybrid_classification": hybrid_extractor
}

results = experiment.run_method_comparison(
    extractors=extractors,
    documents=test_documents,
    ground_truths=ground_truth_labels
)
```

## Best Practices

### 1. **Write Clear Enum Descriptions**
- Be specific about what content indicates each category
- Include examples of keywords or phrases
- Specify edge cases and disambiguation rules

### 2. **Design Comprehensive Schemas**
- Include evidence extraction fields
- Plan for validation and error handling
- Consider hierarchical classifications

### 3. **Use Evidence for Quality Control**
- Review classification evidence for accuracy
- Identify patterns in misclassifications
- Refine enum descriptions based on results

### 4. **Monitor Confidence Scores**
- Set appropriate confidence thresholds
- Route low-confidence results for review
- Track confidence trends over time

## Schema Templates

Find ready-to-use schema templates in the `schemas/` directory:

- `classification_schema.json` - General document classification
- `invoice_schema.json` - Invoice-specific extraction
- `contract_schema.json` - Contract analysis
- `email_classification_schema.json` - Email categorization
- `compliance_schema.json` - Regulatory compliance checking

## Performance Optimization

### Batch Processing
```python
# Process large document sets efficiently
batch_size = 10
documents = load_large_dataset()

for i in range(0, len(documents), batch_size):
    batch = documents[i:i+batch_size]
    results = extractor.classify_batch(batch)
    save_results(results)
```

### Caching and Reuse
```python
# Cache classification models and embeddings
extractor = ClassificationExtractor(
    llm_config=llm_config,
    schema_path="schema.json",
    cache_embeddings=True,
    cache_dir="classification_cache/"
)
```

This classification system turns your document processing into a sophisticated, evidence-based classification pipeline that can handle complex business logic while maintaining transparency and accuracy!
