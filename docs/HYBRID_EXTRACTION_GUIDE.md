# Hybrid Extraction + Classification Guide

## Overview

The DocuVerse hybrid extraction system combines content extraction with classification in a single operation. This allows you to both extract relevant text from documents AND classify that content into predefined categories using enum definitions.

## Key Features

### 1. Dual-Purpose Fields
- **Extract Content**: Pull exact text from documents (e.g., "Net 30 days from invoice date")
- **Classify Content**: Categorize extracted content using enums (e.g., "standard" vs "non_standard")
- **Single Operation**: Both extraction and classification happen simultaneously

### 2. Schema Types

#### Pure Classification
```json
{
  "field_name": {
    "type": "string",
    "enum": ["option1", "option2", "option3"],
    "enumDefinitions": {
      "option1": "Description of option 1",
      "option2": "Description of option 2"
    }
  }
}
```

#### Hybrid Extraction + Classification
```json
{
  "field_name": {
    "type": "object",
    "properties": {
      "extracted_text": {
        "type": "string",
        "description": "Exact text extracted from document"
      },
      "classification": {
        "type": "string",
        "enum": ["standard", "non_standard", "unknown"],
        "enumDefinitions": {
          "standard": "Common industry-standard terms",
          "non_standard": "Custom or unusual terms"
        }
      }
    }
  }
}
```

## Example Use Cases

### 1. Payment Terms Analysis
```python
# Schema Field
"payment_terms": {
  "type": "object",
  "properties": {
    "extracted_text": {"type": "string"},
    "classification": {
      "enum": ["standard", "non_standard", "unknown"],
      "enumDefinitions": {
        "standard": "Net 30, Net 60, Due on Receipt, etc.",
        "non_standard": "Custom payment arrangements"
      }
    }
  }
}

# Example Results
{
  "payment_terms": {
    "extracted_text": "Payment due within 120 days with 5% discount",
    "classification": "non_standard"
  }
}
```

### 2. Contract Value Classification
```python
# Schema Field
"contract_value": {
  "type": "object",
  "properties": {
    "extracted_text": {"type": "string"},
    "amount": {"type": "number"},
    "currency": {"type": "string"},
    "classification": {
      "enum": ["small", "medium_value", "high_value", "enterprise"],
      "enumDefinitions": {
        "small": "Under $10,000",
        "medium_value": "$10,000 - $100,000",
        "high_value": "$100,000 - $1,000,000",
        "enterprise": "Over $1,000,000"
      }
    }
  }
}
```

### 3. Delivery Requirements
```python
# Schema Field
"delivery_terms": {
  "type": "object",
  "properties": {
    "extracted_text": {"type": "string"},
    "classification": {
      "enum": ["standard", "expedited", "rush", "custom"],
      "enumDefinitions": {
        "standard": "5-10 business days",
        "expedited": "2-4 business days",
        "rush": "Within 24-48 hours",
        "custom": "Special delivery arrangements"
      }
    }
  }
}
```

## Implementation

### 1. Schema Design
The hybrid extractor automatically detects field types:
- **Simple Classification**: Has `enum` property directly
- **Hybrid Fields**: Has `properties` with both `extracted_text` and `classification`

### 2. Prompt Generation
For hybrid fields, the system generates prompts that:
1. Request exact text extraction
2. Ask for classification based on enum definitions
3. Provide context for both tasks

### 3. Result Validation
- Validates extracted text is present
- Ensures classification matches enum options
- Provides confidence scoring for both extraction and classification

## Advanced Features

### 1. Multi-Level Classification
```json
{
  "document_analysis": {
    "type": "object",
    "properties": {
      "contract_type": {
        "enum": ["service", "product", "licensing", "consulting"]
      },
      "priority_level": {
        "enum": ["low", "medium", "high", "urgent"]
      },
      "compliance_status": {
        "enum": ["compliant", "needs_review", "non_compliant"]
      }
    }
  }
}
```

### 2. Structured Data Extraction
```json
{
  "contact_information": {
    "type": "object",
    "properties": {
      "classification": {
        "enum": ["complete", "partial", "missing"]
      },
      "extracted_contacts": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "name": {"type": "string"},
            "email": {"type": "string"},
            "phone": {"type": "string"}
          }
        }
      }
    }
  }
}
```

## Best Practices

### 1. Enum Definitions
- Write clear, specific definitions
- Include examples when helpful
- Cover edge cases with "unknown" or "other" options

### 2. Field Design
- Use hybrid fields when you need both content and categorization
- Use pure classification for simple categorization tasks
- Include validation rules for extracted data

### 3. Prompt Engineering
- The system automatically generates appropriate prompts
- Enum definitions are used to guide classification decisions
- Evidence is extracted to support classification choices

## Usage Example

```python
from docuverse.extractors.classification import ClassificationExtractor

# Initialize with hybrid schema
extractor = ClassificationExtractor(
    llm_config=llm_config,
    schema_path="schemas/hybrid_schema.json"
)

# Extract and classify
result = extractor.extract(document)

# Access results
payment_text = result["payment_terms"]["extracted_text"]
payment_class = result["payment_terms"]["classification"]
contract_amount = result["contract_value"]["amount"]
contract_tier = result["contract_value"]["classification"]
```

## Benefits

1. **Efficiency**: Single pass for both extraction and classification
2. **Consistency**: Classifications based on standardized definitions
3. **Flexibility**: Mix pure classification and hybrid fields in same schema
4. **Accuracy**: Evidence-based classification with confidence scoring
5. **Scalability**: Handle multiple document types with different requirements

This hybrid approach enables sophisticated document processing workflows where you need both the exact content and its semantic classification for downstream processing, analytics, and decision-making.
