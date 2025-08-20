"""
Schema validation utilities.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class ValidationResult:
    """Result of schema validation."""
    
    def __init__(self, is_valid: bool, errors: List[str] = None):
        self.is_valid = is_valid
        self.errors = errors or []


class SchemaValidator:
    """
    Utility for validating extracted data against JSON schemas.
    """
    
    def __init__(self, schema_path: Optional[str] = None):
        """
        Initialize schema validator.
        
        Args:
            schema_path: Path to JSON schema file
        """
        self.schema = None
        
        if schema_path:
            self.load_schema(schema_path)
        
        logger.info("Initialized SchemaValidator")
    
    def load_schema(self, schema_path: str) -> None:
        """Load JSON schema from file."""
        try:
            with open(schema_path, 'r') as f:
                self.schema = json.load(f)
            logger.info(f"Loaded schema from {schema_path}")
        except Exception as e:
            logger.error(f"Failed to load schema: {e}")
            raise
    
    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """
        Validate data against the loaded schema.
        
        Args:
            data: Data to validate
            
        Returns:
            ValidationResult with validation status and errors
        """
        if not self.schema:
            logger.warning("No schema loaded, skipping validation")
            return ValidationResult(is_valid=True)
        
        try:
            # Try to use jsonschema for validation
            import jsonschema
            
            jsonschema.validate(instance=data, schema=self.schema)
            return ValidationResult(is_valid=True)
            
        except ImportError:
            # Fallback to basic validation if jsonschema not available
            logger.warning("jsonschema not installed, using basic validation")
            return self._basic_validation(data)
            
        except jsonschema.ValidationError as e:
            return ValidationResult(
                is_valid=False,
                errors=[str(e)]
            )
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation error: {str(e)}"]
            )
    
    def _basic_validation(self, data: Dict[str, Any]) -> ValidationResult:
        """Basic validation when jsonschema is not available."""
        errors = []
        
        if not isinstance(data, dict):
            errors.append("Data must be a dictionary")
            return ValidationResult(is_valid=False, errors=errors)
        
        # Check if schema has required fields
        if "required" in self.schema:
            for required_field in self.schema["required"]:
                if required_field not in data:
                    errors.append(f"Missing required field: {required_field}")
        
        # Check field types if properties are defined
        if "properties" in self.schema:
            for field, field_schema in self.schema["properties"].items():
                if field in data:
                    expected_type = field_schema.get("type")
                    actual_value = data[field]
                    
                    if expected_type == "string" and not isinstance(actual_value, str):
                        errors.append(f"Field '{field}' should be string, got {type(actual_value).__name__}")
                    elif expected_type == "number" and not isinstance(actual_value, (int, float)):
                        errors.append(f"Field '{field}' should be number, got {type(actual_value).__name__}")
                    elif expected_type == "boolean" and not isinstance(actual_value, bool):
                        errors.append(f"Field '{field}' should be boolean, got {type(actual_value).__name__}")
                    elif expected_type == "array" and not isinstance(actual_value, list):
                        errors.append(f"Field '{field}' should be array, got {type(actual_value).__name__}")
                    elif expected_type == "object" and not isinstance(actual_value, dict):
                        errors.append(f"Field '{field}' should be object, got {type(actual_value).__name__}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors
        )
