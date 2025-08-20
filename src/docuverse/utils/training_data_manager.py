"""
Utility for managing few-shot training data with *.labels.json files.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class TrainingDataManager:
    """
    Manages training data for few-shot contract extraction.
    Handles creation, validation, and organization of *.labels.json files.
    """
    
    def __init__(self, data_directory: Union[str, Path]):
        """
        Initialize training data manager.
        
        Args:
            data_directory: Path to directory containing training data
        """
        self.data_directory = Path(data_directory)
        self.data_directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized TrainingDataManager for {self.data_directory}")
    
    def create_labels_file(
        self, 
        document_path: Union[str, Path],
        labels: Dict[str, Any],
        overwrite: bool = False
    ) -> Path:
        """
        Create a *.labels.json file for a document.
        
        Args:
            document_path: Path to the document file
            labels: Ground truth labels dictionary
            overwrite: Whether to overwrite existing labels file
            
        Returns:
            Path to created labels file
        """
        document_path = Path(document_path)
        
        # Generate labels file path
        labels_file = document_path.with_suffix('.labels.json')
        
        if labels_file.exists() and not overwrite:
            raise FileExistsError(f"Labels file already exists: {labels_file}")
        
        # Validate labels structure
        self._validate_labels(labels)
        
        # Write labels file
        with open(labels_file, 'w', encoding='utf-8') as f:
            json.dump(labels, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Created labels file: {labels_file}")
        return labels_file
    
    def validate_training_data(self) -> Dict[str, Any]:
        """
        Validate all training data in the directory.
        
        Returns:
            Validation report
        """
        report = {
            "total_documents": 0,
            "valid_pairs": 0,
            "invalid_pairs": 0,
            "missing_labels": [],
            "missing_documents": [],
            "validation_errors": [],
            "field_coverage": {},
            "document_types": {}
        }
        
        # Find all document files
        document_extensions = ['.txt', '.md']
        document_files = []
        for ext in document_extensions:
            document_files.extend(self.data_directory.glob(f"*{ext}"))
        
        report["total_documents"] = len(document_files)
        
        for doc_file in document_files:
            base_name = doc_file.stem
            labels_file = doc_file.with_suffix('.labels.json')
            
            if not labels_file.exists():
                report["missing_labels"].append(str(doc_file))
                report["invalid_pairs"] += 1
                continue
            
            try:
                # Load and validate labels
                with open(labels_file, 'r', encoding='utf-8') as f:
                    labels = json.load(f)
                
                self._validate_labels(labels)
                report["valid_pairs"] += 1
                
                # Track field coverage
                for field in labels.keys():
                    if field not in report["field_coverage"]:
                        report["field_coverage"][field] = 0
                    report["field_coverage"][field] += 1
                
                # Track document types
                doc_type = labels.get("document_type", "unknown")
                if doc_type not in report["document_types"]:
                    report["document_types"][doc_type] = 0
                report["document_types"][doc_type] += 1
                
            except Exception as e:
                report["validation_errors"].append({
                    "file": str(labels_file),
                    "error": str(e)
                })
                report["invalid_pairs"] += 1
        
        # Find orphaned labels files
        label_files = list(self.data_directory.glob("*.labels.json"))
        for labels_file in label_files:
            base_name = labels_file.stem.replace('.labels', '')
            doc_found = False
            for ext in document_extensions:
                if (self.data_directory / f"{base_name}{ext}").exists():
                    doc_found = True
                    break
            
            if not doc_found:
                report["missing_documents"].append(str(labels_file))
        
        return report
    
    def _validate_labels(self, labels: Dict[str, Any]):
        """Validate labels structure and content."""
        required_fields = ["document_type"]
        hybrid_fields = ["payment_terms", "contract_value", "delivery_terms", "compliance_requirements"]
        
        # Check required fields
        for field in required_fields:
            if field not in labels:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate hybrid fields structure
        for field in hybrid_fields:
            if field in labels:
                field_data = labels[field]
                if not isinstance(field_data, dict):
                    raise ValueError(f"Hybrid field {field} must be an object")
                
                if "extracted_text" not in field_data:
                    raise ValueError(f"Hybrid field {field} missing extracted_text")
                
                if "classification" not in field_data:
                    raise ValueError(f"Hybrid field {field} missing classification")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training data."""
        validation_report = self.validate_training_data()
        
        summary = {
            "data_directory": str(self.data_directory),
            "total_examples": validation_report["valid_pairs"],
            "invalid_examples": validation_report["invalid_pairs"],
            "field_coverage": validation_report["field_coverage"],
            "document_types": validation_report["document_types"],
            "most_common_fields": [],
            "coverage_percentage": {}
        }
        
        # Calculate most common fields
        if validation_report["field_coverage"]:
            sorted_fields = sorted(
                validation_report["field_coverage"].items(),
                key=lambda x: x[1],
                reverse=True
            )
            summary["most_common_fields"] = sorted_fields[:10]
            
            # Calculate coverage percentages
            total_examples = validation_report["valid_pairs"]
            if total_examples > 0:
                for field, count in validation_report["field_coverage"].items():
                    summary["coverage_percentage"][field] = (count / total_examples) * 100
        
        return summary
    
    def create_example_from_template(
        self,
        document_content: str,
        filename: str,
        payment_terms: str = "Net 30 days",
        contract_value: float = 50000,
        document_type: str = "contract"
    ) -> tuple[Path, Path]:
        """
        Create a training example from a template.
        
        Args:
            document_content: Content of the document
            filename: Base filename (without extension)
            payment_terms: Payment terms text
            contract_value: Contract value amount
            document_type: Type of document
            
        Returns:
            Tuple of (document_path, labels_path)
        """
        # Create document file
        doc_path = self.data_directory / f"{filename}.txt"
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write(document_content)
        
        # Create basic labels structure
        labels = {
            "payment_terms": {
                "extracted_text": payment_terms,
                "classification": self._classify_payment_terms(payment_terms),
                "standardized_value": self._standardize_payment_terms(payment_terms)
            },
            "contract_value": {
                "extracted_text": f"${contract_value:,}",
                "amount": contract_value,
                "currency": "USD",
                "classification": self._classify_contract_value(contract_value)
            },
            "document_type": document_type,
            "priority_level": "medium"
        }
        
        # Create labels file
        labels_path = self.create_labels_file(doc_path, labels)
        
        return doc_path, labels_path
    
    def _classify_payment_terms(self, payment_terms: str) -> str:
        """Classify payment terms based on content."""
        payment_terms_lower = payment_terms.lower()
        
        if any(term in payment_terms_lower for term in ["net 30", "net 15", "net 60", "due on receipt"]):
            return "standard"
        elif "days" in payment_terms_lower and any(char.isdigit() for char in payment_terms_lower):
            # Extract number of days
            import re
            days_match = re.search(r'(\d+)\s*days?', payment_terms_lower)
            if days_match:
                days = int(days_match.group(1))
                return "standard" if days <= 60 else "non_standard"
        
        return "non_standard"
    
    def _classify_contract_value(self, amount: float) -> str:
        """Classify contract value based on amount."""
        if amount < 10000:
            return "low_value"
        elif amount < 100000:
            return "medium_value"
        elif amount < 1000000:
            return "high_value"
        else:
            return "enterprise"
    
    def _standardize_payment_terms(self, payment_terms: str) -> str:
        """Standardize payment terms format."""
        payment_terms_lower = payment_terms.lower()
        
        if "net 30" in payment_terms_lower:
            return "Net 30"
        elif "net 15" in payment_terms_lower:
            return "Net 15"
        elif "net 60" in payment_terms_lower:
            return "Net 60"
        elif "due on receipt" in payment_terms_lower:
            return "Due on Receipt"
        else:
            return payment_terms.strip()
    
    def export_training_data(self, output_path: Union[str, Path]) -> Path:
        """
        Export training data as a single JSON file.
        
        Args:
            output_path: Path for exported JSON file
            
        Returns:
            Path to exported file
        """
        output_path = Path(output_path)
        training_data = []
        
        # Collect all valid training pairs
        validation_report = self.validate_training_data()
        
        for doc_file in self.data_directory.glob("*.txt"):
            labels_file = doc_file.with_suffix('.labels.json')
            
            if labels_file.exists():
                try:
                    # Load document
                    with open(doc_file, 'r', encoding='utf-8') as f:
                        document_content = f.read()
                    
                    # Load labels
                    with open(labels_file, 'r', encoding='utf-8') as f:
                        labels = json.load(f)
                    
                    training_data.append({
                        "input": document_content.strip(),
                        "output": labels,
                        "metadata": {
                            "document_file": doc_file.name,
                            "labels_file": labels_file.name,
                            "document_type": labels.get("document_type", "unknown")
                        }
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to export {doc_file}: {e}")
        
        # Write exported data
        export_data = {
            "training_examples": training_data,
            "metadata": {
                "total_examples": len(training_data),
                "export_source": str(self.data_directory),
                "validation_report": validation_report
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(training_data)} training examples to {output_path}")
        return output_path
