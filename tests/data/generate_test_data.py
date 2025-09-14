"""
Test Data Generator for DocuVerse

This module generates test data for evaluating extraction methods.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import uuid


@dataclass
class TestDocument:
    """Represents a test document with ground truth."""
    name: str
    content: str
    ground_truth: Dict[str, Any]
    metadata: Dict[str, Any]


class TestDataGenerator:
    """Generates synthetic test data for DocuVerse evaluation."""
    
    def __init__(self):
        """Initialize the test data generator."""
        self.contract_templates = [
            {
                "type": "service_agreement",
                "template": """SERVICE AGREEMENT

This Service Agreement is entered into between {company_a} and {customer_name}.

TERMS:
- Payment Terms: {payment_terms}
- Warranty: {warranty}
- Service Period: {service_period}

Customer Information:
Company: {customer_name}
Address: {customer_address}

Agreed and signed,
{company_a}
""",
                "fields": {
                    "payment_terms": ["Payment due within 30 days", "Net 15 payment terms", "Payment due within 45 days", "Immediate payment required"],
                    "warranty": ["12-month limited warranty", "6-month full warranty", "2-year extended warranty", "No warranty provided"],
                    "service_period": ["1 year", "6 months", "2 years", "3 months"]
                }
            },
            {
                "type": "purchase_order",
                "template": """PURCHASE ORDER #{po_number}

From: {company_a}
To: {customer_name}

ORDER DETAILS:
- Payment Terms: {payment_terms}
- Warranty Coverage: {warranty}
- Delivery Date: {delivery_date}

BILLING INFORMATION:
Customer: {customer_name}
Billing Address: {customer_address}

Total Amount: ${amount}
""",
                "fields": {
                    "payment_terms": ["Net 30", "Net 15", "Cash on delivery", "Payment due in 60 days"],
                    "warranty": ["Limited 1-year warranty", "Full warranty coverage", "No warranty", "Extended 3-year warranty"],
                    "delivery_date": ["2024-03-15", "2024-04-20", "2024-02-28", "2024-05-10"],
                    "po_number": ["PO-2024-001", "PO-2024-002", "PO-2024-003", "PO-2024-004"],
                    "amount": ["5,000", "12,500", "8,750", "25,000"]
                }
            },
            {
                "type": "consulting_agreement",
                "template": """CONSULTING AGREEMENT

Agreement between {company_a} (Consultant) and {customer_name} (Client).

PAYMENT AND TERMS:
- Payment Schedule: {payment_terms}
- Warranty: {warranty}
- Project Duration: {project_duration}

CLIENT DETAILS:
Name: {customer_name}
Location: {customer_address}

This agreement is binding upon signature.

Consultant: {company_a}
Date: {agreement_date}
""",
                "fields": {
                    "payment_terms": ["Monthly invoicing", "Payment upon completion", "Weekly payments", "Quarterly billing"],
                    "warranty": ["Work guaranteed for 90 days", "No warranty on consulting services", "6-month warranty on deliverables", "Limited warranty applies"],
                    "project_duration": ["3 months", "6 months", "1 year", "18 months"],
                    "agreement_date": ["2024-01-15", "2024-02-01", "2024-03-10", "2024-04-05"]
                }
            }
        ]
        
        self.companies = [
            "TechCorp Ltd.", "DataSoft Inc.", "InnovateCo", "GlobalTech Solutions",
            "AlphaSystems", "BetaWorks LLC", "Gamma Enterprises", "Delta Technologies"
        ]
        
        self.customers = [
            {"name": "ClientCorp", "address": "123 Business Ave, Tech City"},
            {"name": "MegaCorp Inc.", "address": "456 Corporate Blvd, Metro City"},
            {"name": "StartupXYZ", "address": "789 Innovation St, Silicon Valley"},
            {"name": "Enterprise Solutions", "address": "321 Commerce Way, Business District"},
            {"name": "Global Industries", "address": "654 Industrial Pkwy, Manufacturing Zone"},
            {"name": "Digital Dynamics", "address": "987 Tech Lane, Innovation Hub"}
        ]
    
    def generate_test_documents(self, count: int = 10) -> List[TestDocument]:
        """Generate a set of test documents."""
        
        documents = []
        
        for i in range(count):
            # Select random template
            template = random.choice(self.contract_templates)
            
            # Generate variables
            company_a = random.choice(self.companies)
            customer = random.choice(self.customers)
            
            variables = {
                "company_a": company_a,
                "customer_name": customer["name"],
                "customer_address": customer["address"]
            }
            
            # Add template-specific fields
            for field, options in template["fields"].items():
                variables[field] = random.choice(options)
            
            # Generate content
            content = template["template"].format(**variables)
            
            # Create ground truth based on schema
            ground_truth = self._create_ground_truth(variables, template)
            
            # Create test document
            doc = TestDocument(
                name=f"test_document_{i+1}",
                content=content,
                ground_truth=ground_truth,
                metadata={
                    "template_type": template["type"],
                    "generated": True,
                    "variables": variables
                }
            )
            
            documents.append(doc)
        
        return documents
    
    def _create_ground_truth(self, variables: Dict[str, str], template: Dict[str, Any]) -> Dict[str, Any]:
        """Create ground truth labels based on variables and template."""
        
        ground_truth = {}
        
        # Extract payment terms with classification
        if "payment_terms" in variables:
            payment_value = variables["payment_terms"]
            
            # Classify payment terms
            if "30" in payment_value or "Net 30" in payment_value:
                classification = "standard"
            elif "15" in payment_value or "immediate" in payment_value.lower():
                classification = "expedited"
            elif "45" in payment_value or "60" in payment_value:
                classification = "extended"
            else:
                classification = "standard"
            
            ground_truth["payment_terms"] = {
                "value": payment_value,
                "classification": classification
            }
        
        # Extract warranty with classification
        if "warranty" in variables:
            warranty_value = variables["warranty"]
            
            # Classify warranty
            if "no warranty" in warranty_value.lower() or "no warranty" in warranty_value.lower():
                classification = "none"
            elif "limited" in warranty_value.lower():
                classification = "limited"
            else:
                classification = "full"
            
            ground_truth["warranty"] = {
                "value": warranty_value,
                "classification": classification
            }
        
        # Extract customer name (no classification needed)
        if "customer_name" in variables:
            ground_truth["customer_name"] = variables["customer_name"]
        
        # Add template-specific fields
        if template["type"] == "purchase_order":
            if "po_number" in variables:
                ground_truth["po_number"] = variables["po_number"]
            if "amount" in variables:
                ground_truth["total_amount"] = variables["amount"]
        
        elif template["type"] == "consulting_agreement":
            if "project_duration" in variables:
                ground_truth["project_duration"] = variables["project_duration"]
            if "agreement_date" in variables:
                ground_truth["agreement_date"] = variables["agreement_date"]
        
        return ground_truth
    
    def save_test_data(self, documents: List[TestDocument], output_dir: Path):
        """Save test documents and labels to files."""
        
        output_dir = Path(output_dir)
        data_dir = output_dir / "data"
        labels_dir = data_dir / "labels"
        
        # Create directories
        data_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        for doc in documents:
            # Save document content
            doc_path = data_dir / f"{doc.name}.txt"
            with open(doc_path, 'w') as f:
                f.write(doc.content)
            
            # Save ground truth labels
            label_path = labels_dir / f"{doc.name}_label.json"
            with open(label_path, 'w') as f:
                json.dump(doc.ground_truth, f, indent=2)
            
            # Save metadata
            metadata_path = labels_dir / f"{doc.name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(doc.metadata, f, indent=2)
    
    def create_evaluation_dataset(self, output_dir: Path, num_documents: int = 20):
        """Create a complete evaluation dataset."""
        
        print(f"Generating {num_documents} test documents...")
        
        # Generate documents
        documents = self.generate_test_documents(num_documents)
        
        # Save to files
        self.save_test_data(documents, output_dir)
        
        # Create dataset info
        dataset_info = {
            "name": "DocuVerse Evaluation Dataset",
            "description": "Synthetic dataset for evaluating extraction methods",
            "document_count": len(documents),
            "templates_used": list(set(doc.metadata["template_type"] for doc in documents)),
            "generated_timestamp": "2024-01-15T10:00:00Z",
            "schema_fields": [
                "payment_terms",
                "warranty", 
                "customer_name",
                "po_number",
                "total_amount",
                "project_duration",
                "agreement_date"
            ]
        }
        
        info_path = output_dir / "dataset_info.json"
        with open(info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"Dataset created with {len(documents)} documents")
        print(f"Files saved to: {output_dir}")
        
        return documents


def generate_sample_data():
    """Generate sample test data for development."""
    
    generator = TestDataGenerator()
    
    # Get base path
    base_path = Path(__file__).parent.parent.parent
    output_dir = base_path / "tests" / "data"
    
    # Generate test dataset
    documents = generator.create_evaluation_dataset(output_dir, num_documents=10)
    
    print("\nGenerated test documents:")
    for doc in documents:
        print(f"- {doc.name}: {doc.metadata['template_type']}")
    
    print(f"\nDataset saved to: {output_dir}")
    print("You can now run evaluations using this test data!")


if __name__ == "__main__":
    generate_sample_data()
