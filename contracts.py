# Contract Standardization Classifier using LangExtract with Ollama
# Simplified version - pass model manually, no setup checks

import langextract as lx
import textwrap
from typing import Dict, List, Any
import json

class ContractStandardizationClassifier:
    """
    A classifier that evaluates contract terms against standard definitions
    to identify standard vs non-standard clauses using Ollama local models
    """
    
    def __init__(self, standards_definition: Dict[str, Dict], model_name: str = "llama3.1:8b"):
        """
        Initialize with your standards definitions
        
        Args:
            standards_definition: Dictionary defining what's standard for each category
            model_name: Ollama model to use (e.g., "llama3.1:8b", "mistral", "codellama")
        """
        self.standards = standards_definition
        self.model_name = model_name
        self.model_id = f"{model_name}"
    
    def classify_contract(self, contract_text: str) -> Any:
        """
        Classify entire contract sections as standard/non-standard
        """
        prompt = self._build_classification_prompt()
        examples = self._build_classification_examples()
        print(f"Using Ollama model: {self.model_name}")
        result = lx.extract(
            text_or_documents=contract_text,
            prompt_description=prompt,
            examples=examples,
            language_model_type=lx.inference.OllamaLanguageModel,
            model_id=self.model_id,
            model_url="http://localhost:11434",  # Ollama local URL
            extraction_passes=1,
            max_workers=5,
            max_char_buffer=2000
        )
        
        return result
    
    def _build_classification_prompt(self) -> str:
        """Build the prompt with standards definitions"""
        
        standards_text = self._format_standards_for_prompt()
        
        prompt = textwrap.dedent(f"""\
        You are a contract analysis expert. Analyze the contract text and classify each clause/section 
        according to the provided standards definitions.
        
        STANDARDS DEFINITIONS:
        {standards_text}
        
        INSTRUCTIONS:
        1. Extract each distinct clause/section related to the categories above
        2. Classify each as "standard" or "non-standard" based on the definitions
        3. Provide specific reasoning for non-standard classifications
        4. Extract the exact text of each clause
        5. Identify the category (insurance, warranty, termination, etc.)
        
        For each extraction:
        - Use exact text from the contract
        - Clearly state if it's standard or non-standard
        - Explain WHY it's non-standard if applicable
        - Reference which standard criteria it violates or meets
        """)
        
        return prompt
    
    def _format_standards_for_prompt(self) -> str:
        """Format standards definitions for the prompt"""
        formatted = []
        
        for category, standards in self.standards.items():
            formatted.append(f"\n{category.upper()}:")
            formatted.append(f"  Standard Requirements: {standards.get('standard_requirements', 'Not defined')}")
            formatted.append(f"  Non-Standard Indicators: {standards.get('non_standard_indicators', 'Not defined')}")
            formatted.append(f"  Key Terms: {standards.get('key_terms', 'Not defined')}")
        
        return "\n".join(formatted)
    
    def _build_classification_examples(self) -> List[lx.data.ExampleData]:
        """Build training examples for the model"""
        
        examples = [
            lx.data.ExampleData(
                text="""
                The Contractor shall maintain comprehensive general liability insurance 
                with minimum coverage of $1,000,000 per occurrence and $2,000,000 aggregate. 
                Insurance shall name Client as additional insured and provide 30 days written notice of cancellation.
                """,
                extractions=[
                    lx.data.Extraction(
                        extraction_class="insurance_clause",
                        extraction_text="comprehensive general liability insurance with minimum coverage of $1,000,000 per occurrence and $2,000,000 aggregate",
                        attributes={
                            "category": "insurance",
                            "classification": "standard",
                            "reasoning": "Meets standard requirements for liability coverage amounts",
                            "coverage_amount": "$1,000,000/$2,000,000",
                            "compliance_status": "compliant"
                        }
                    ),
                    lx.data.Extraction(
                        extraction_class="insurance_requirement",
                        extraction_text="name Client as additional insured and provide 30 days written notice of cancellation",
                        attributes={
                            "category": "insurance",
                            "classification": "standard",
                            "reasoning": "Includes standard additional insured and notice requirements",
                            "notice_period": "30 days"
                        }
                    )
                ]
            ),
            lx.data.ExampleData(
                text="""
                Contractor warrants all work for a period of 90 days from completion. 
                Warranty excludes damage caused by Client misuse or normal wear and tear.
                """,
                extractions=[
                    lx.data.Extraction(
                        extraction_class="warranty_clause",
                        extraction_text="Contractor warrants all work for a period of 90 days from completion",
                        attributes={
                            "category": "warranty",
                            "classification": "non-standard",
                            "reasoning": "90 days is below standard 12-month warranty period",
                            "warranty_period": "90 days",
                            "standard_period": "12 months",
                            "deviation": "insufficient_period"
                        }
                    ),
                    lx.data.Extraction(
                        extraction_class="warranty_exclusion",
                        extraction_text="excludes damage caused by Client misuse or normal wear and tear",
                        attributes={
                            "category": "warranty",
                            "classification": "standard",
                            "reasoning": "Standard exclusions for misuse and normal wear",
                            "exclusion_type": "standard_exclusions"
                        }
                    )
                ]
            ),
            lx.data.ExampleData(
                text="""
                Either party may terminate this agreement with 15 days written notice. 
                Upon termination, all work product becomes property of Client regardless of payment status.
                """,
                extractions=[
                    lx.data.Extraction(
                        extraction_class="termination_clause",
                        extraction_text="Either party may terminate this agreement with 15 days written notice",
                        attributes={
                            "category": "termination",
                            "classification": "non-standard",
                            "reasoning": "15 days is below standard 30-day notice period",
                            "notice_period": "15 days",
                            "standard_period": "30 days",
                            "deviation": "insufficient_notice"
                        }
                    ),
                    lx.data.Extraction(
                        extraction_class="work_product_clause",
                        extraction_text="all work product becomes property of Client regardless of payment status",
                        attributes={
                            "category": "intellectual_property",
                            "classification": "non-standard",
                            "reasoning": "Standard practice requires payment before IP transfer",
                            "ip_transfer_condition": "regardless_of_payment",
                            "standard_condition": "upon_payment",
                            "risk_level": "high"
                        }
                    )
                ]
            )
        ]
        
        return examples

def create_standards_definition() -> Dict[str, Dict]:
    """
    Define what constitutes standard vs non-standard for each contract category
    Customize this based on your organization's standards
    """
    
    standards = {
        "insurance": {
            "standard_requirements": [
                "General liability minimum $1M per occurrence, $2M aggregate",
                "Professional liability minimum $1M per claim",
                "Additional insured coverage for client",
                "30-day cancellation notice",
                "Primary and non-contributory coverage"
            ],
            "non_standard_indicators": [
                "Coverage below minimum amounts",
                "Missing additional insured",
                "Cancellation notice less than 30 days",
                "Exclusions for professional services",
                "Self-insurance without adequate reserves"
            ],
            "key_terms": ["liability", "insurance", "coverage", "additional insured", "cancellation"]
        },
        
        "warranty": {
            "standard_requirements": [
                "Minimum 12-month warranty period",
                "Warranty covers defects in materials and workmanship",
                "Standard exclusions only (misuse, normal wear)",
                "Clear repair/replacement process",
                "No limitation of consequential damages"
            ],
            "non_standard_indicators": [
                "Warranty period less than 12 months",
                "Excessive exclusions or limitations",
                "Limited to repair only (no replacement)",
                "Client responsible for warranty service costs",
                "Warranty void for any modification"
            ],
            "key_terms": ["warranty", "defects", "workmanship", "materials", "repair", "replacement"]
        },
        
        "termination": {
            "standard_requirements": [
                "30-day written notice for convenience termination",
                "Immediate termination for material breach after cure period",
                "7-day cure period for breaches",
                "Payment for work performed through termination date",
                "Return of confidential information"
            ],
            "non_standard_indicators": [
                "Termination notice less than 30 days",
                "No cure period for breaches",
                "Forfeiture of payment for completed work",
                "Termination without cause restrictions",
                "Excessive termination fees"
            ],
            "key_terms": ["termination", "notice", "breach", "cure period", "convenience"]
        },
        
        "payment": {
            "standard_requirements": [
                "Net 30 payment terms",
                "Progress payments based on milestones",
                "Interest on late payments (1.5% per month max)",
                "Right to suspend work for non-payment",
                "No payment terms longer than 60 days"
            ],
            "non_standard_indicators": [
                "Payment terms longer than 60 days",
                "Payment contingent on end-customer payment",
                "No interest on late payments",
                "Excessive withholding percentages",
                "Pay-when-paid clauses"
            ],
            "key_terms": ["payment", "net", "days", "interest", "late", "milestone"]
        },
        
        "limitation_of_liability": {
            "standard_requirements": [
                "Liability cap at contract value or $100K minimum",
                "Mutual liability limitations",
                "Standard carve-outs (IP infringement, confidentiality breach)",
                "No limitation on gross negligence or willful misconduct",
                "Separate cap for IP indemnity"
            ],
            "non_standard_indicators": [
                "Liability cap below $100K",
                "One-sided limitations favoring only client",
                "No carve-outs for standard exclusions",
                "Limitations on gross negligence",
                "Combined cap for all claims"
            ],
            "key_terms": ["liability", "limitation", "cap", "damages", "negligence", "indemnity"]
        },
        
        "intellectual_property": {
            "standard_requirements": [
                "Client owns deliverables upon full payment",
                "Contractor retains pre-existing IP",
                "Standard IP indemnification provisions",
                "Clear work-for-hire language",
                "License to use pre-existing tools/methodologies"
            ],
            "non_standard_indicators": [
                "Automatic IP transfer regardless of payment",
                "Client claims to pre-existing contractor IP",
                "Excessive IP indemnification scope",
                "Broad work-for-hire definitions",
                "No license for contractor tools"
            ],
            "key_terms": ["intellectual property", "IP", "ownership", "work for hire", "license", "indemnification"]
        }
    }
    
    return standards

def analyze_contract(contract_text: str, model_name: str = "llama3.1:8b") -> Any:
    """
    Analyze a contract and generate standardization report
    
    Args:
        contract_text: The contract text to analyze
        model_name: Ollama model to use (e.g., "llama3.1:8b", "mistral:7b")
    """
    standards = create_standards_definition()
    classifier = ContractStandardizationClassifier(standards, model_name)
    
    print(f"Analyzing contract using {model_name}...")
    result = classifier.classify_contract(contract_text)
    
    return result

def generate_standardization_report(result: Any, contract_name: str = "Contract") -> None:
    """
    Generate a detailed standardization report
    """
    
    print(f"\n{'='*60}")
    print(f"CONTRACT STANDARDIZATION ANALYSIS REPORT")
    print(f"Contract: {contract_name}")
    print(f"{'='*60}")
    
    # Categorize findings
    standard_clauses = []
    non_standard_clauses = []
    categories = {}
    
    for doc in result.documents:
        for extraction in doc.extractions:
            classification = extraction.attributes.get('classification', 'unknown')
            category = extraction.attributes.get('category', 'uncategorized')
            
            # Group by category
            if category not in categories:
                categories[category] = {'standard': 0, 'non_standard': 0}
            categories[category][classification] += 1
            
            if classification == 'standard':
                standard_clauses.append(extraction)
            elif classification == 'non_standard':
                non_standard_clauses.append(extraction)
    
    # Summary statistics
    total_clauses = len(standard_clauses) + len(non_standard_clauses)
    compliance_rate = (len(standard_clauses) / total_clauses * 100) if total_clauses > 0 else 0
    
    print(f"\nSUMMARY:")
    print(f"Total Clauses Analyzed: {total_clauses}")
    print(f"Standard Clauses: {len(standard_clauses)}")
    print(f"Non-Standard Clauses: {len(non_standard_clauses)}")
    print(f"Compliance Rate: {compliance_rate:.1f}%")
    
    # Category breakdown
    print(f"\nCATEGORY BREAKDOWN:")
    for category, counts in categories.items():
        total_cat = counts['standard'] + counts['non_standard']
        cat_compliance = (counts['standard'] / total_cat * 100) if total_cat > 0 else 0
        print(f"  {category.upper()}: {cat_compliance:.1f}% compliant ({counts['standard']}/{total_cat})")
    
    # Non-standard clause details
    if non_standard_clauses:
        print(f"\nNON-STANDARD CLAUSES REQUIRING REVIEW:")
        print(f"{'-'*50}")
        
        for i, clause in enumerate(non_standard_clauses, 1):
            category = clause.attributes.get('category', 'Unknown')
            reasoning = clause.attributes.get('reasoning', 'No reasoning provided')
            risk_level = clause.attributes.get('risk_level', 'medium')
            
            print(f"\n{i}. {category.upper()} - Risk Level: {risk_level.upper()}")
            print(f"   Text: \"{clause.extraction_text[:100]}{'...' if len(clause.extraction_text) > 100 else ''}\"")
            print(f"   Issue: {reasoning}")
            
            # Additional context if available
            for key, value in clause.attributes.items():
                if key not in ['classification', 'category', 'reasoning', 'risk_level']:
                    print(f"   {key.replace('_', ' ').title()}: {value}")
    
    # Recommendations
    print(f"\nRECOMMENDATIONS:")
    print(f"{'-'*30}")
    
    high_risk_count = sum(1 for clause in non_standard_clauses 
                         if clause.attributes.get('risk_level') == 'high')
    
    if high_risk_count > 0:
        print(f"🚨 HIGH PRIORITY: {high_risk_count} high-risk non-standard clause(s) require immediate attention")
    
    if compliance_rate < 80:
        print(f"⚠️  MEDIUM PRIORITY: Overall compliance rate ({compliance_rate:.1f}%) is below recommended 80%")
    
    if compliance_rate >= 90:
        print(f"✅ LOW RISK: Good compliance rate ({compliance_rate:.1f}%). Review non-standard clauses as time permits")

def save_results(result: Any, filename: str = "contract_analysis"):
    """
    Save results and create visualization
    """
    # Save results to JSONL format
    lx.io.save_annotated_documents([result], output_name=f"{filename}.jsonl")
    
    # Generate interactive HTML visualization
    html_content = lx.visualize(f"{filename}.jsonl")
    with open(f"{filename}.html", "w") as f:
        f.write(html_content)
    
    print(f"Results saved to {filename}.jsonl and {filename}.html")

# Example usage
def test_sample_contract(model_name: str = "llama3.1:8b"):
    """
    Test the classifier with a sample contract
    """
    
    sample_contract = textwrap.dedent("""\
    PROFESSIONAL SERVICES AGREEMENT
    
    1. INSURANCE: Contractor shall maintain general liability insurance with minimum 
    coverage of $500,000 per occurrence. Insurance shall name Client as additional 
    insured with 15 days notice of cancellation.
    
    2. WARRANTY: Contractor warrants all deliverables for 6 months from delivery. 
    This warranty excludes any defects caused by Client modifications or third-party integrations.
    
    3. TERMINATION: Either party may terminate this agreement with 10 days written notice. 
    Upon termination, Contractor forfeits all unpaid amounts for work completed.
    
    4. PAYMENT: Client shall pay all invoices within Net 45 days. Late payments 
    will not accrue interest. Payment is due regardless of Client's receipt of 
    payment from their end customers.
    
    5. LIABILITY: Contractor's liability is limited to $50,000 total for all claims 
    arising from this agreement, including gross negligence and willful misconduct.
    
    6. INTELLECTUAL PROPERTY: All work product, including any improvements to 
    Contractor's pre-existing methodologies, shall become the sole property of Client 
    immediately upon creation, regardless of payment status.
    """)
    
    # Run analysis
    result = analyze_contract(sample_contract, model_name)
    
    # Generate report
    generate_standardization_report(result, "Sample Contract")
    
    # Save results
    save_results(result, "sample_contract_analysis")
    
    return result

if __name__ == "__main__":
    # Simple usage - just pass the model name
    try:
        # Test with sample contract using llama3.1:8b
        result = test_sample_contract("llama3.1:8b")
        print(f"\nAnalysis complete! Found {len(result.documents[0].extractions)} total extractions.")
        
        # Example of analyzing your own contract:
        # with open("your_contract.txt", "r") as f:
        #     contract_text = f.read()
        # result = analyze_contract(contract_text, "llama3.1:8b")
        # generate_standardization_report(result, "Your Contract")
        # save_results(result, "your_contract_analysis")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure:")
        print("1. Ollama is running: ollama serve")
        print("2. Model is installed: ollama pull llama3.1:8b")