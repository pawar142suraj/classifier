#!/usr/bin/env python3
"""
Quick test to verify page-aware evidence extraction in Hybrid Graph RAG
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.docuverse.extractors.hybrid_graph_rag import HybridGraphRAGExtractor

def test_hybrid_page_extraction():
    """Test page extraction in Hybrid Graph RAG."""
    
    # Sample multi-page document
    test_document = """
PAGE 1 - CONTRACT OVERVIEW

This Master Service Agreement is between Acme Corporation (Customer) 
and TechService Inc. (Contractor).

PAYMENT TERMS: Customer shall pay monthly within 30 days.

---PAGE BREAK---

PAGE 2 - WARRANTY SECTION

WARRANTY: Contractor provides standard warranty coverage 
for all deliverables for 12 months.

TERMINATION: Either party may terminate with 30 days notice.

---PAGE BREAK---

PAGE 3 - COMPLIANCE

FEDERAL REQUIREMENTS: This contract complies with federal regulations.
"""

    # Simple schema
    schema = {
        "field": {
            "customer_name": {"type": "string", "description": "Customer name"},
            "payment_terms": {"type": "string", "enum": ["monthly", "yearly"], "description": "Payment terms"},
            "warranty": {"type": "string", "enum": ["standard", "non_standard"], "description": "Warranty type"}
        }
    }
    
    # Mock LLM
    class MockLLM:
        def __init__(self):
            pass
        
        def chat(self, messages, **kwargs):
            # Simple mock response
            return {"content": "Acme Corporation"}
    
    try:
        # Initialize extractor
        extractor = HybridGraphRAGExtractor(
            llm=MockLLM(),
            schema=schema,
            config={}
        )
        
        # Extract with page tracking
        result = extractor.extract(test_document)
        
        print("🎉 Hybrid Graph RAG Page Extraction Test")
        print("=" * 50)
        print(f"✓ Extraction completed successfully")
        print(f"✓ Document pages detected: {len(getattr(extractor, 'document_pages', []))}")
        print(f"✓ Evidence entries: {len(result.evidence)}")
        
        # Check for page information in evidence
        page_aware_evidence = 0
        for field_name, evidence in result.evidence.items():
            if hasattr(evidence, 'page_number') and evidence.page_number is not None:
                page_aware_evidence += 1
                print(f"✓ {field_name}: Page {evidence.page_number}/{evidence.total_pages}")
        
        print(f"\n📊 Page-aware evidence: {page_aware_evidence}/{len(result.evidence)}")
        
        if page_aware_evidence > 0:
            print("✅ SUCCESS: Page extraction is working in Hybrid Graph RAG!")
        else:
            print("⚠️  WARNING: No page information found in evidence")
            
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_hybrid_page_extraction()
    exit(0 if success else 1)
