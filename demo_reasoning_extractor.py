#!/usr/bin/env python3
"""
Demonstration script for the Reasoning Extractor with Chain of Thought and ReAct methodologies.
Shows how the reasoning extractor builds on Vector RAG for enhanced contract information extraction.
"""

import json
import time
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from docuverse.core.config import (
    LLMConfig, ReasoningConfig, VectorRAGConfig, 
    ExtractionMethod, ChunkingStrategy
)
from docuverse.extractors.reasoning import ReasoningExtractor
from docuverse.extractors.vector_rag import VectorRAGExtractor


def demo_reasoning_extractors():
    """Demonstrate Reasoning extractors with CoT and ReAct methodologies."""
    
    print("🧠 DocuVerse: Reasoning-Enhanced Contract Extraction Demo")
    print("=" * 65)
    
    # Load schema
    schema_path = Path(__file__).parent / "schemas" / "contract_schema.json"
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    
    # Load test document
    contract_path = Path(__file__).parent / "data" / "contract1.txt"
    with open(contract_path, 'r') as f:
        contract_content = f.read()
    
    document = {
        "content": contract_content,
        "metadata": {
            "filename": "contract1.txt",
            "source": "demo"
        }
    }
    
    print(f"📄 Processing: {contract_path.name}")
    print(f"📊 Document length: {len(contract_content)} characters")
    print(f"📋 Schema fields: {list(schema['properties'].keys())}")
    
    # Configure LLM
    llm_config = LLMConfig(
        provider="ollama",
        model_name="llama3.2:latest",
        ollama_base_url="http://localhost:11434",
        temperature=0.1,
        max_tokens=4096
    )
    
    # Configure reasoning
    reasoning_config = ReasoningConfig(
        use_cot=True,
        use_react=True,
        max_reasoning_steps=5,
        verification_enabled=True,
        auto_repair_enabled=True,
        uncertainty_threshold=0.7
    )
    
    print(f"\n⚙️ Configuration:")
    print(f"  • LLM: {llm_config.provider.value} ({llm_config.model_name})")
    print(f"  • Reasoning: CoT & ReAct enabled")
    print(f"  • Verification: {reasoning_config.verification_enabled}")
    print(f"  • Auto-repair: {reasoning_config.auto_repair_enabled}")
    print(f"  • Uncertainty threshold: {reasoning_config.uncertainty_threshold}")
    
    # Test Chain of Thought Reasoning
    print(f"\n🔗 Testing Chain of Thought (CoT) Reasoning")
    print("-" * 50)
    
    try:
        cot_extractor = ReasoningExtractor(
            llm_config=llm_config,
            reasoning_config=reasoning_config,
            method_type=ExtractionMethod.REASONING_COT,
            schema=schema,
            use_vector_rag=True
        )
        
        print(f"✅ Initialized CoT extractor with Vector RAG integration")
        
        start_time = time.time()
        cot_result = cot_extractor.extract(document)
        cot_time = time.time() - start_time
        
        cot_analysis = cot_extractor.get_reasoning_analysis()
        
        print(f"⏱️ Processing time: {cot_time:.2f}s")
        print(f"📊 Overall confidence: {cot_analysis['overall_confidence']:.3f}")
        print(f"🧠 Reasoning steps: {cot_analysis['total_steps']}")
        print(f"📋 Evidence pieces: {cot_analysis['evidence_pieces']}")
        print(f"🔍 Vector RAG enabled: {cot_analysis['vector_rag_enabled']}")
        
        print(f"\n📋 CoT Extraction Results:")
        for field_name, field_value in cot_result.get("fields", {}).items():
            if isinstance(field_value, dict) and len(str(field_value)) > 100:
                print(f"  • {field_name}: [Complex object]")
            else:
                value_str = str(field_value)[:80]
                print(f"  • {field_name}: {value_str}{'...' if len(str(field_value)) > 80 else ''}")
        
        print(f"\n🧠 Reasoning Step Breakdown:")
        step_breakdown = cot_analysis['step_breakdown']
        for step_type, count in step_breakdown.items():
            print(f"  • {step_type.title()}: {count}")
        
        print(f"\n📊 Evidence Analysis:")
        for field_name, evidence in cot_analysis['evidence_by_field'].items():
            conf = evidence['confidence']
            confidence_emoji = "🟢" if conf > 0.8 else "🟡" if conf > 0.6 else "🔴"
            print(f"  • {field_name}: {confidence_emoji} {conf:.3f}")
        
    except Exception as e:
        print(f"❌ CoT Reasoning failed: {e}")
        cot_result = None
        cot_time = 0
        cot_analysis = {}
    
    # Test ReAct Reasoning
    print(f"\n⚡ Testing ReAct (Reasoning + Acting) Methodology")
    print("-" * 50)
    
    try:
        react_extractor = ReasoningExtractor(
            llm_config=llm_config,
            reasoning_config=reasoning_config,
            method_type=ExtractionMethod.REASONING_REACT,
            schema=schema,
            use_vector_rag=True
        )
        
        print(f"✅ Initialized ReAct extractor with Vector RAG integration")
        
        start_time = time.time()
        react_result = react_extractor.extract(document)
        react_time = time.time() - start_time
        
        react_analysis = react_extractor.get_reasoning_analysis()
        
        print(f"⏱️ Processing time: {react_time:.2f}s")
        print(f"📊 Overall confidence: {react_analysis['overall_confidence']:.3f}")
        print(f"🧠 Reasoning steps: {react_analysis['total_steps']}")
        print(f"📋 Evidence pieces: {react_analysis['evidence_pieces']}")
        print(f"🔍 Vector RAG enabled: {react_analysis['vector_rag_enabled']}")
        
        print(f"\n📋 ReAct Extraction Results:")
        for field_name, field_value in react_result.get("fields", {}).items():
            if isinstance(field_value, dict) and len(str(field_value)) > 100:
                print(f"  • {field_name}: [Complex object]")
            else:
                value_str = str(field_value)[:80]
                print(f"  • {field_name}: {value_str}{'...' if len(str(field_value)) > 80 else ''}")
        
        print(f"\n🧠 ReAct Step Breakdown:")
        step_breakdown = react_analysis['step_breakdown']
        for step_type, count in step_breakdown.items():
            print(f"  • {step_type.title()}: {count}")
        
        print(f"\n📊 Evidence Analysis:")
        for field_name, evidence in react_analysis['evidence_by_field'].items():
            conf = evidence['confidence']
            confidence_emoji = "🟢" if conf > 0.8 else "🟡" if conf > 0.6 else "🔴"
            print(f"  • {field_name}: {confidence_emoji} {conf:.3f}")
        
    except Exception as e:
        print(f"❌ ReAct Reasoning failed: {e}")
        react_result = None
        react_time = 0
        react_analysis = {}
    
    # Comparison with Vector RAG baseline
    print(f"\n📊 Comparison with Vector RAG Baseline")
    print("-" * 45)
    
    try:
        # Test basic Vector RAG for comparison
        rag_config = VectorRAGConfig(
            chunk_size=384,
            chunk_overlap=50,
            chunking_strategy=ChunkingStrategy.SEMANTIC,
            retrieval_k=5,
            rerank_top_k=3
        )
        
        vector_rag_extractor = VectorRAGExtractor(
            llm_config=llm_config,
            rag_config=rag_config,
            schema=schema
        )
        
        start_time = time.time()
        rag_result = vector_rag_extractor.extract(document)
        rag_time = time.time() - start_time
        
        rag_analysis = vector_rag_extractor.get_retrieval_analysis()
        
        print(f"⏱️ Processing times:")
        print(f"  • Vector RAG: {rag_time:.2f}s")
        if cot_result:
            print(f"  • CoT Reasoning: {cot_time:.2f}s ({cot_time/rag_time:.1f}x slower)")
        if react_result:
            print(f"  • ReAct Reasoning: {react_time:.2f}s ({react_time/rag_time:.1f}x slower)")
        
        print(f"\n📊 Confidence comparison:")
        print(f"  • Vector RAG: {rag_analysis['confidence']:.3f}")
        if cot_analysis:
            print(f"  • CoT Reasoning: {cot_analysis['overall_confidence']:.3f}")
        if react_analysis:
            print(f"  • ReAct Reasoning: {react_analysis['overall_confidence']:.3f}")
        
        # Determine winner
        methods = []
        if rag_result:
            methods.append(("Vector RAG", rag_analysis['confidence'], rag_time))
        if cot_result:
            methods.append(("CoT Reasoning", cot_analysis['overall_confidence'], cot_time))
        if react_result:
            methods.append(("ReAct Reasoning", react_analysis['overall_confidence'], react_time))
        
        if methods:
            best_method = max(methods, key=lambda x: x[1])  # By confidence
            fastest_method = min(methods, key=lambda x: x[2])  # By time
            
            print(f"\n🏆 Performance Winners:")
            print(f"  • Highest confidence: {best_method[0]} ({best_method[1]:.3f})")
            print(f"  • Fastest processing: {fastest_method[0]} ({fastest_method[2]:.2f}s)")
        
    except Exception as e:
        print(f"❌ Vector RAG baseline failed: {e}")
    
    # Recommendations
    print(f"\n🎯 Method Recommendations:")
    print(f"  • Use CoT for complex documents requiring detailed analysis")
    print(f"  • Use ReAct for iterative refinement and high accuracy")
    print(f"  • Use Vector RAG for speed when basic extraction suffices")
    print(f"  • Reasoning methods provide better evidence tracking")
    print(f"  • Enable verification for critical applications")
    
    # Show detailed reasoning trace (if available)
    if cot_result and 'reasoning_trace' in cot_result.get('metadata', {}):
        print(f"\n🔍 Detailed CoT Reasoning Trace (first 3 steps):")
        trace = cot_result['metadata']['reasoning_trace'][:3]
        for step in trace:
            step_type = step['type'].title()
            content = step['content'][:100] + "..." if len(step['content']) > 100 else step['content']
            print(f"  Step {step['step']}: {step_type}")
            print(f"    Content: {content}")
            print(f"    Confidence: {step['confidence']:.3f}")
            if step['uncertainty_flags']:
                print(f"    Uncertainties: {step['uncertainty_flags']}")
            print()
    
    print(f"\n🏁 Reasoning Extraction Demo completed!")
    print(f"📖 The reasoning extractors show enhanced analytical capabilities")
    print(f"📊 Trade-off: Higher processing time for better reasoning and evidence")


if __name__ == "__main__":
    demo_reasoning_extractors()
