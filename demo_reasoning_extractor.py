#!/usr/bin/env python3
"""
Demonstration script for the Reasoning Extractor with Chain of Thought and ReAct methodologies.
Shows how the reasoning extractor builds on Vector RAG for enhanced contract information extraction.
"""

import json
import time
import sys
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from docuverse.core.config import (
    LLMConfig, ReasoningConfig, VectorRAGConfig, 
    ExtractionMethod, ChunkingStrategy
)
from docuverse.extractors.reasoning import ReasoningExtractor
from docuverse.extractors.vector_rag import VectorRAGExtractor


def save_results_to_output(rag_result, cot_result, react_result, rag_analysis, cot_analysis, react_analysis, timings):
    """Save all extraction results to output folder for debugging."""
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / "output" / f"reasoning_extraction_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving results to: {output_dir}")
    
    # Save Vector RAG results
    if rag_result:
        rag_output = {
            "method": "vector_rag",
            "timestamp": timestamp,
            "processing_time": timings.get("rag_time", 0),
            "extraction_result": rag_result,
            "analysis": rag_analysis,
            "metadata": {
                "description": "Vector RAG baseline extraction results",
                "confidence": rag_analysis.get("confidence", 0),
                "chunks_processed": rag_analysis.get("chunks_processed", 0)
            }
        }
        
        with open(output_dir / "vector_rag_result.json", 'w', encoding='utf-8') as f:
            json.dump(rag_output, f, indent=2, ensure_ascii=False)
        print(f"  ✅ Saved Vector RAG results")
    
    # Save CoT results
    if cot_result:
        cot_output = {
            "method": "chain_of_thought",
            "timestamp": timestamp,
            "processing_time": timings.get("cot_time", 0),
            "extraction_result": cot_result,
            "reasoning_analysis": cot_analysis,
            "metadata": {
                "description": "Chain of Thought reasoning extraction results",
                "confidence": cot_analysis.get("overall_confidence", 0),
                "reasoning_steps": cot_analysis.get("total_steps", 0),
                "evidence_pieces": cot_analysis.get("evidence_pieces", 0)
            }
        }
        
        with open(output_dir / "cot_reasoning_result.json", 'w', encoding='utf-8') as f:
            json.dump(cot_output, f, indent=2, ensure_ascii=False)
        print(f"  ✅ Saved CoT reasoning results")
        
        # Save detailed reasoning trace if available
        if 'reasoning_trace' in cot_result.get('metadata', {}):
            trace_output = {
                "method": "chain_of_thought",
                "timestamp": timestamp,
                "reasoning_trace": cot_result['metadata']['reasoning_trace'],
                "evidence_summary": cot_result['metadata'].get('evidence_summary', []),
                "step_breakdown": cot_analysis.get('step_breakdown', {}),
                "evidence_by_field": cot_analysis.get('evidence_by_field', {})
            }
            
            with open(output_dir / "cot_detailed_trace.json", 'w', encoding='utf-8') as f:
                json.dump(trace_output, f, indent=2, ensure_ascii=False)
            print(f"  ✅ Saved CoT detailed reasoning trace")
    
    # Save ReAct results
    if react_result:
        react_output = {
            "method": "react_reasoning",
            "timestamp": timestamp,
            "processing_time": timings.get("react_time", 0),
            "extraction_result": react_result,
            "reasoning_analysis": react_analysis,
            "metadata": {
                "description": "ReAct (Reasoning + Acting) extraction results",
                "confidence": react_analysis.get("overall_confidence", 0),
                "reasoning_steps": react_analysis.get("total_steps", 0),
                "evidence_pieces": react_analysis.get("evidence_pieces", 0)
            }
        }
        
        with open(output_dir / "react_reasoning_result.json", 'w', encoding='utf-8') as f:
            json.dump(react_output, f, indent=2, ensure_ascii=False)
        print(f"  ✅ Saved ReAct reasoning results")
        
        # Save detailed reasoning trace if available
        if 'reasoning_trace' in react_result.get('metadata', {}):
            trace_output = {
                "method": "react_reasoning",
                "timestamp": timestamp,
                "reasoning_trace": react_result['metadata']['reasoning_trace'],
                "evidence_summary": react_result['metadata'].get('evidence_summary', []),
                "step_breakdown": react_analysis.get('step_breakdown', {}),
                "evidence_by_field": react_analysis.get('evidence_by_field', {})
            }
            
            with open(output_dir / "react_detailed_trace.json", 'w', encoding='utf-8') as f:
                json.dump(trace_output, f, indent=2, ensure_ascii=False)
            print(f"  ✅ Saved ReAct detailed reasoning trace")
    
    # Save comparative analysis
    comparison_output = {
        "timestamp": timestamp,
        "comparison_summary": {
            "methods_tested": [],
            "performance_metrics": {},
            "recommendations": []
        },
        "detailed_comparison": {},
        "timings": timings
    }
    
    # Add method summaries
    if rag_result:
        comparison_output["comparison_summary"]["methods_tested"].append("vector_rag")
        comparison_output["detailed_comparison"]["vector_rag"] = {
            "confidence": rag_analysis.get("confidence", 0),
            "processing_time": timings.get("rag_time", 0),
            "chunks_processed": rag_analysis.get("chunks_processed", 0),
            "retrieval_time": rag_analysis.get("retrieval_time", 0)
        }
    
    if cot_result:
        comparison_output["comparison_summary"]["methods_tested"].append("chain_of_thought")
        comparison_output["detailed_comparison"]["chain_of_thought"] = {
            "confidence": cot_analysis.get("overall_confidence", 0),
            "processing_time": timings.get("cot_time", 0),
            "reasoning_steps": cot_analysis.get("total_steps", 0),
            "evidence_pieces": cot_analysis.get("evidence_pieces", 0)
        }
    
    if react_result:
        comparison_output["comparison_summary"]["methods_tested"].append("react_reasoning")
        comparison_output["detailed_comparison"]["react_reasoning"] = {
            "confidence": react_analysis.get("overall_confidence", 0),
            "processing_time": timings.get("react_time", 0),
            "reasoning_steps": react_analysis.get("total_steps", 0),
            "evidence_pieces": react_analysis.get("evidence_pieces", 0)
        }
    
    # Add performance analysis
    all_methods = []
    if rag_result:
        all_methods.append(("Vector RAG", rag_analysis.get("confidence", 0), timings.get("rag_time", 0)))
    if cot_result:
        all_methods.append(("CoT Reasoning", cot_analysis.get("overall_confidence", 0), timings.get("cot_time", 0)))
    if react_result:
        all_methods.append(("ReAct Reasoning", react_analysis.get("overall_confidence", 0), timings.get("react_time", 0)))
    
    if all_methods:
        best_confidence = max(all_methods, key=lambda x: x[1])
        fastest_method = min(all_methods, key=lambda x: x[2])
        
        comparison_output["comparison_summary"]["performance_metrics"] = {
            "highest_confidence": {"method": best_confidence[0], "confidence": best_confidence[1]},
            "fastest_processing": {"method": fastest_method[0], "time": fastest_method[2]},
            "total_methods": len(all_methods)
        }
    
    # Add recommendations
    comparison_output["comparison_summary"]["recommendations"] = [
        "Use CoT for complex documents requiring detailed analysis",
        "Use ReAct for iterative refinement and high accuracy",
        "Use Vector RAG for speed when basic extraction suffices",
        "Reasoning methods provide better evidence tracking",
        "Enable verification for critical applications"
    ]
    
    with open(output_dir / "comparative_analysis.json", 'w', encoding='utf-8') as f:
        json.dump(comparison_output, f, indent=2, ensure_ascii=False)
    print(f"  ✅ Saved comparative analysis")
    
    # Save a summary README for easy navigation
    readme_content = f"""# Reasoning Extraction Results - {timestamp}

## Overview
This folder contains detailed results from the reasoning extraction demo run on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.

## Files Generated

### Core Results
- `vector_rag_result.json` - Vector RAG baseline extraction results
- `cot_reasoning_result.json` - Chain of Thought reasoning results
- `react_reasoning_result.json` - ReAct reasoning results
- `comparative_analysis.json` - Performance comparison and metrics

### Detailed Traces
- `cot_detailed_trace.json` - Complete CoT reasoning trace with evidence
- `react_detailed_trace.json` - Complete ReAct reasoning trace with evidence

## Quick Stats
"""
    
    if rag_result:
        readme_content += f"- **Vector RAG**: {rag_analysis.get('confidence', 0):.3f} confidence, {timings.get('rag_time', 0):.2f}s\n"
    if cot_result:
        readme_content += f"- **CoT Reasoning**: {cot_analysis.get('overall_confidence', 0):.3f} confidence, {timings.get('cot_time', 0):.2f}s\n"
    if react_result:
        readme_content += f"- **ReAct Reasoning**: {react_analysis.get('overall_confidence', 0):.3f} confidence, {timings.get('react_time', 0):.2f}s\n"
    
    readme_content += """
## Usage for Debugging

1. **Examine extraction results**: Check the main result files for extracted fields and classifications
2. **Trace reasoning steps**: Look at detailed trace files to understand the reasoning process
3. **Compare performance**: Use comparative_analysis.json to understand trade-offs
4. **Debug issues**: Look at evidence_by_field in traces to see supporting text for each extraction

## File Structure
Each result file contains:
- `extraction_result`: The actual extracted fields and metadata
- `reasoning_analysis`/`analysis`: Performance metrics and analysis
- `metadata`: Summary information and confidence scores
- `reasoning_trace`: (For reasoning methods) Step-by-step reasoning process
"""
    
    with open(output_dir / "README.md", 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"  ✅ Saved README with navigation guide")
    
    print(f"\n📁 Output saved to: {output_dir}")
    print(f"📖 See README.md in output folder for file descriptions")
    
    return output_dir


def demo_reasoning_extractors():
    """Demonstrate Reasoning extractors with CoT and ReAct methodologies."""
    
    print("🧠 DocuVerse: Reasoning-Enhanced Contract Extraction Demo")
    print("=" * 65)
    
    # Load schema
    schema_path = Path(__file__).parent / "schemas" / "contracts_schema_hybrid.json"
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
    print(f"📋 Schema fields: {list(schema['field'].keys())}")
    
    # Configure LLM
    llm_config = LLMConfig(
        provider="ollama",
        model_name="llama3.2:latest",
        ollama_base_url="http://localhost:11434",
        temperature=0.1,
        max_tokens=4096,
        timeout=300,  # Increase timeout to 5 minutes (300 seconds)
        max_retries=3,  # Retry up to 3 times on failure
        retry_delay=2.0  # Wait 2 seconds between retries
    )
    
    # Configure reasoning
    reasoning_config = ReasoningConfig(
        use_cot=True,
        use_react=True,
        max_reasoning_steps=3,  # Reduced from 5 to speed up processing
        verification_enabled=False,  # Disable for faster processing
        auto_repair_enabled=False,  # Disable for faster processing
        uncertainty_threshold=0.6  # Lower threshold for faster acceptance
    )
    
    print(f"\n⚙️ Configuration:")
    print(f"  • LLM: {llm_config.provider.value} ({llm_config.model_name})")
    print(f"  • Reasoning: CoT & ReAct enabled")
    print(f"  • Verification: {reasoning_config.verification_enabled}")
    print(f"  • Auto-repair: {reasoning_config.auto_repair_enabled}")
    print(f"  • Uncertainty threshold: {reasoning_config.uncertainty_threshold}")
    
    # Initialize variables to store results
    cot_result = None
    react_result = None
    rag_result = None
    cot_analysis = {}
    react_analysis = {}
    rag_analysis = {}
    timings = {}
    
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
        timings["cot_time"] = cot_time
        
        cot_analysis = cot_extractor.get_reasoning_analysis()
        
        print(f"⏱️ Processing time: {cot_time:.2f}s")
        print(f"📊 Overall confidence: {cot_analysis['overall_confidence']:.3f}")
        print(f"🧠 Reasoning steps: {cot_analysis['total_steps']}")
        print(f"📋 Evidence pieces: {cot_analysis['evidence_pieces']}")
        print(f"🔍 Vector RAG enabled: {cot_analysis['vector_rag_enabled']}")
        
        print(f"\n📋 CoT Extraction Results:")
        for field_name, field_value in cot_result.get("fields", {}).items():
            if isinstance(field_value, dict) and len(str(field_value)) > 100:
                print(f"  • {field_name}: {field_value}[Complex object]")
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
        timings["cot_time"] = 0
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
        timings["react_time"] = react_time
        
        react_analysis = react_extractor.get_reasoning_analysis()
        
        print(f"⏱️ Processing time: {react_time:.2f}s")
        print(f"📊 Overall confidence: {react_analysis['overall_confidence']:.3f}")
        print(f"🧠 Reasoning steps: {react_analysis['total_steps']}")
        print(f"📋 Evidence pieces: {react_analysis['evidence_pieces']}")
        print(f"🔍 Vector RAG enabled: {react_analysis['vector_rag_enabled']}")
        
        print(f"\n📋 ReAct Extraction Results:")
        for field_name, field_value in react_result.get("fields", {}).items():
            if isinstance(field_value, dict):
                extracted_content = field_value.get("extracted_content", "")
                classification = field_value.get("classification", "")
                if extracted_content:
                    print(f"  • {field_name}: {extracted_content}")
                    if classification:
                        print(f"    Classification: {classification}")
                else:
                    print(f"  • {field_name}: [No content extracted]")
                    if classification:
                        print(f"    Classification: {classification}")
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
        timings["react_time"] = 0
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
        timings["rag_time"] = rag_time
        
        rag_analysis = vector_rag_extractor.get_retrieval_analysis()
        
        print(f"⏱️ Processing times:")
        print(f"  • Vector RAG: {rag_time:.2f}s")
        if cot_result:
            print(f"  • CoT Reasoning: {timings['cot_time']:.2f}s ({timings['cot_time']/rag_time:.1f}x slower)")
        if react_result:
            print(f"  • ReAct Reasoning: {timings['react_time']:.2f}s ({timings['react_time']/rag_time:.1f}x slower)")
        
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
        rag_result = None
        rag_analysis = {}
        timings["rag_time"] = 0
    
    # Recommendations
    print(f"\n🎯 Method Recommendations:")
    print(f"  • Use CoT for complex documents requiring detailed analysis")
    print(f"  • Use ReAct for iterative refinement and high accuracy")
    print(f"  • Use Vector RAG for speed when basic extraction suffices")
    print(f"  • Reasoning methods provide better evidence tracking")
    print(f"  • Enable verification for critical applications")
    
    # Show detailed reasoning trace (if available) and save it in output log
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
    
    # Save all results to output folder for debugging
    try:
        output_dir = save_results_to_output(
            rag_result, cot_result, react_result, 
            rag_analysis, cot_analysis, react_analysis, 
            timings
        )
        print(f"\n🎯 Debug files saved successfully!")
        print(f"📂 Location: {output_dir}")
        print(f"🔍 Use these files to debug extraction traces and analyze performance")
    except Exception as e:
        print(f"\n⚠️ Failed to save debug files: {e}")
        print(f"💡 Results are still available in memory for immediate inspection")


if __name__ == "__main__":
    demo_reasoning_extractors()
