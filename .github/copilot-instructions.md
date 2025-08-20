<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# DocuVerse Development Guidelines

## Project Overview
This is a research library for advanced document information extraction that systematically evaluates and compares different methods including few-shot baselines, vector RAG, graph RAG, and novel dynamic graph-RAG approaches.

## Code Standards
- Follow PEP 8 for Python code formatting
- Use type hints throughout the codebase
- Implement comprehensive error handling
- Write docstrings for all public methods and classes
- Use Pydantic models for data validation and schema definition

## Architecture Principles
- **Modular Design**: Each extraction method should be implemented as a separate, swappable component
- **Research-First**: All components should support evaluation, benchmarking, and ablation studies
- **Extensible**: New methods should be easy to add without modifying existing code
- **Reproducible**: All experiments should be deterministic and reproducible

## Key Components
- `extractors/`: Implement different extraction methods (few-shot, vector RAG, graph RAG, etc.)
- `evaluation/`: Metrics, benchmarking, and comparison utilities
- `rag/`: RAG-specific implementations (retrieval, reranking, chunking)
- `graph/`: Graph processing and knowledge graph utilities
- `reasoning/`: CoT, ReAct, and verification implementations

## Testing Requirements
- Unit tests for all core functionality
- Integration tests for end-to-end extraction pipelines
- Benchmark tests for performance evaluation
- Property-based testing for robustness

## Documentation
- Comprehensive API documentation
- Research methodology explanations
- Performance benchmarking results
- Example usage for each extraction method
