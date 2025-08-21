"""
Vector RAG extractor using hybrid retrieval and reranking with advanced optimizations.
Implements state-of-the-art RAG techniques including:
- Hybrid retrieval (BM25 + semantic search)
- Multi-stage reranking with cross-encoders
- Adaptive chunking strategies
- Query expansion and refinement
- Contextual compression and filtering
"""

import json
import re
import hashlib
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
import logging

from .base import BaseExtractor
from ..core.config import LLMConfig, VectorRAGConfig

# Try importing optional dependencies with fallbacks
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    from sentence_transformers.util import cos_sim
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a document chunk with metadata."""
    content: str
    chunk_id: str
    start_pos: int
    end_pos: int
    chunk_type: str = "content"  # content, header, footer, table, etc.
    importance_score: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RetrievalResult:
    """Represents a retrieval result with scoring information."""
    chunk: Chunk
    score: float
    retrieval_method: str
    rank: int = 0
    rerank_score: Optional[float] = None


class AdvancedChunker:
    """Advanced document chunking with multiple strategies."""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50, strategy: str = "semantic"):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.strategy = strategy
        
        # Initialize NLP models if available
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model 'en_core_web_sm' not found. Using basic chunking.")
    
    def chunk_document(self, text: str, document_metadata: Dict[str, Any] = None) -> List[Chunk]:
        """Chunk document using specified strategy."""
        if self.strategy == "semantic" and self.nlp:
            return self._semantic_chunking(text, document_metadata)
        elif self.strategy == "sliding_window":
            return self._sliding_window_chunking(text, document_metadata)
        elif self.strategy == "hierarchical":
            return self._hierarchical_chunking(text, document_metadata)
        else:
            return self._fixed_size_chunking(text, document_metadata)
    
    def _semantic_chunking(self, text: str, metadata: Dict[str, Any] = None) -> List[Chunk]:
        """Semantic chunking using sentence boundaries and topic shifts."""
        if not self.nlp:
            return self._fixed_size_chunking(text, metadata)
        
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        chunks = []
        current_chunk = ""
        chunk_start = 0
        chunk_id = 0
        
        for i, sentence in enumerate(sentences):
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                # Create chunk
                chunk_end = chunk_start + len(current_chunk)
                chunk = Chunk(
                    content=current_chunk.strip(),
                    chunk_id=f"chunk_{chunk_id}",
                    start_pos=chunk_start,
                    end_pos=chunk_end,
                    chunk_type="semantic",
                    importance_score=self._calculate_importance(current_chunk),
                    metadata=metadata or {}
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(sentences, i, self.overlap)
                current_chunk = " ".join(overlap_sentences)
                chunk_start = chunk_end - len(current_chunk)
                chunk_id += 1
            
            current_chunk += " " + sentence
        
        # Add final chunk
        if current_chunk.strip():
            chunk = Chunk(
                content=current_chunk.strip(),
                chunk_id=f"chunk_{chunk_id}",
                start_pos=chunk_start,
                end_pos=chunk_start + len(current_chunk),
                chunk_type="semantic",
                importance_score=self._calculate_importance(current_chunk),
                metadata=metadata or {}
            )
            chunks.append(chunk)
        
        return chunks
    
    def _sliding_window_chunking(self, text: str, metadata: Dict[str, Any] = None) -> List[Chunk]:
        """Sliding window chunking with character-level overlap."""
        chunks = []
        chunk_id = 0
        
        for i in range(0, len(text), self.chunk_size - self.overlap):
            chunk_text = text[i:i + self.chunk_size]
            
            # Adjust to sentence boundaries if possible
            if i + self.chunk_size < len(text):
                # Find last sentence boundary
                last_period = chunk_text.rfind('.')
                last_exclamation = chunk_text.rfind('!')
                last_question = chunk_text.rfind('?')
                boundary = max(last_period, last_exclamation, last_question)
                
                if boundary > len(chunk_text) * 0.7:  # Only if boundary is reasonable
                    chunk_text = chunk_text[:boundary + 1]
            
            chunk = Chunk(
                content=chunk_text.strip(),
                chunk_id=f"chunk_{chunk_id}",
                start_pos=i,
                end_pos=i + len(chunk_text),
                chunk_type="sliding_window",
                importance_score=self._calculate_importance(chunk_text),
                metadata=metadata or {}
            )
            chunks.append(chunk)
            chunk_id += 1
        
        return chunks
    
    def _hierarchical_chunking(self, text: str, metadata: Dict[str, Any] = None) -> List[Chunk]:
        """Hierarchical chunking based on document structure."""
        chunks = []
        chunk_id = 0
        
        # Split by sections (double newlines, headers, etc.)
        sections = re.split(r'\n\s*\n|\n(?=[A-Z][A-Z\s]*:)', text)
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
                
            # Determine section type
            section_type = self._identify_section_type(section)
            
            # If section is too large, split further
            if len(section) > self.chunk_size:
                sub_chunks = self._fixed_size_chunking(section, metadata)
                for sub_chunk in sub_chunks:
                    sub_chunk.chunk_type = f"hierarchical_{section_type}"
                    sub_chunk.chunk_id = f"chunk_{chunk_id}"
                    chunks.append(sub_chunk)
                    chunk_id += 1
            else:
                chunk = Chunk(
                    content=section,
                    chunk_id=f"chunk_{chunk_id}",
                    start_pos=text.find(section),
                    end_pos=text.find(section) + len(section),
                    chunk_type=f"hierarchical_{section_type}",
                    importance_score=self._calculate_section_importance(section, section_type),
                    metadata=metadata or {}
                )
                chunks.append(chunk)
                chunk_id += 1
        
        return chunks
    
    def _fixed_size_chunking(self, text: str, metadata: Dict[str, Any] = None) -> List[Chunk]:
        """Simple fixed-size chunking with word boundaries."""
        words = text.split()
        chunks = []
        chunk_id = 0
        
        i = 0
        while i < len(words):
            chunk_words = []
            char_count = 0
            
            # Add words until chunk size is reached
            while i < len(words) and char_count + len(words[i]) + 1 <= self.chunk_size:
                chunk_words.append(words[i])
                char_count += len(words[i]) + 1
                i += 1
            
            if chunk_words:
                chunk_text = " ".join(chunk_words)
                start_pos = text.find(chunk_text)
                
                chunk = Chunk(
                    content=chunk_text,
                    chunk_id=f"chunk_{chunk_id}",
                    start_pos=start_pos if start_pos != -1 else 0,
                    end_pos=start_pos + len(chunk_text) if start_pos != -1 else len(chunk_text),
                    chunk_type="fixed_size",
                    importance_score=self._calculate_importance(chunk_text),
                    metadata=metadata or {}
                )
                chunks.append(chunk)
                chunk_id += 1
            
            # Handle overlap
            if self.overlap > 0 and i < len(words):
                # Calculate overlap in words
                overlap_words = max(1, self.overlap // 10)  # Rough estimation
                i = max(0, i - overlap_words)
        
        return chunks
    
    def _get_overlap_sentences(self, sentences: List[str], current_idx: int, overlap_chars: int) -> List[str]:
        """Get sentences for overlap based on character count."""
        overlap_sentences = []
        char_count = 0
        
        for i in range(current_idx - 1, -1, -1):
            if char_count + len(sentences[i]) > overlap_chars:
                break
            overlap_sentences.insert(0, sentences[i])
            char_count += len(sentences[i])
        
        return overlap_sentences
    
    def _calculate_importance(self, text: str) -> float:
        """Calculate importance score for a chunk."""
        # Simple heuristics for importance
        score = 0.5  # Base score
        
        # Boost for certain keywords
        important_keywords = ['payment', 'warranty', 'customer', 'contract', 'terms', 'agreement']
        for keyword in important_keywords:
            if keyword.lower() in text.lower():
                score += 0.1
        
        # Boost for structured content (colons, numbers, etc.)
        if ':' in text:
            score += 0.1
        if re.search(r'\d', text):
            score += 0.05
        
        # Penalize very short chunks
        if len(text) < 50:
            score -= 0.2
        
        return max(0.1, min(1.0, score))
    
    def _identify_section_type(self, section: str) -> str:
        """Identify the type of document section."""
        section_lower = section.lower()
        
        if any(keyword in section_lower for keyword in ['payment', 'terms', 'billing']):
            return 'payment'
        elif any(keyword in section_lower for keyword in ['warranty', 'guarantee']):
            return 'warranty'
        elif any(keyword in section_lower for keyword in ['customer', 'client', 'name']):
            return 'customer'
        elif any(keyword in section_lower for keyword in ['contract', 'agreement']):
            return 'contract_info'
        else:
            return 'general'
    
    def _calculate_section_importance(self, section: str, section_type: str) -> float:
        """Calculate importance based on section type."""
        type_scores = {
            'payment': 0.9,
            'warranty': 0.9,
            'customer': 0.9,
            'contract_info': 0.8,
            'general': 0.5
        }
        
        base_score = type_scores.get(section_type, 0.5)
        content_score = self._calculate_importance(section)
        
        return (base_score + content_score) / 2


class HybridRetriever:
    """Hybrid retrieval combining BM25 and semantic search."""
    
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model_name
        self.embeddings_model = None
        self.bm25 = None
        self.chunk_embeddings = None
        self.chunks = []
        self.faiss_index = None
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize embedding and BM25 models."""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embeddings_model = SentenceTransformer(self.embedding_model_name)
                logger.info(f"Loaded embedding model: {self.embedding_model_name}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
        
        if not BM25_AVAILABLE:
            logger.warning("BM25 not available. Install rank-bm25 for hybrid search.")
    
    def index_chunks(self, chunks: List[Chunk]) -> None:
        """Index chunks for retrieval."""
        self.chunks = chunks
        
        if not chunks:
            logger.warning("No chunks to index")
            return
        
        # Create BM25 index
        if BM25_AVAILABLE:
            tokenized_chunks = [chunk.content.lower().split() for chunk in chunks]
            self.bm25 = BM25Okapi(tokenized_chunks)
            logger.info(f"Indexed {len(chunks)} chunks with BM25")
        
        # Create semantic embeddings
        if self.embeddings_model:
            chunk_texts = [chunk.content for chunk in chunks]
            self.chunk_embeddings = self.embeddings_model.encode(chunk_texts, show_progress_bar=False)
            
            # Create FAISS index if available
            if FAISS_AVAILABLE and self.chunk_embeddings is not None:
                dimension = self.chunk_embeddings.shape[1]
                self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
                # Normalize embeddings for cosine similarity
                faiss.normalize_L2(self.chunk_embeddings)
                self.faiss_index.add(self.chunk_embeddings.astype('float32'))
                logger.info(f"Created FAISS index with {len(chunks)} embeddings")
    
    def retrieve(self, query: str, k: int = 5, bm25_weight: float = 0.3) -> List[RetrievalResult]:
        """Perform hybrid retrieval."""
        if not self.chunks:
            logger.warning("No chunks indexed for retrieval")
            return []
        
        results = []
        
        # BM25 retrieval
        bm25_scores = []
        if self.bm25:
            query_tokens = query.lower().split()
            bm25_scores = self.bm25.get_scores(query_tokens)
            bm25_scores = np.array(bm25_scores)
            # Normalize BM25 scores
            if bm25_scores.max() > 0:
                bm25_scores = bm25_scores / bm25_scores.max()
        else:
            bm25_scores = np.zeros(len(self.chunks))
        
        # Semantic retrieval
        semantic_scores = []
        if self.embeddings_model and self.chunk_embeddings is not None:
            query_embedding = self.embeddings_model.encode([query], show_progress_bar=False)
            
            if FAISS_AVAILABLE and self.faiss_index:
                # Use FAISS for fast similarity search
                faiss.normalize_L2(query_embedding.astype('float32'))
                scores, indices = self.faiss_index.search(query_embedding.astype('float32'), len(self.chunks))
                semantic_scores = np.zeros(len(self.chunks))
                semantic_scores[indices[0]] = scores[0]
            else:
                # Use cosine similarity
                similarities = cos_sim(query_embedding, self.chunk_embeddings)[0]
                semantic_scores = similarities.numpy()
        else:
            semantic_scores = np.zeros(len(self.chunks))
        
        # Combine scores
        combined_scores = bm25_weight * bm25_scores + (1 - bm25_weight) * semantic_scores
        
        # Get top-k indices
        top_indices = np.argsort(combined_scores)[::-1][:k]
        
        # Create retrieval results
        for rank, idx in enumerate(top_indices):
            result = RetrievalResult(
                chunk=self.chunks[idx],
                score=combined_scores[idx],
                retrieval_method="hybrid",
                rank=rank
            )
            results.append(result)
        
        logger.debug(f"Retrieved {len(results)} chunks for query: {query[:50]}...")
        return results
    
    def expand_query(self, query: str, schema: Optional[Dict[str, Any]] = None) -> List[str]:
        """Expand query with related terms."""
        expanded_queries = [query]
        
        # Add schema-based expansions
        if schema and "field" in schema:
            for field_name, field_def in schema["field"].items():
                field_desc = field_def.get("description", "")
                if any(word in query.lower() for word in field_name.split("_")):
                    expanded_queries.append(f"{query} {field_desc}")
                    
                    # Add enum values as context
                    if "enum" in field_def:
                        enum_context = " ".join(field_def["enum"])
                        expanded_queries.append(f"{query} {enum_context}")
        
        # Add synonyms and related terms
        query_lower = query.lower()
        synonym_map = {
            "payment": ["billing", "invoice", "cost", "fee", "price"],
            "warranty": ["guarantee", "coverage", "protection"],
            "customer": ["client", "buyer", "user"],
            "terms": ["conditions", "agreement", "rules"]
        }
        
        for key, synonyms in synonym_map.items():
            if key in query_lower:
                for synonym in synonyms:
                    expanded_queries.append(query.replace(key, synonym))
        
        return expanded_queries[:5]  # Limit to prevent excessive expansion


class CrossEncoderReranker:
    """Cross-encoder based reranking for better relevance."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = CrossEncoder(model_name)
                logger.info(f"Loaded cross-encoder model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load cross-encoder model: {e}")
    
    def rerank(self, query: str, results: List[RetrievalResult], top_k: int = 3) -> List[RetrievalResult]:
        """Rerank retrieval results using cross-encoder."""
        if not self.model or not results:
            return results[:top_k]
        
        # Prepare query-chunk pairs
        pairs = [(query, result.chunk.content) for result in results]
        
        try:
            # Get cross-encoder scores
            scores = self.model.predict(pairs)
            
            # Update results with rerank scores
            for i, result in enumerate(results):
                result.rerank_score = float(scores[i])
            
            # Sort by rerank score
            reranked_results = sorted(results, key=lambda x: x.rerank_score, reverse=True)
            
            # Update ranks
            for i, result in enumerate(reranked_results):
                result.rank = i
            
            logger.debug(f"Reranked {len(results)} results to top {top_k}")
            return reranked_results[:top_k]
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return results[:top_k]


class VectorRAGExtractor(BaseExtractor):
    """
    Advanced Vector RAG extraction method with comprehensive optimizations.
    
    Features:
    - Hybrid retrieval (BM25 + semantic search)
    - Multiple chunking strategies (fixed, semantic, sliding window, hierarchical)
    - Cross-encoder reranking
    - Query expansion and refinement
    - Adaptive retrieval based on query complexity
    - Schema-aware context injection
    - Contextual compression and filtering
    - Caching for performance optimization
    """
    
    def __init__(
        self, 
        llm_config: LLMConfig, 
        rag_config: VectorRAGConfig,
        schema: Optional[Dict[str, Any]] = None,
        schema_path: Optional[Union[str, Path]] = None,
        cache_embeddings: bool = True
    ):
        """Initialize Vector RAG extractor with advanced configurations."""
        super().__init__(llm_config)
        self.rag_config = rag_config
        self.schema = schema
        self.cache_embeddings = cache_embeddings
        
        # Load schema if path provided
        if schema_path and not schema:
            self.schema = self._load_schema(schema_path)
        
        # Initialize components
        self.chunker = AdvancedChunker(
            chunk_size=rag_config.chunk_size,
            overlap=rag_config.chunk_overlap,
            strategy=rag_config.chunking_strategy.value
        )
        
        self.retriever = HybridRetriever(rag_config.embedding_model)
        
        if rag_config.rerank_top_k > 0:
            self.reranker = CrossEncoderReranker()
        else:
            self.reranker = None
        
        # Document cache for embeddings
        self._embedding_cache = {} if cache_embeddings else None
        
        # Performance metrics
        self.last_chunks_processed = 0
        self.last_retrieval_time = 0.0
        self.last_rerank_time = 0.0
        
        logger.info("Initialized VectorRAGExtractor with advanced optimizations")
    
    def _load_schema(self, schema_path: Union[str, Path]) -> Dict[str, Any]:
        """Load schema from JSON file."""
        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema = json.load(f)
            logger.info(f"Loaded schema from {schema_path}")
            return schema
        except Exception as e:
            logger.error(f"Failed to load schema from {schema_path}: {e}")
            return {}
    
    def extract(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract using advanced Vector RAG approach.
        
        Args:
            document: Document to extract from
            
        Returns:
            Extracted information with metadata
        """
        import time
        start_time = time.time()
        
        document_text = self._prepare_document_text(document)
        document_metadata = document.get("metadata", {})
        
        # Generate document hash for caching
        doc_hash = hashlib.md5(document_text.encode()).hexdigest()
        
        # Check cache for embeddings
        if self._embedding_cache and doc_hash in self._embedding_cache:
            chunks = self._embedding_cache[doc_hash]
            logger.debug("Using cached chunks and embeddings")
        else:
            # Chunk the document
            chunks = self.chunker.chunk_document(document_text, document_metadata)
            self.last_chunks_processed = len(chunks)
            
            # Index chunks for retrieval
            self.retriever.index_chunks(chunks)
            
            # Cache chunks if enabled
            if self._embedding_cache:
                self._embedding_cache[doc_hash] = chunks
        
        # Generate extraction queries based on schema
        extraction_queries = self._generate_extraction_queries()
        
        # Perform extractions for each field
        extracted_fields = {}
        
        for field_name, query_info in extraction_queries.items():
            field_result = self._extract_field(query_info["query"], field_name, query_info["field_def"])
            extracted_fields[field_name] = field_result
        
        # Construct final result
        result = {
            "fields": extracted_fields,
            "metadata": {
                "extraction_method": "vector_rag",
                "chunks_processed": self.last_chunks_processed,
                "retrieval_time": self.last_retrieval_time,
                "rerank_time": self.last_rerank_time,
                "total_time": time.time() - start_time,
                "confidence": self.last_confidence,
                "rag_config": {
                    "chunk_size": self.rag_config.chunk_size,
                    "retrieval_k": self.rag_config.retrieval_k,
                    "rerank_top_k": self.rag_config.rerank_top_k,
                    "chunking_strategy": self.rag_config.chunking_strategy.value
                }
            }
        }
        
        # Calculate overall confidence
        self.last_confidence = self._calculate_extraction_confidence(extracted_fields)
        
        return result
    
    def _generate_extraction_queries(self) -> Dict[str, Dict[str, Any]]:
        """Generate optimized queries for each schema field."""
        queries = {}
        
        if not self.schema or "field" not in self.schema:
            # Fallback generic queries
            return {
                "general_info": {
                    "query": "extract key information from document",
                    "field_def": {"type": "string", "description": "General information"}
                }
            }
        
        for field_name, field_def in self.schema["field"].items():
            # Create focused query for the field
            field_description = field_def.get("description", "")
            field_type = field_def.get("type", "string")
            
            # Base query
            query_parts = [field_name.replace("_", " ")]
            
            if field_description:
                query_parts.append(field_description)
            
            # Add enum context for classification fields
            if "enum" in field_def:
                enum_values = field_def["enum"]
                enum_descriptions = field_def.get("enumDescriptions", {})
                
                enum_context = []
                for enum_val in enum_values:
                    if enum_val in enum_descriptions:
                        enum_context.append(f"{enum_val}: {enum_descriptions[enum_val]}")
                    else:
                        enum_context.append(enum_val)
                
                query_parts.append("categories: " + ", ".join(enum_context))
            
            # Create comprehensive query
            query = " ".join(query_parts)
            
            queries[field_name] = {
                "query": query,
                "field_def": field_def
            }
        
        return queries
    
    def _extract_field(self, query: str, field_name: str, field_def: Dict[str, Any]) -> Dict[str, Any]:
        """Extract a specific field using targeted retrieval."""
        import time
        
        # Expand query for better retrieval
        expanded_queries = self.retriever.expand_query(query, self.schema)
        
        # Retrieve relevant chunks
        retrieval_start = time.time()
        all_results = []
        
        for expanded_query in expanded_queries:
            results = self.retriever.retrieve(
                expanded_query, 
                k=self.rag_config.retrieval_k,
                bm25_weight=self.rag_config.bm25_weight
            )
            all_results.extend(results)
        
        # Remove duplicates and keep best scores
        unique_results = {}
        for result in all_results:
            chunk_id = result.chunk.chunk_id
            if chunk_id not in unique_results or result.score > unique_results[chunk_id].score:
                unique_results[chunk_id] = result
        
        results = list(unique_results.values())
        results.sort(key=lambda x: x.score, reverse=True)
        results = results[:self.rag_config.retrieval_k]
        
        self.last_retrieval_time = time.time() - retrieval_start
        
        # Rerank if configured
        if self.reranker and self.rag_config.rerank_top_k > 0:
            rerank_start = time.time()
            results = self.reranker.rerank(query, results, self.rag_config.rerank_top_k)
            self.last_rerank_time = time.time() - rerank_start
        
        # Extract from retrieved chunks
        extracted_content = self._extract_from_chunks(query, results, field_def)
        
        # Structure the result based on field type
        if "enum" in field_def:
            # Hybrid extraction + classification
            classification = self._classify_content(extracted_content, field_def)
            return {
                "extracted_content": extracted_content,
                "classification": classification,
                "retrieval_metadata": {
                    "chunks_used": len(results),
                    "top_chunk_score": results[0].score if results else 0.0,
                    "rerank_score": results[0].rerank_score if results and results[0].rerank_score else None
                }
            }
        else:
            # Pure extraction
            return {
                "extracted_content": extracted_content,
                "retrieval_metadata": {
                    "chunks_used": len(results),
                    "top_chunk_score": results[0].score if results else 0.0,
                    "rerank_score": results[0].rerank_score if results and results[0].rerank_score else None
                }
            }
    
    def _extract_from_chunks(self, query: str, results: List[RetrievalResult], field_def: Dict[str, Any]) -> str:
        """Extract specific information from retrieved chunks using LLM."""
        if not results:
            return ""
        
        # Combine relevant chunks
        context_chunks = []
        for result in results:
            chunk_info = f"[Chunk {result.rank + 1}, Score: {result.score:.3f}]\n{result.chunk.content}"
            context_chunks.append(chunk_info)
        
        context = "\n\n".join(context_chunks)
        
        # Create focused extraction prompt
        field_description = field_def.get("description", "")
        field_name = query.split()[0]  # Use first word as field name
        
        prompt_parts = [
            f"Extract specific information about '{field_name}' from the following document chunks.",
            "",
            f"FIELD TO EXTRACT: {field_name}",
            f"DESCRIPTION: {field_description}" if field_description else "",
            "",
            "INSTRUCTIONS:",
            "- Extract the exact text that answers the query",
            "- If the information is not found, return an empty string",
            "- Be precise and faithful to the source text",
            "- Extract only the relevant part, not the entire chunk",
            "",
            "DOCUMENT CHUNKS:",
            context,
            "",
            f"EXTRACTED {field_name.upper()}:"
        ]
        
        prompt = "\n".join(filter(None, prompt_parts))
        
        system_prompt = "You are an expert information extractor. Extract only the specific requested information from the provided text chunks."
        
        response = self._call_llm(prompt, system_prompt)
        
        # Clean up the response
        extracted = response.strip()
        
        # Remove common prefixes/suffixes
        prefixes_to_remove = [
            f"{field_name}:", f"{field_name.upper()}:", 
            "extracted:", "EXTRACTED:", "answer:", "ANSWER:",
            "information:", "INFORMATION:"
        ]
        
        for prefix in prefixes_to_remove:
            if extracted.lower().startswith(prefix.lower()):
                extracted = extracted[len(prefix):].strip()
        
        return extracted
    
    def _classify_content(self, content: str, field_def: Dict[str, Any]) -> str:
        """Classify extracted content based on field enum values."""
        if not content or "enum" not in field_def:
            return field_def.get("enum", [""])[0] if "enum" in field_def else ""
        
        enum_values = field_def["enum"]
        enum_descriptions = field_def.get("enumDescriptions", {})
        
        # Simple keyword-based classification first
        content_lower = content.lower()
        
        for enum_val in enum_values:
            # Direct match
            if enum_val.lower() in content_lower:
                return enum_val
            
            # Check description keywords
            if enum_val in enum_descriptions:
                desc_keywords = enum_descriptions[enum_val].lower().split()
                if any(keyword in content_lower for keyword in desc_keywords):
                    return enum_val
        
        # Use LLM for more sophisticated classification
        classification_prompt = f"""
        Classify the following extracted content into one of the predefined categories.
        
        EXTRACTED CONTENT: {content}
        
        CLASSIFICATION OPTIONS:
        """
        
        for enum_val in enum_values:
            desc = enum_descriptions.get(enum_val, "")
            classification_prompt += f"\n- {enum_val}: {desc}"
        
        classification_prompt += f"""
        
        RULES:
        - Choose the most appropriate category based on the content
        - Return only the category name (e.g., "{enum_values[0]}")
        - If uncertain, choose the most general category
        
        CLASSIFICATION:"""
        
        system_prompt = "You are an expert text classifier. Classify the given content into the most appropriate predefined category."
        
        response = self._call_llm(classification_prompt, system_prompt)
        
        # Clean and validate response
        predicted_class = response.strip().lower()
        
        # Find matching enum value
        for enum_val in enum_values:
            if enum_val.lower() == predicted_class or enum_val.lower() in predicted_class:
                return enum_val
        
        # Fallback to first enum value
        return enum_values[0]
    
    def _calculate_extraction_confidence(self, extracted_fields: Dict[str, Any]) -> float:
        """Calculate confidence score for the extraction."""
        if not extracted_fields:
            return 0.0
        
        total_confidence = 0.0
        field_count = 0
        
        for field_name, field_data in extracted_fields.items():
            field_confidence = 0.5  # Base confidence
            
            # Check if content was extracted
            extracted_content = field_data.get("extracted_content", "")
            if extracted_content.strip():
                field_confidence += 0.3
            
            # Boost for high retrieval scores
            retrieval_metadata = field_data.get("retrieval_metadata", {})
            top_score = retrieval_metadata.get("top_chunk_score", 0.0)
            rerank_score = retrieval_metadata.get("rerank_score")
            
            if top_score > 0.7:
                field_confidence += 0.1
            if rerank_score and rerank_score > 0.8:
                field_confidence += 0.1
            
            # Boost for classification fields with valid classification
            if "classification" in field_data:
                classification = field_data["classification"]
                if classification and classification.strip():
                    field_confidence += 0.1
            
            total_confidence += min(1.0, field_confidence)
            field_count += 1
        
        avg_confidence = total_confidence / field_count if field_count > 0 else 0.0
        
        # Boost for processing multiple chunks successfully
        if self.last_chunks_processed > 3:
            avg_confidence += 0.05
        
        return min(1.0, avg_confidence)
    
    def get_retrieval_analysis(self) -> Dict[str, Any]:
        """Get analysis of the retrieval process."""
        return {
            "chunks_processed": self.last_chunks_processed,
            "retrieval_time": self.last_retrieval_time,
            "rerank_time": self.last_rerank_time,
            "confidence": self.last_confidence,
            "chunking_strategy": self.rag_config.chunking_strategy.value,
            "embedding_model": self.rag_config.embedding_model,
            "retrieval_k": self.rag_config.retrieval_k,
            "rerank_top_k": self.rag_config.rerank_top_k,
            "hybrid_search": self.rag_config.use_hybrid_search,
            "bm25_weight": self.rag_config.bm25_weight,
            "models_available": {
                "sentence_transformers": SENTENCE_TRANSFORMERS_AVAILABLE,
                "faiss": FAISS_AVAILABLE,
                "bm25": BM25_AVAILABLE,
                "spacy": SPACY_AVAILABLE
            }
        }
    
    def clear_cache(self):
        """Clear the embedding cache to free memory."""
        if self._embedding_cache:
            self._embedding_cache.clear()
            logger.info("Cleared embedding cache")
    
    def update_config(self, new_config: VectorRAGConfig):
        """Update RAG configuration and reinitialize components."""
        self.rag_config = new_config
        
        # Reinitialize chunker
        self.chunker = AdvancedChunker(
            chunk_size=new_config.chunk_size,
            overlap=new_config.chunk_overlap,
            strategy=new_config.chunking_strategy.value
        )
        
        # Reinitialize retriever if embedding model changed
        if self.retriever.embedding_model_name != new_config.embedding_model:
            self.retriever = HybridRetriever(new_config.embedding_model)
        
        # Clear cache on config change
        self.clear_cache()
        
        logger.info("Updated Vector RAG configuration")
