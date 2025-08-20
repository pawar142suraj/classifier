"""
DocuVerse: Advanced Document Information Extraction Research Library

A comprehensive framework for evaluating and comparing different document
information extraction methods, from baseline approaches to novel techniques.
"""

__version__ = "0.1.0"
__author__ = "Research Team"
__email__ = "research@docuverse.ai"

from .core.extractor import DocumentExtractor
from .core.config import ExtractionConfig
from .evaluation.evaluator import Evaluator

__all__ = [
    "DocumentExtractor",
    "ExtractionConfig", 
    "Evaluator",
]
