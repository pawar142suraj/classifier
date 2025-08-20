"""
Setup script for DocuVerse research library.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = requirements_path.read_text(encoding="utf-8").strip().split("\n")
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith("#")]

setup(
    name="docuverse",
    version="0.1.0",
    description="Advanced Document Information Extraction Research Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Research Team",
    author_email="research@docuverse.ai",
    url="https://github.com/yourusername/docuverse",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "research": [
            "jupyter>=1.0.0",
            "nbconvert>=7.0.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "plotly>=5.15.0",
        ],
        "full": [
            "graph-tool>=2.45",  # Advanced graph analysis
            "neo4j>=5.0.0",      # Graph database
            "transformers>=4.30.0",  # Advanced NLP models
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    keywords="document extraction, information extraction, RAG, graph neural networks, NLP, research",
    entry_points={
        "console_scripts": [
            "docuverse=docuverse.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "docuverse": ["schemas/*.json"],
    },
)
