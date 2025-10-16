#!/usr/bin/env python
"""
Setup script for HBT Analysis package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="hbt-analysis",
    version="1.0.0",
    author="HBT-EP Analysis Team",
    author_email="your-email@example.com",
    description="A comprehensive package for HBT-EP analysis using machine learning and GPU optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/HBT-EP-Boeckmann",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "tensorflow>=2.8.0",
        "numpy>=1.21.0",
        "Pillow>=8.0.0",
        "matplotlib>=3.5.0",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "gpu": [
            "tensorflow-gpu>=2.8.0",
        ],
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
    },
    include_package_data=True,
    package_data={
        "hbt_analysis": ["*.md", "*.txt"],
    },
)
