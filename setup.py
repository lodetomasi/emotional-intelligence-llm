"""
Setup configuration for Emotional Intelligence Framework
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ei-framework",
    version="1.0.0",
    author="Lorenzo De Tomasi",
    author_email="lorenzo.detomasi@graduate.univaq.it",
    description="A framework for evaluating emotional intelligence in Large Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lodetomasi/emotional-intelligence-llm",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.990",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "ei-evaluate=scripts.run_evaluation:main",
        ],
    },
    include_package_data=True,
    package_data={
        "ei_framework": ["config/*.json", "data/*.json"],
    },
)