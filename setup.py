"""Setup script for MCGPURPTDosimetry package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="MCGPURPTDosimetry",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="GPU-Accelerated Internal Dosimetry Monte Carlo Calculation System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/devhliu/MCGPURPTDosimetry",
    packages=find_packages(),
    package_data={
        'MCGPURPTDosimetry': [
            'physics_data/decay_databases/*.json',
            'physics_data/cross_section_databases/*.h5',
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
