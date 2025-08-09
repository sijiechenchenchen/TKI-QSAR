"""
Setup script for TKI-QSAR package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="tki-qsar",
    version="1.0.0",
    author="Sijie Chen",
    author_email="sijiechen070@gmail.com",
    description="Machine Learning-Based QSAR Modeling for Tyrosine Kinase Inhibitors",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sijiechenchenchen/TKI-QSAR",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    keywords="qsar, machine-learning, drug-discovery, cheminformatics, tyrosine-kinase-inhibitors",
    project_urls={
        "Bug Reports": "https://github.com/sijiechenchenchen/TKI-QSAR/issues",
        "Source": "https://github.com/sijiechenchenchen/TKI-QSAR",
        "Documentation": "https://github.com/sijiechenchenchen/TKI-QSAR#readme",
    },
)