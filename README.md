# TKI-QSAR: Machine Learning Models for MATE1 Inhibition by Tyrosine Kinase Inhibitors

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![RDKit](https://img.shields.io/badge/RDKit-2022+-green.svg)](https://rdkit.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.0+-orange.svg)](https://pytorch.org)

A computational framework implementing machine learning-based QSAR modeling to predict tyrosine kinase inhibitor (TKI) inhibition of the MATE1 transporter, based on the research published in *Pharmaceutics* (2021).

## 🎯 Project Overview

This repository contains the implementation of machine learning models used to classify tyrosine kinase inhibitors as active or inactive inhibitors of the MATE1 (Multidrug and Toxin Extrusion 1) transporter. The work supports understanding of drug-drug interactions and transporter-mediated cellular uptake mechanisms.

## 🧬 Scientific Background

MATE1 is a crucial renal and hepatic transporter responsible for the elimination of organic cations including many drugs. Out of 57 tested TKIs, 37 showed potent MATE1 inhibition (>80%), occurring through a non-competitive, reversible, substrate-independent mechanism. Understanding these interactions is critical for predicting drug-drug interactions in cancer therapy.

## 📊 Research Findings

- **High Inhibition Rate**: 65% of tested TKIs potently inhibited MATE1
- **Mechanism**: Non-competitive, reversible, substrate-independent inhibition  
- **Model Performance**: ML-QSAR models achieved q² values of 0.83 and 0.75
- **Clinical Relevance**: Findings inform potential drug-drug interactions in oncology

## 🔬 Methodology

### Molecular Featurization
- **Morgan Circular Fingerprints**: Chemical structure conversion to numerical features
- **Binary Classification**: Active (>80% inhibition) vs Inactive (≤80% inhibition)
- **Data Processing**: Standardized molecular descriptors for ML compatibility

### Machine Learning Models

#### 1. Artificial Neural Network (ANN)
- Multi-layer perceptron architecture
- Morgan fingerprint input features
- Binary classification output
- Cross-validated performance assessment

#### 2. Support Vector Machine (SVM) 
- RBF kernel implementation
- Feature scaling and normalization
- Grid search hyperparameter optimization
- Robust classification performance

#### 3. Additional Methods
- Linear Discriminant Analysis (LDA)
- t-distributed Stochastic Neighbor Embedding (t-SNE)
- Comparative model evaluation

## 📁 Repository Structure

```
TKI-QSAR/
├── QSAR_SVM_ANN.py          # Main QSAR modeling script
├── sdf_tools.py             # Molecular data processing utilities  
├── TKI2_uptake.csv          # TKI MATE1 inhibition dataset
├── tki_tested.sdf           # Molecular structure data (SDF format)
├── TKI-QSAR-requirements.txt # Python dependencies
├── TKI-QSAR-setup.py        # Package setup script
└── README.md                # This documentation
```

## 🚀 Quick Start

### Installation
```bash
pip install -r TKI-QSAR-requirements.txt
```

### Usage
```python
# Run complete QSAR analysis
python QSAR_SVM_ANN.py

# Use molecular processing utilities
from sdf_tools import molsfeaturizer, read_sdf_mol, get_cosin_similarity

# Load and process molecular data
molecules = read_sdf_mol('tki_tested.sdf')
features = molsfeaturizer(molecules)
```

## 📈 Model Performance

The implemented models demonstrate strong predictive capability:
- **ANN Model**: High accuracy with robust cross-validation
- **SVM Model**: Excellent classification performance
- **Statistical Validation**: Multiple randomized training cycles
- **Feature Importance**: Morgan fingerprint contribution analysis

## 🔬 Research Applications

- **Drug Discovery**: Predict TKI-MATE1 interactions early in development
- **Clinical Pharmacology**: Assess drug-drug interaction potential
- **Transporter Biology**: Understand MATE1 inhibition mechanisms
- **Regulatory Science**: Support safety assessment of new TKIs

## 🛠️ Technical Requirements

### Dependencies
- **RDKit**: Molecular informatics and fingerprint generation
- **PyTorch**: Neural network implementation  
- **Scikit-learn**: Traditional ML algorithms
- **Pandas/NumPy**: Data manipulation and analysis

### Input Data Format
- **Molecular Structures**: SDF file format
- **Activity Data**: CSV with MATE1 inhibition percentages
- **Output**: Binary classification predictions

## 📚 Citation

This work is based on the research published in:

**Uddin, M.E., Garrison, D.A., Kim, K. et al.** *In Vitro and In Vivo Inhibition of MATE1 by Tyrosine Kinase Inhibitors.* Pharmaceutics 13, 2107 (2021). https://doi.org/10.3390/pharmaceutics13122107

```bibtex
@article{uddin2021mate1,
  title={In Vitro and In Vivo Inhibition of MATE1 by Tyrosine Kinase Inhibitors},
  author={Uddin, Muhammad Erfan and Garrison, David A and Kim, Kyeong-Hun and others},
  journal={Pharmaceutics},
  volume={13},
  number={12},
  pages={2107},
  year={2021},
  publisher={MDPI},
  doi={10.3390/pharmaceutics13122107}
}
```

## 📄 License

This project is licensed under the terms specified in the TKI-QSAR-LICENSE file.

## 🤝 Contributing

Contributions are welcome! Please submit issues or pull requests for improvements to the modeling approaches or additional analysis methods.

## 👨‍🔬 Author

**Sijie Chen**  
Computational Scientist  
Based on research from The Ohio State University College of Pharmacy

---

*This computational framework implements the machine learning methodology described in the peer-reviewed research on TKI-MATE1 interactions, supporting further research in drug transporter biology and clinical pharmacology.*