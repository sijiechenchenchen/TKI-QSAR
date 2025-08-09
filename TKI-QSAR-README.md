# TKI-QSAR: Tyrosine Kinase Inhibitor QSAR Modeling

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![RDKit](https://img.shields.io/badge/RDKit-2022+-green.svg)](https://rdkit.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.0+-orange.svg)](https://pytorch.org)

A computational framework for predicting tyrosine kinase inhibitor (TKI) cellular uptake activity using machine learning-based Quantitative Structure-Activity Relationship (QSAR) modeling.

## 🎯 Project Overview

This project implements advanced QSAR modeling techniques to predict the cellular uptake activity of tyrosine kinase inhibitors, supporting drug-drug interaction studies and therapeutic efficacy optimization. The implementation compares Support Vector Machines (SVM) and Artificial Neural Networks (ANN) for molecular activity prediction.

## 🧬 Scientific Background

Tyrosine kinase inhibitors are crucial cancer therapeutics, but their cellular uptake and resulting therapeutic efficacy can be affected by drug-drug interactions with human transporter proteins (e.g., MATE1). This QSAR model helps predict molecular uptake properties based on structural features.

## 🔬 Methodology

### Molecular Featurization
- **Morgan Fingerprints**: Radius 2, 2048-bit vectors
- **Chemical Space Representation**: Molecular structures → numerical feature vectors
- **Data Processing**: Standardized scaling for machine learning compatibility

### Machine Learning Models

#### 1. Support Vector Classification (SVC)
- **Preprocessing**: StandardScaler normalization
- **Kernel**: RBF with automatic gamma selection
- **Optimization**: Grid search for hyperparameters

#### 2. Artificial Neural Network (ANN)
- **Architecture**: 4-layer deep neural network
  - Input Layer: 2048 neurons (Morgan fingerprint features)
  - Hidden Layer 1: 524 neurons (ReLU activation)
  - Hidden Layer 2: 10 neurons (ReLU activation)
  - Hidden Layer 3: 10 neurons (ReLU activation)
  - Output Layer: 2 neurons (Sigmoid activation)
- **Optimizer**: Adam
- **Loss Function**: Cross-entropy
- **Training**: 20 randomized iterations for statistical robustness

### Model Evaluation
- **Classification Threshold**: Activity < 10% (binary classification)
- **Validation**: 80/20 train-test split
- **Metrics**: Accuracy assessment across multiple training cycles
- **Statistical Robustness**: Randomized dataset shuffling

## 📁 Repository Structure

```
TKI-QSAR/
├── QSAR_SVM_ANN.py          # Main QSAR modeling script
├── sdf_tools.py             # Molecular data processing utilities
├── TKI2_uptake.csv          # TKI uptake activity dataset
├── tki_tested.sdf           # Molecular structure data (SDF format)
└── README.md                # Project documentation
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install rdkit-pypi torch scikit-learn pandas numpy
```

### Usage
```python
# Run QSAR analysis
python QSAR_SVM_ANN.py

# Calculate molecular similarity
from sdf_tools import get_cosin_similarity, molsfeaturizer, read_sdf_mol

# Load molecules and calculate similarity
molecules = read_sdf_mol('tki_tested.sdf')
features = molsfeaturizer(molecules)
similarity_scores = get_cosin_similarity(features, reference_features)
```

## 📊 Results Interpretation

The model outputs comparative performance metrics between SVM and ANN approaches:
- **Accuracy Scores**: Model performance across 20 randomized training cycles
- **Feature Importance**: Morgan fingerprint contribution to predictions
- **Molecular Similarity**: Cosine similarity scores for structure-activity relationships

## 🔬 Research Applications

- **Drug Discovery**: Predict TKI cellular uptake properties
- **DMPK Modeling**: Support drug metabolism and pharmacokinetics studies
- **Transporter Interaction Studies**: Investigate MATE1 and other transporter effects
- **Lead Optimization**: Guide medicinal chemistry design decisions

## 🛠️ Technical Details

### Dependencies
- **RDKit**: Molecular informatics and fingerprint generation
- **PyTorch**: Neural network implementation
- **Scikit-learn**: Traditional machine learning algorithms
- **Pandas/NumPy**: Data manipulation and numerical computing

### Data Format
- **Input**: SDF molecular structure files + CSV activity data
- **Features**: 2048-bit Morgan fingerprints
- **Output**: Binary classification (active/inactive)

## 📈 Performance Considerations

- **Computational Complexity**: O(n × m) where n = molecules, m = fingerprint bits
- **Memory Usage**: Optimized for datasets up to 10K molecules
- **Training Time**: ~1-5 minutes per iteration on standard hardware

## 🎓 Citation

If you use this work in your research, please cite:
```bibtex
@software{chen2023_tki_qsar,
  title={TKI-QSAR: Machine Learning-Based QSAR Modeling for Tyrosine Kinase Inhibitors},
  author={Chen, Sijie},
  year={2023},
  url={https://github.com/sijiechenchenchen/TKI-QSAR}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue for bugs and feature requests.

## 👨‍🔬 Author

**Sijie Chen**  
PhD Computational Scientist  
Cheminformatics AI/ML Scientist @ Bristol Myers Squibb  
📧 sijiechen070@gmail.com

---
*This project was developed as part of PhD research in Medicinal Chemistry and Pharmacognosy at The Ohio State University, supporting drug discovery efforts in oncology and immunology therapeutic areas.*