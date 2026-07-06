# MLA Pipelines - Machine Learning Analysis

Complete machine learning pipelines for EEG signal classification with proper evaluation methodologies to prevent data leakage.

## Overview

This directory contains two specialized MLA pipelines:

1. **EDF Pipeline** (`Computational/`) - For real EEG data from Emotiv Epoc+
2. **BrainFlow Pipeline** (`Phantom/`) - For phantom/Emotiv data from BrainFlow

Both pipelines address critical flaws in common EEG-ML workflows, particularly data leakage from improper train/test splitting.

## Quick Usage

### EDF Pipeline (Real EEG)

```bash
cd MLA/Computational
python EEG_MLA_Complete_EDF.py
```

**Input:** EDF files in `data/` directory  
**Labeling:** `*_1.edf` = Non-Cognitive (0), `*_2.edf` = Cognitive (1)

### BrainFlow Pipeline (Phantom Data)

```bash
cd MLA/Phantom
python EEG_MLA_Complete_BrainFlow.py
```

**Input:** CSV/TXT files in `data_brainflow/` directory  
**Labeling:** Files with "cog" = Cognitive (1), others = Non-Cognitive (0)

## Key Features

- **Data Leakage Prevention**: File-level separation ensures epochs from same recording never mix in train/test
- **Comprehensive Features**: Time-domain, frequency-domain, and asymmetry features (~100+ per epoch)
- **Dual Formats**: Both `.py` (production) and `.ipynb` (interactive) versions included
- **Classifiers**: Random Forest and SVM supported
- **Automatic Augmentation**: BrainFlow pipeline includes data augmentation for small datasets

## Evaluation Methods

- **EDF Pipeline**: Leave-One-File-Out Cross-Validation (LOFO CV)
- **BrainFlow Pipeline**: Stratified K-Fold Cross-Validation

## Outputs

Each pipeline generates:
- Confusion matrices (PNG)
- Class distribution plots (PNG)
- Feature importance charts (Random Forest only)
- Console output with accuracy, precision, recall, F1 scores

## Documentation

- [EDF Pipeline Details](Computational/EEG_MLA_Complete_EDF.py)
- [BrainFlow Pipeline Details](Phantom/EEG_MLA_Complete_BrainFlow.py)
- Interactive notebooks: `EEG_MLA_Complete_EDF.ipynb` and `EEG_MLA_Complete_BrainFlow.ipynb`

## Configuration

Edit the `CONFIG` dictionary in each script to customize:
- Classifier type (RF or SVM)
- Number of epochs per file
- Overlap between epochs
- Augmentation settings
- Random state for reproducibility

## Dependencies

See root `requirements.txt` for all dependencies.

## License

MIT License - Copyright © 2024-2025 Lucan Leung