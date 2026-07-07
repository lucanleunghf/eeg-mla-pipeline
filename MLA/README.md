# MLA Pipelines - Machine Learning Analysis

Complete machine learning pipelines for EEG signal classification with proper evaluation methodologies to prevent data leakage.

## Overview

This directory contains two specialized MLA pipelines developed for different experimental setups:

1. **Simulation-EDF Pipeline** (`Simulation-EDF/`) - For open-source EEG database analysis
2. **Phantom-BrainFlow Pipeline** (`Phantom-BrainFlow/`) - For physical phantom head experiments

### Simulation-EDF Pipeline: Open-Source Database Analysis

**Purpose:** Analyzes EEG data from open-source databases (e.g., physionet.org) using computational simulation methods.

**Characteristics:**
- Uses publicly available EDF-format EEG recordings
- No physical data collection required
- Ideal for algorithm development and validation
- Large sample sizes from existing datasets
- Leave-One-File-Out CV for subject-independent evaluation

**Research Context:** This pipeline was designed for computational analysis of existing EEG databases, enabling rapid prototyping and validation without the need for physical data collection infrastructure.

### Phantom-BrainFlow Pipeline: Mock-up Head with In-House cBCI Wearables

**Purpose:** Processes EEG data collected through a gelatin-based mock-up head configured with in-house developed parts and technology, where pseudo-EEG signals are transmitted through the head and captured by the in-house developed wearable cBCI device.

**Characteristics:**
- Gelatin-based mock-up head simulating human scalp/skull properties
- In-house developed parts and technology for signal transmission
- Pseudo-EEG data fed through mock-up head to simulate realistic signal propagation
- Data captured by in-house developed wearable cBCI device via BrainFlow API
- Validates ML pipelines on hardware-derived signals from prototype wearable
- Tests signal integrity through mock-up head medium and wearable acquisition
- Stratified K-Fold CV for balanced evaluation

**Research Context:** This pipeline validates the complete signal chain: pseudo-EEG signals → gelatin mock-up head (simulating human tissue) → in-house wearable cBCI device → BrainFlow acquisition → ML classification. The mock-up head serves as a controlled testbed to validate that our in-house developed wearable can successfully acquire and classify EEG-like signals after they pass through a tissue-simulating medium, demonstrating the viability of the complete hardware-software integration before human trials.

---

Both pipelines address critical flaws in common EEG-ML workflows, particularly data leakage from improper train/test splitting.

## Quick Usage

### Simulation-EDF Pipeline (Open-Source Database)

```bash
cd MLA/Simulation-EDF
python EEG_MLA_Complete_EDF.py
```

**Input:** EDF files in `data/` directory  
**Labeling:** `*_1.edf` = Non-Cognitive (0), `*_2.edf` = Cognitive (1)  
**Source:** Open-source EEG databases (e.g., physionet.org)

### Phantom-BrainFlow Pipeline (Mock-up Head with In-House Wearable)

```bash
cd MLA/Phantom-BrainFlow
python EEG_MLA_Complete_BrainFlow.py
```

**Input:** CSV/TXT files in `data_brainflow/` directory  
**Labeling:** Files with "cog" = Cognitive (1), others = Non-Cognitive (0)  
**Source:** Gelatin-based mock-up head with in-house parts; pseudo-EEG signals captured by in-house wearable cBCI device

## Key Features

- **Data Leakage Prevention**: File-level separation ensures epochs from same recording never mix in train/test
- **Comprehensive Features**: Time-domain, frequency-domain, and asymmetry features (~100+ per epoch)
- **Dual Formats**: Both `.py` (production) and `.ipynb` (interactive) versions included
- **Classifiers**: Random Forest and SVM supported
- **Automatic Augmentation**: BrainFlow pipeline includes data augmentation for small datasets

## Evaluation Methods

- **Simulation-EDF Pipeline**: Leave-One-File-Out Cross-Validation (LOFO CV) - ensures subject independence
- **Phantom-BrainFlow Pipeline**: Stratified K-Fold Cross-Validation - balanced evaluation for controlled experiments

## Outputs

Each pipeline generates:
- Confusion matrices (PNG)
- Class distribution plots (PNG)
- Feature importance charts (Random Forest only)
- Console output with accuracy, precision, recall, F1 scores

## Documentation

- [Simulation-EDF Pipeline Details](Simulation-EDF/EEG_MLA_Complete_EDF.py)
- [Phantom-BrainFlow Pipeline Details](Phantom-BrainFlow/EEG_MLA_Complete_BrainFlow.py)
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