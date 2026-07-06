# EEG Machine Learning Analysis and Conversion Pipeline

Complete research software for EEG data processing, from raw data conversion to machine learning classification. This repository provides an end-to-end workflow for EEG signal analysis developed as part of MRes research at Imperial College London.

---

## Author

**Lucan Leung**  
MRes Researcher, Imperial College London  
Closed-Loop Brain Computer Interface for Modulation of Cognitive Functions Project  
Development Period: September 2024 - October 2025

This software was developed as part of my MRes research. The intellectual property belongs entirely to the author.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{eeg_mla_pipeline,
  author = {Leung, Lucan},
  title = {EEG Machine Learning Analysis and Conversion Pipeline},
  year = {2024-2025},
  url = {https://github.com/lucanleunghf/EEG-MLA-Conversion-Programmes-Pipeline},
  note = {Developed during MRes research at Imperial College London}
}
```

---

## Overview

This repository contains two main components that form a complete EEG analysis workflow:

```
Raw EEG Data → Conversion Tools → Processed Data → MLA Pipelines → Classification Results
```

### 1. **Conversion Programmes** (`Conversion-Programmes/`)

Tools for converting EEG data between different formats (available as both Python scripts and Jupyter Notebooks):
- EDF to ASCII (MC-Stimulus format, 60s window)
- EDF to CSV (middle 60s extraction)
- BrainFlow Channel Merger (8/12/16 channel output)

[→ See Conversion Programmes README for detailed usage](Conversion-Programmes/README.md)

### 2. **MLA Pipelines** (`MLA/`)

Complete machine learning pipelines for EEG classification:
- **EDF Pipeline** - Real EEG data with Leave-One-File-Out CV
- **BrainFlow Pipeline** - Phantom/Emotiv data with augmentation

[→ See MLA Pipelines README for detailed usage](MLA/README.md)

---

## Quick Start

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/lucanleunghf/EEG-MLA-Conversion-Programmes-Pipeline.git
cd EEG-MLA-Conversion-Programmes-Pipeline
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify installation**:
```bash
python -c "import numpy, pandas, scipy, sklearn, mne, brainflow; print('All dependencies installed!')"
```

### Workflow

**Step 1: Convert your data** (if needed)
- Navigate to `Conversion-Programmes/`
- Use the appropriate converter for your data format
- See [Conversion Programmes README](Conversion-Programmes/README.md)

**Step 2: Run MLA analysis**
- Navigate to `MLA/Computational/` (for EDF data) or `MLA/Phantom/` (for BrainFlow data)
- Run the pipeline script or notebook
- See [MLA Pipelines README](MLA/README.md)

---

## Directory Structure

```
EEG-MLA-Conversion-Programmes-Pipeline/
├── README.md                    # This file - main project overview
├── LICENSE                      # MIT License
├── requirements.txt             # All Python dependencies
├── .gitignore                   # Git ignore rules
│
├── Conversion-Programmes/     # Data conversion tools (publishable)
│   ├── README.md               # Conversion tools documentation
│   ├── 01_EDF_to_ASCII_Converter.py      # Python script
│   ├── 01_EDF_to_ASCII_Converter.ipynb   # Jupyter Notebook
│   ├── 02_EDF_to_CSV_Converter.py        # Python script
│   ├── 02_EDF_to_CSV_Converter.ipynb     # Jupyter Notebook
│   ├── 03_BrainFlow_Channel_Merger.py    # Python script
│   └── 03_BrainFlow_Channel_Merger.ipynb # Jupyter Notebook
│
├── MLA/                         # Machine Learning Analysis pipelines
│   ├── README.md               # MLA pipelines documentation
│   ├── Computational/          # EDF pipeline for real EEG data
│   │   ├── EEG_MLA_Complete_EDF.py
│   │   ├── EEG_MLA_Complete_EDF.ipynb
│   │   ├── data/               # Place EDF files here (not committed)
│   │   └── outputs/            # Generated visualizations (not committed)
│   │
│   └── Phantom/                # BrainFlow pipeline for phantom data
│       ├── EEG_MLA_Complete_BrainFlow.py
│       ├── EEG_MLA_Complete_BrainFlow.ipynb
│       ├── data_brainflow/     # Place BrainFlow files here (not committed)
│       └── outputs/            # Generated visualizations (not committed)
│
└── _archive/                   # Original notebooks and working files (not committed)
```

**Note:** The `_archive/` directory contains original Jupyter notebooks and development files that are not intended for publishing. These are kept for reference and version history.

---

## Dependencies

All required packages are listed in `requirements.txt`:

```
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
mne>=1.0.0
brainflow>=5.0.0
pillow>=8.0.0
```

Install with:
```bash
pip install -r requirements.txt
```

### Optional: GUI Support

For file dialog support in conversion tools:

```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# macOS
brew install python-tk
```

---

## Usage Summary

### Conversion Tools

| Tool | File | Input | Output | Purpose |
|------|------|-------|--------|---------|
| EDF to ASCII | `01_EDF_to_ASCII_Converter.py` | EDF files | ASCII .dat | MC-Stimulus format (60s window) |
| EDF to CSV | `02_EDF_to_CSV_Converter.py` | EDF file | CSV | Middle 60s extraction |
| BrainFlow Merger | `03_BrainFlow_Channel_Merger.py` | Multiple OpenBCI files | CSV/TXT | Channel merging (8/12/16 ch) |

**Note:** Both Python scripts (`.py`) and Jupyter Notebooks (`.ipynb`) are provided. Use scripts for command-line automation or notebooks for interactive exploration.

[→ Detailed conversion tool documentation](Conversion-Programmes/README.md)

### MLA Pipelines

| Pipeline | Data Type | Features | Evaluation |
|----------|-----------|----------|------------|
| EDF Pipeline | Real EEG (EDF) | Time, frequency, asymmetry | Leave-One-File-Out CV |
| BrainFlow Pipeline | Phantom/Emotiv | Time, frequency, asymmetry | Stratified K-Fold CV |

[→ Detailed MLA pipeline documentation](MLA/README.md)

---

## Key Features

### Data Leakage Prevention
- File-level train/test separation
- Leave-One-File-Out cross-validation
- Proper epoch handling

### Comprehensive Feature Extraction
- Time-domain: mean, std, skewness, kurtosis
- Frequency-domain: bandpower (delta, theta, alpha, beta, gamma)
- Asymmetry: left-right hemisphere differences

### Dual Format Support
- Production-ready `.py` scripts
- Interactive `.ipynb` notebooks for exploration

### Classifiers
- Random Forest (default)
- Support Vector Machine (SVM)

---

## Contributing

Contributions are welcome. Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## Acknowledgments

- [MNE-Python](https://mne.tools/) for EEG data processing
- [BrainFlow](https://brainflow.org/) for EEG data acquisition
- [scikit-learn](https://scikit-learn.org/) for machine learning algorithms
- Multi Channel Systems for MC-Stimulus software
- Imperial College London for research support

---

## Support

For issues, questions, or contributions, please open an issue on GitHub.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Copyright**: © 2024-2025 Lucan Leung. All rights reserved.

You are free to use, modify, and distribute this software as long as proper attribution is given to the author.
