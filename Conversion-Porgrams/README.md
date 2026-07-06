# EEG Data Conversion Programmes

A collection of Jupyter Notebook tools for converting EEG data between different formats. These programmes were developed as part of the Closed-Loop Brain Computer Interface for Modulation of Cognitive Functions project at Imperial College London.

---

## Author

**Luca Can Leung**  
MRes Researcher, Imperial College London  
Closed-Loop Brain Computer Interface for Modulation of Cognitive Functions Project  
Development Period: September 2024 - October 2025

These conversion tools were developed as part of my MRes research to facilitate EEG data processing and format conversion. The intellectual property belongs entirely to the author.

## Citation

If you use these conversion tools in your research, please cite:

```bibtex
@software{eeg_conversion_tools,
  author = {Leung, Luca Can},
  title = {EEG Data Conversion Programmes},
  year = {2024-2025},
  url = {https://github.com/lucanleunghf/EEG-MLA-Conversion-Programmes-Pipeline/tree/main/Conversion-Porgrams},
  note = {Developed during MRes research at Imperial College London}
}
```

---

## Overview

This repository contains four Jupyter Notebook tools for EEG data conversion:

1. **EDF to ASCII Converter (Setting 1)** - Converts EDF files to MC-Stimulus ASCII format (60-second window)
2. **EDF to ASCII Converter (Setting 2)** - Converts EDF files to MC-Stimulus ASCII format (30-second window)
3. **EDF to CSV Converter** - Extracts middle 60 seconds from EDF files to CSV format
4. **BrainFlow Merger** - Merges multiple OpenBCI/BrainFlow files into unified 8/12/16-channel output

---

## Tools

### 1. EDF to ASCII Converter

**File:** `01_EDF_to_ASCII_Converter.ipynb`

**Purpose:** Converts EDF EEG files to ASCII format compatible with Multi Channel Systems MC-Stimulus software, extracting a 60-second time window (90-150 seconds from recording start).

**Features:**
- Interactive channel mapping to 8 phantom stimulus channels
- Automatic file numbering to prevent overwriting
- Time window extraction (90-150 seconds)
- Voltage conversion (Volts to microVolts)
- MC-Stimulus compliant header format

**Input:** `.edf` files (European Data Format)

**Output:** `.dat` files (ASCII format for MC-Stimulus)

**Output Format:**
```
Multi Channel Systems MC_Stimulus 
ASCII import Version 1.10 
channels: 8 
output mode: voltage 
format: 4
```

**Usage:**
1. Open the notebook in Jupyter
2. Run all cells
3. Select directory containing EDF files via GUI dialog
4. For each of the 8 phantom channels, enter the EEG channel number (1-based index) to map
5. Press Enter to leave a phantom channel empty
6. Output files saved as `EEGx1.dat`, `EEGx1_1.dat`, etc.

**Phantom Channel Mapping:**
- R Frontal arrow up/down
- R Medial
- R Posterial
- L Frontal arrow up/down
- L Medial
- L Posterial

---

### 2. EDF to ASCII Converter (30s Window) [Archive]

**File:** `archive/01b_EDF_to_ASCII_Converter_30s.ipynb`

**Purpose:** Alternative version that extracts a 30-second time window (30-60 seconds from recording start).

**Status:** Archived - not included in main distribution. Available in `archive/` folder for reference.

---

### 3. EDF to CSV Converter

**File:** `02_EDF_to_CSV_Converter.ipynb`

**Purpose:** Converts EDF files to CSV format, extracting the middle 60 seconds of the recording.

**Features:**
- Automated conversion (no user interaction)
- Extracts middle 60-second window
- Preserves all EEG channels
- Time-stamped output

**Input:** Single `.edf` file (filename must be modified in code)

**Output:** `.csv` file with columns: `Time,EEG Fp1,EEG Fp2,...`

**Usage:**
1. Edit the filename in the notebook (currently hardcoded as `Subject00_1.edf`)
2. Run all cells
3. Output saved as `<filename>_middle60s.csv`

**Example Output:**
```csv
Time,EEG Fp1,EEG Fp2,EEG F3,...
0.0,1.23,4.56,7.89,...
0.0039,1.25,4.58,7.91,...
```

---

### 4. BrainFlow Channel Merger

**File:** `03_BrainFlow_Channel_Merger.ipynb`

**Purpose:** Merges multiple OpenBCI/BrainFlow raw data files into a single unified file with configurable channel count (8, 12, or 16 channels).

**Features:**
- Configurable output channel count (8, 12, or 16)
- Interactive channel selection and mapping
- Per-channel filtering options:
  - Bandpass filter (1-40 Hz)
  - Attenuation (50% reduction)
  - No filter
- Automatic file synchronization (aligns by sample index)
- Preview plot generation
- Dual output format (CSV and BrainFlow TXT)

**Input:** Multiple `OpenBCI-RAW-*.txt` files

**Output:**
- `Merged_BrainFlow_{n}ch.csv`
- `Merged_BrainFlow_{n}ch.txt`

**Usage:**
1. Place OpenBCI raw files in `data/` directory (or modify path in notebook)
2. Run all cells
3. Enter desired output channel count (8, 12, or 16)
4. For each file:
   - Select channels: use format `0,2,5-7` or `all` or `skip`
   - Map each selected channel to output position (1-n)
   - Choose filter: `b` (bandpass), `a` (attenuate), `n` (none)
5. Enter seconds to trim from start (0 = no trim)
6. Review preview plot
7. Files automatically saved

**Example Interaction:**
```
Enter number of output channels (8, 12, or 16): 8

File 1 - OpenBCI-RAW-001.txt
Available channels: EXG Channel 0, EXG Channel 1, ..., EXG Channel 7
Select channels (e.g., '0,2,5-7' or 'all' or 'skip'): 0,1,2,3
Channel 0 -> Map to position: 1
Apply filter to channel 0? (b)andpass, (a)ttenuate, (n)one: n
Channel 1 -> Map to position: 2
Apply filter to channel 1? (b)andpass, (a)ttenuate, (n)one: n
...
```

**Known Issue:** The `export_as_brainflow()` function is referenced but not defined in the notebook. You may need to implement this function or modify the export section.

---

## Installation

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab

### Dependencies

Install required packages:

```bash
pip install numpy pandas scipy matplotlib mne brainflow pillow
```

Or create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install numpy pandas scipy matplotlib mne brainflow pillow
```

### Optional: GUI Support

The EDF to ASCII converters use `tkinter` for file selection dialogs. On most systems, this is included with Python. If not available:

```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# macOS (with Homebrew)
brew install python-tk

# Windows
# Usually included with Python installation
```

---

## Directory Structure

```
Conversion-Porgrams/
├── 01_EDF_to_ASCII_Converter.ipynb
├── 02_EDF_to_CSV_Converter.ipynb
├── 03_BrainFlow_Channel_Merger.ipynb
├── README.md                    # This file
├── data/                        # Place input files here (BrainFlow Merger)
└── archive/                     # Archived tools (not for distribution)
    └── 01b_EDF_to_ASCII_Converter_30s.ipynb
```

---

## File Format Specifications

### EDF (European Data Format)
- Standard format for biomedical signal recording
- Supported by MNE-Python library
- Contains header with sampling rate, channel names, and metadata

### ASCII (MC-Stimulus Format)
- Tab-separated values with time-value pairs
- Specific header required by Multi Channel Systems MC-Stimulus
- 8 channels maximum
- Time in seconds, voltage in microVolts

### CSV (Comma-Separated Values)
- Standard CSV with time column
- All EEG channels preserved
- Time in seconds, data in original units

### BrainFlow TXT
- OpenBCI raw data format
- Contains timestamp and EXG channel columns
- Tab or comma-separated

---

## Configuration

### EDF to ASCII Converters

Time windows are hardcoded:
- **Setting 1:** 90-150 seconds (60 seconds of data)
- **Setting 2:** 30-60 seconds (30 seconds of data) [Archived]

To modify, edit the `start_time` and `end_time` variables in the respective notebooks.

### EDF to CSV Converter

- Extracts middle 60 seconds automatically
- Filename is hardcoded - must be edited before each use

### BrainFlow Merger

- Output channel count: User-specified (8, 12, or 16)
- Bandpass filter: 1-40 Hz (4th order Butterworth)
- Attenuation: 50% signal reduction
- Trimming: User-specified seconds to remove from start

---

## Troubleshooting

### Common Issues

**1. "No module named 'mne'"**
```bash
pip install mne
```

**2. "tkinter not found"**
Install python-tk for your system (see Installation section)

**3. "File not found" (BrainFlow Merger)**
Ensure OpenBCI raw files are in the `data/` directory or update the path in the notebook

**4. "export_as_brainflow() is not defined"**
This is a known issue in the BrainFlow Merger. The function is referenced but not implemented. You can:
- Comment out the export line
- Implement the function based on your BrainFlow format requirements
- Use the CSV output instead

**5. "Index out of range" during channel mapping**
Ensure you are using 1-based indexing for EDF channels (as displayed in the GUI)

**6. EDF file fails to load**
Verify the EDF file is not corrupted and is compatible with MNE-Python

---

## Archive

The `archive/` directory contains additional conversion tools that are not included in the main distribution but are kept for reference:

- `01b_EDF_to_ASCII_Converter_30s.ipynb` - Alternative 30-second window converter
- Original development notebooks and working files

These files are not tracked by git and are available for local reference only.

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

- [MNE-Python](https://mne.tools/) for EDF file handling
- [BrainFlow](https://brainflow.org/) for OpenBCI data format
- [Multi Channel Systems](https://www.multichannelsystems.com/) for MC-Stimulus software
- Imperial College London for research support

---

## Support

For issues, questions, or contributions, please open an issue on GitHub.

---

## License

This project is licensed under the MIT License - see the root [LICENSE](../LICENSE) file for details.

**Copyright**: © 2024-2025 Luca Can Leung. All rights reserved.

You are free to use, modify, and distribute this software as long as proper attribution is given to the author.