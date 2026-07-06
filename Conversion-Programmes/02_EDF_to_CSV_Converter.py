#!/usr/bin/env python3
"""
EDF to CSV Converter
Converts EDF files to CSV format, extracting the middle 60 seconds

Author: Lucan Leung
Institution: Imperial College London
Project: Closed-Loop Brain Computer Interface for Modulation of Cognitive Functions
"""

import argparse
import os
import sys
import numpy as np
import mne
from tkinter import filedialog, Tk


def select_file():
    """Open GUI dialog to select EDF file."""
    root = Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title='Select EDF file',
        filetypes=[('EDF files', '*.edf'), ('All files', '*.*')]
    )
    root.destroy()
    return file_path


def convert_edf_to_csv(edf_path, output_dir=None):
    """
    Convert EDF file to CSV format (middle 60 seconds).
    
    Parameters:
    -----------
    edf_path : str
        Path to EDF file
    output_dir : str, optional
        Output directory (default: same as input file)
    """
    if not os.path.exists(edf_path):
        print(f"Error: File {edf_path} does not exist")
        return
    
    print(f"Processing: {os.path.basename(edf_path)}")
    
    edf = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    
    duration = edf.times[-1]
    midpoint = duration / 2.0
    tmin = max(0, midpoint - 30)
    tmax = min(duration, midpoint + 30)
    
    print(f"Total duration: {duration:.2f}s")
    print(f"Extracting middle 60s: {tmin:.2f}s to {tmax:.2f}s")
    
    edf.crop(tmin=tmin, tmax=tmax)
    
    data = edf.get_data().T
    times = edf.times[:data.shape[0]].reshape(-1, 1)
    data_with_time = np.hstack((times, data))
    
    header = 'Time,' + ','.join(edf.ch_names)
    
    if output_dir is None:
        output_dir = os.path.dirname(edf_path)
    
    base_name = os.path.splitext(os.path.basename(edf_path))[0]
    output_filename = f"{base_name}_middle60s.csv"
    output_path = os.path.join(output_dir, output_filename)
    
    np.savetxt(output_path, data_with_time, delimiter=',', header=header, comments='')
    
    print(f"Saved: {output_filename}")
    print(f"Shape: {data.shape[0]} samples x {data.shape[1] + 1} columns (Time + {data.shape[1]} channels)")
    print("Conversion complete!")


def main():
    parser = argparse.ArgumentParser(
        description='Convert EDF file to CSV format (middle 60 seconds)'
    )
    parser.add_argument(
        '--file', '-f',
        type=str,
        default=None,
        help='Path to EDF file (default: interactive selection)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory (default: same as input file)'
    )
    
    args = parser.parse_args()
    
    if args.file:
        edf_path = args.file
        if not os.path.exists(edf_path):
            print(f"Error: File {edf_path} does not exist")
            sys.exit(1)
    else:
        print("Select EDF file to convert...")
        edf_path = select_file()
        if not edf_path:
            print("No file selected. Exiting.")
            sys.exit(0)
    
    convert_edf_to_csv(edf_path, args.output)


if __name__ == '__main__':
    main()