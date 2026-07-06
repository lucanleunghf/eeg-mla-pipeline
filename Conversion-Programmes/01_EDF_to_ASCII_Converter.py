#!/usr/bin/env python3
"""
EDF to ASCII Converter
Converts EDF EEG files to MC-Stimulus ASCII format (60-second window: 90-150s)

Author: Lucan Leung
Institution: Imperial College London
Project: Closed-Loop Brain Computer Interface for Modulation of Cognitive Functions
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import mne
from tkinter import filedialog, Tk


def select_directory():
    """Open GUI dialog to select directory containing EDF files."""
    root = Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    folder_path = filedialog.askdirectory()
    root.destroy()
    return folder_path


def get_phantom_channel_mapping(channel_names):
    """Interactively map EEG channels to 8 phantom stimulus channels."""
    phantom_chs = [
        'R Frontal arrow up',
        'R Frontal arrow down',
        'R Medial',
        'R Posterial',
        'L Frontal arrow up',
        'L Frontal arrow down',
        'L Medial',
        'L Posterial'
    ]
    
    chosen_channels = []
    print(f"\nShowing channel list {channel_names}")
    print("Choose EEG channel for phantom channel by printing their order number (1-based).")
    print("Press Enter for empty phantom channel\n")
    
    for ch_name in phantom_chs:
        while True:
            choice = input(f"{ch_name}: ").strip()
            if choice == '':
                chosen_channels.append(None)
                break
            try:
                idx = int(choice)
                if 1 <= idx <= len(channel_names):
                    chosen_channels.append(idx)
                    break
                else:
                    print(f"Please enter a number between 1 and {len(channel_names)}, or press Enter for empty")
            except ValueError:
                print("Invalid input. Please enter a number or press Enter for empty")
    
    return chosen_channels


def convert_edf_to_ascii(folder_path, multiplication_factor=1):
    """
    Convert all EDF files in folder to MC-Stimulus ASCII format.
    
    Parameters:
    -----------
    folder_path : str
        Path to directory containing EDF files
    multiplication_factor : float
        Factor to multiply signal amplitude (default: 1)
    """
    files = [f for f in os.listdir(folder_path) if f.endswith('.edf')]
    
    if not files:
        print(f"No EDF files found in {folder_path}")
        return
    
    print(f"Found {len(files)} EDF file(s)")
    
    for j, filename in enumerate(files):
        filepath = os.path.join(folder_path, filename)
        print(f"\nProcessing {j+1}/{len(files)}: {filename}")
        
        raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
        fs = int(raw.info['sfreq'])
        datas, times = raw[:, :]
        channel_names = raw.info["ch_names"]
        
        start = 90
        end = 150
        time_mask = (times >= start) & (times <= end)
        datas = datas[:, time_mask]
        times = times[time_mask]
        times = times * 1_000_000
        datas = datas * 1_000_000
        
        if j == 0:
            chosen_channels = get_phantom_channel_mapping(channel_names)
        
        base_filename = f'EEGx{multiplication_factor}.dat'
        base, ext = os.path.splitext(base_filename)
        counter = 1
        new_filename = base_filename
        while os.path.exists(os.path.join(folder_path, new_filename)):
            new_filename = f"{base}_{counter}{ext}"
            counter += 1
        
        output_path = os.path.join(folder_path, new_filename)
        with open(output_path, 'wb+') as f:
            head = 'Multi Channel Systems MC_Stimulus \nASCII import Version 1.10 \nchannels: 8 \noutput mode: voltage \nformat: \t4 \n '
            f.write(head.encode('ascii'))
            
            eeg_data_ascii = pd.DataFrame([], columns=['1', 'Time'])
            for i in range(len(chosen_channels)):
                if chosen_channels[i] is None:
                    eeg_data_ascii['1'] = datas[0][:-1] * 0
                    eeg_data_ascii['Time'] = np.diff(times)
                    eeg_data_ascii = eeg_data_ascii[['1', 'Time']]
                    channel_number = f'\nchannel: {i+1}\nvalue\ttime \n'
                    f.write(channel_number.encode('ascii'))
                    eeg_data_ascii.to_csv(f, sep='\t', index=False, header=False, lineterminator='\n')
                else:
                    eeg_data_ascii['1'] = datas[int(chosen_channels[i]) - 1][:-1] * multiplication_factor
                    eeg_data_ascii['Time'] = np.diff(times)
                    eeg_data_ascii = eeg_data_ascii[['1', 'Time']]
                    channel_number = f'\nchannel: {i+1}\nvalue\ttime \n'
                    f.write(channel_number.encode('ascii'))
                    eeg_data_ascii.to_csv(f, sep='\t', index=False, header=False, lineterminator='\n')
        
        print(f"Saved: {new_filename}")
    
    print(f"\nConversion complete! {len(files)} file(s) processed.")


def main():
    parser = argparse.ArgumentParser(
        description='Convert EDF EEG files to MC-Stimulus ASCII format'
    )
    parser.add_argument(
        '--dir', '-d',
        type=str,
        default=None,
        help='Directory containing EDF files (default: interactive selection)'
    )
    parser.add_argument(
        '--factor', '-f',
        type=float,
        default=1,
        help='Multiplication factor for signal amplitude (default: 1)'
    )
    
    args = parser.parse_args()
    
    if args.dir:
        folder_path = args.dir
        if not os.path.exists(folder_path):
            print(f"Error: Directory {folder_path} does not exist")
            sys.exit(1)
    else:
        print("Select directory containing EDF files...")
        folder_path = select_directory()
        if not folder_path:
            print("No directory selected. Exiting.")
            sys.exit(0)
    
    convert_edf_to_ascii(folder_path, args.factor)


if __name__ == '__main__':
    main()