#!/usr/bin/env python3
"""
BrainFlow Channel Merger
Merges multiple OpenBCI/BrainFlow raw data files into unified 8/12/16-channel output

Author: Lucan Leung
Institution: Imperial College London
Project: Closed-Loop Brain Computer Interface for Modulation of Cognitive Functions
"""

import argparse
import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from typing import List, Dict, Tuple


def load_brainflow_raw(filepath: str) -> pd.DataFrame:
    """Load BrainFlow raw CSV/TXT file."""
    df = pd.read_csv(filepath, comment='%', header=0)
    df.columns = df.columns.str.strip()
    if "Timestamp" not in df.columns:
        raise KeyError(f"No 'Timestamp' column in {filepath}. Columns: {df.columns.tolist()}")
    return df


def list_eeg_cols(df: pd.DataFrame) -> List[str]:
    """List all EXG channel columns."""
    return [c for c in df.columns if "EXG Channel" in c]


def parse_selection(txt: str, max_idx: int) -> List[int]:
    """Parse channel selection string like '0,2,5-7' or 'all'."""
    txt = txt.strip().lower()
    if txt in ("", "none", "skip"):
        return []
    if txt in ("all", "a"):
        return list(range(max_idx))
    picked = set()
    for part in txt.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            a, b = int(a), int(b)
            for i in range(min(a, b), max(a, b) + 1):
                if 0 <= i < max_idx:
                    picked.add(i)
        else:
            i = int(part)
            if 0 <= i < max_idx:
                picked.add(i)
    return sorted(picked)


def bandpass_filter(data, low=1.0, high=40.0, fs=250, order=4):
    """Standard EEG bandpass filter (1-40 Hz)."""
    nyq = 0.5 * fs
    low /= nyq
    high /= nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data)


def attenuate_channel(data, factor=0.5):
    """Attenuate dead/noisy channel."""
    return data * factor


def manual_select_channels_with_mapping(filepaths: List[str], target_channels: int):
    """
    Ask user to select channels, map them to output EXG indices,
    and decide if filtering is applied.
    
    Returns:
      dfs: list of raw DataFrames
      mapping: list of dicts {file_idx, orig_col, output_idx, filter_type}
    """
    dfs = [load_brainflow_raw(fp) for fp in filepaths]
    mapping = []
    
    for file_idx, (fp, df) in enumerate(zip(filepaths, dfs)):
        eeg_cols = list_eeg_cols(df)
        if not eeg_cols:
            print(f"\n{fp}\n  No EEG columns found.")
            continue
        
        print(f"\nFile {file_idx}: {fp}")
        print("Available channels (index: name):")
        for i, c in enumerate(eeg_cols):
            s = df[c]
            print(f"  {i:2d}: {c:>14s}  mean={s.mean():8.2f}  std={s.std():8.2f}")
        
        inp = input("Pick channels (e.g., '0,2,5', or 'skip' to skip this file): ").strip().lower()
        if inp in ("skip", ""):
            continue
        
        idxs = parse_selection(inp, max_idx=len(eeg_cols))
        for i in idxs:
            col = eeg_cols[i]
            out_idx = int(input(f"Map {col} (File {file_idx}) → EXG output index (1–{target_channels}, 0=skip): "))
            if out_idx == 0:
                print(f"  Skipping {col}")
                continue
            if not (1 <= out_idx <= target_channels):
                print(f"  Invalid index {out_idx}, skipping channel.")
                continue
            
            filt_inp = input(f"Apply filter to {col}? (b = bandpass, a = attenuate, n = none): ").strip().lower()
            if filt_inp == "b":
                ftype = "bandpass"
            elif filt_inp == "a":
                ftype = "attenuate"
            else:
                ftype = "none"
            
            mapping.append({
                "file_idx": file_idx,
                "orig_col": col,
                "output_idx": out_idx - 1,
                "filter_type": ftype
            })
    
    return dfs, mapping


def merge_with_mapping(dfs: List[pd.DataFrame],
                       mapping: List[dict],
                       max_channels: int,
                       fs: int = 250) -> Tuple[pd.DataFrame, List[str]]:
    """
    Merge multiple DataFrames using user-provided mapping with optional filters.
    Aligns by sample index (assumes same sample rate), not absolute timestamps.
    """
    per_file = []
    for file_idx, df in enumerate(dfs):
        chosen_here = [m["orig_col"] for m in mapping if m["file_idx"] == file_idx]
        if not chosen_here:
            continue
        sub = df[["Timestamp"] + chosen_here].copy()
        sub = sub.rename(columns={c: f"File{file_idx}_{c}" for c in chosen_here})
        per_file.append(sub.reset_index(drop=True))
    
    if not per_file:
        raise ValueError("❌ No dataframes to merge. Check selections.")
    
    min_len = min(len(df) for df in per_file)
    aligned = [df.iloc[:min_len] for df in per_file]
    merged = pd.concat(aligned, axis=1)
    
    common_time = np.arange(min_len) / fs
    
    final_df = pd.DataFrame(index=range(min_len))
    exg_cols = [f"EXG Channel {i}" for i in range(max_channels)]
    for col in exg_cols:
        final_df[col] = 0.0
    
    mapping_descriptions = []
    for m in mapping:
        name = f"File{m['file_idx']}_{m['orig_col']}"
        if name not in merged.columns:
            continue
        series = merged[name].astype(float).copy()
        
        if m["filter_type"] == "bandpass":
            try:
                series = bandpass_filter(series.values, fs=fs)
            except Exception as e:
                print(f"  ⚠️ Could not bandpass {name}: {e}")
        elif m["filter_type"] == "attenuate":
            series = attenuate_channel(series.values)
        
        final_df.iloc[:, m["output_idx"]] = series
        mapping_descriptions.append(f"EXG Channel {m['output_idx']} ← {name}, filter={m['filter_type']}")
    
    final_df.insert(0, "Timestamp", common_time)
    
    return final_df, mapping_descriptions


def chop_after_merge(df: pd.DataFrame, seconds: float, sample_rate_hz: int = 250) -> pd.DataFrame:
    """
    Drop the first `seconds` worth of samples from the merged dataset.
    Keeps alignment across all channels.
    """
    if seconds <= 0:
        return df
    n_samples = int(seconds * sample_rate_hz)
    if n_samples >= len(df):
        raise ValueError(f"Requested to trim {n_samples} samples, but dataset only has {len(df)} samples.")
    return df.iloc[n_samples:].reset_index(drop=True)


def export_as_brainflow(df: pd.DataFrame, output_path: str, sample_rate_hz: int = 250):
    """
    Export DataFrame to BrainFlow TXT format.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with Timestamp and EXG Channel columns
    output_path : str
        Output file path
    sample_rate_hz : int
        Sampling rate in Hz
    """
    with open(output_path, 'w') as f:
        f.write(f"% BrainFlow Export\n")
        f.write(f"% Sample Rate: {sample_rate_hz} Hz\n")
        f.write(f"% Channels: {len([c for c in df.columns if 'EXG' in c])}\n")
        
        header_cols = ['Timestamp'] + [c for c in df.columns if 'EXG' in c]
        f.write('\t'.join(header_cols) + '\n')
        
        for _, row in df.iterrows():
            values = [f"{row[col]:.6f}" for col in header_cols]
            f.write('\t'.join(values) + '\n')


def merge_brainflow_files(data_dir: str = 'data', n_channels: int = 8, 
                          trim_secs: float = 0, preview_duration: float = 20,
                          output_dir: str = None):
    """
    Main function to merge BrainFlow files.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing OpenBCI-RAW-*.txt files
    n_channels : int
        Number of output channels (8, 12, or 16)
    trim_secs : float
        Seconds to trim from start after merging (0 = no trim)
    preview_duration : float
        Duration in seconds for preview plot (0 = skip plot)
    output_dir : str
        Output directory (default: data_dir)
    """
    if output_dir is None:
        output_dir = data_dir
    
    file_list = sorted(glob.glob(os.path.join(data_dir, "OpenBCI-RAW-*.txt")))
    
    if not file_list:
        print(f"No OpenBCI-RAW-*.txt files found in {data_dir}")
        return
    
    print(f"Files found: {len(file_list)}")
    for i, fp in enumerate(file_list):
        print(f"  [{i}] {fp}")
    
    dfs, mapping = manual_select_channels_with_mapping(file_list, target_channels=n_channels)
    
    final_dataset, mapping_desc = merge_with_mapping(dfs, mapping, max_channels=n_channels, fs=250)
    
    if trim_secs > 0:
        print(f"Trimming first {trim_secs} seconds...")
        final_dataset = chop_after_merge(final_dataset, trim_secs, sample_rate_hz=250)
    
    if preview_duration > 0:
        n_samples = int(preview_duration * 250)
        preview = final_dataset.iloc[:n_samples]
        
        plt.figure(figsize=(12, 6))
        exg_cols = [c for c in preview.columns if "EXG" in c]
        for i, col in enumerate(exg_cols):
            plt.plot(preview["Timestamp"], preview[col] + i * 500, label=col)
        plt.title(f"Preview of first {preview_duration} seconds (stacked by channel)")
        plt.xlabel("Timestamp")
        plt.ylabel("EEG (offset vertically)")
        plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
        plt.tight_layout()
        plt.show()
    
    csv_out = os.path.join(output_dir, f"Merged_BrainFlow_{n_channels}ch.csv")
    txt_out = os.path.join(output_dir, f"Merged_BrainFlow_{n_channels}ch.txt")
    
    final_dataset.to_csv(csv_out, index=False)
    export_as_brainflow(final_dataset, txt_out, sample_rate_hz=250)
    
    print(f"\n✅ Exported: {os.path.basename(csv_out)} and {os.path.basename(txt_out)}")
    print("\nChannel mapping:")
    for m in mapping_desc:
        print("  ", m)


def main():
    parser = argparse.ArgumentParser(
        description='Merge multiple OpenBCI/BrainFlow files into unified multi-channel output'
    )
    parser.add_argument(
        '--dir', '-d',
        type=str,
        default='data',
        help='Directory containing OpenBCI-RAW-*.txt files (default: data)'
    )
    parser.add_argument(
        '--channels', '-c',
        type=int,
        choices=[8, 12, 16],
        default=8,
        help='Number of output channels (default: 8)'
    )
    parser.add_argument(
        '--trim', '-t',
        type=float,
        default=0,
        help='Seconds to trim from start after merging (default: 0)'
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Skip preview plot generation'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory (default: same as input directory)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dir):
        print(f"Error: Directory {args.dir} does not exist")
        sys.exit(1)
    
    merge_brainflow_files(
        data_dir=args.dir,
        n_channels=args.channels,
        trim_secs=args.trim,
        preview_duration=0 if args.no_plot else 20,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()