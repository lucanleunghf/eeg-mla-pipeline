#!/usr/bin/env python3
"""
EEG Machine Learning Analysis Pipeline - Complete EDF Version
Classifies cognitive vs non-cognitive states from EDF EEG recordings.

Features:
- File-level train/test split (no data leakage)
- Both epoch-level and file-level evaluation
- Random Forest and SVM classifiers
- Comprehensive feature extraction (time, frequency, asymmetry)
- Proper cross-validation and confusion matrices
"""

import os
import glob
import numpy as np
import mne
from scipy.stats import skew, kurtosis
from mne.time_frequency import psd_array_welch
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================
DATA_DIR = "data"  # Directory containing EDF files
EPOCH_LENGTH_S = 2  # Epoch length in seconds
MIDDLE_SECONDS = 60  # Extract from middle of recording
RANDOM_STATE = 42

# EEG frequency bands
BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45)
}

# Frontal asymmetry pairs
ASYM_PAIRS = [("Fp1", "Fp2"), ("F3", "F4"), ("F7", "F8")]


# ============================================
# FEATURE EXTRACTION
# ============================================
def extract_epoch_features(epoch_data, sfreq, channel_names):
    """
    Extract comprehensive features from one epoch of EEG data.
    
    Parameters:
    -----------
    epoch_data : ndarray (n_channels, n_samples)
    sfreq : float - sampling frequency
    channel_names : list of str - channel names
    
    Returns:
    --------
    features : ndarray - concatenated feature vector
    """
    features = []
    
    for ch_idx, sig in enumerate(epoch_data):
        # Time-domain features
        features.append(np.mean(sig))
        features.append(np.std(sig))
        features.append(skew(sig))
        features.append(kurtosis(sig))
        
        # Frequency-domain: PSD and band powers
        psd, freqs = psd_array_welch(
            sig, sfreq=sfreq, fmin=1, fmax=45,
            n_fft=min(512, len(sig)), n_overlap=min(256, len(sig)//2),
            verbose=False
        )
        psd = np.squeeze(psd) if psd.ndim > 1 else psd
        
        # Total power
        total_power = np.sum(psd)
        
        # Absolute band powers
        for band_name, (fmin, fmax) in BANDS.items():
            mask = (freqs >= fmin) & (freqs < fmax)
            bp = np.sum(psd[mask]) if np.any(mask) else 0
            features.append(bp)
        
        # Relative band powers
        for band_name, (fmin, fmax) in BANDS.items():
            mask = (freqs >= fmin) & (freqs < fmax)
            bp = np.sum(psd[mask]) if np.any(mask) else 0
            rel_bp = bp / total_power if total_power > 0 else 0
            features.append(rel_bp)
        
        # Beta/Alpha ratio
        alpha_mask = (freqs >= BANDS["alpha"][0]) & (freqs < BANDS["alpha"][1])
        beta_mask = (freqs >= BANDS["beta"][0]) & (freqs < BANDS["beta"][1])
        alpha_power = np.sum(psd[alpha_mask]) if np.any(alpha_mask) else 0
        beta_power = np.sum(psd[beta_mask]) if np.any(beta_mask) else 0
        features.append(beta_power / alpha_power if alpha_power > 0 else 0)
    
    # Frontal asymmetry features (alpha band)
    for left_ch, right_ch in ASYM_PAIRS:
        if left_ch in channel_names and right_ch in channel_names:
            l_idx = channel_names.index(left_ch)
            r_idx = channel_names.index(right_ch)
            
            # Compute alpha power for left channel
            psd_L, freqs = psd_array_welch(
                epoch_data[l_idx], sfreq=sfreq,
                fmin=BANDS["alpha"][0], fmax=BANDS["alpha"][1],
                n_fft=min(512, len(epoch_data[l_idx])),
                n_overlap=min(256, len(epoch_data[l_idx])//2),
                verbose=False
            )
            alpha_L = np.sum(np.squeeze(psd_L) if psd_L.ndim > 1 else psd_L)
            
            # Compute alpha power for right channel
            psd_R, _ = psd_array_welch(
                epoch_data[r_idx], sfreq=sfreq,
                fmin=BANDS["alpha"][0], fmax=BANDS["alpha"][1],
                n_fft=min(512, len(epoch_data[r_idx])),
                n_overlap=min(256, len(epoch_data[r_idx])//2),
                verbose=False
            )
            alpha_R = np.sum(np.squeeze(psd_R) if psd_R.ndim > 1 else psd_R)
            
            # Asymmetry index
            denom = alpha_L + alpha_R
            asym = (alpha_R - alpha_L) / denom if denom > 0 else 0
            features.append(asym)
        else:
            features.append(0)  # Missing channel pair
    
    return np.array(features)


def build_epochs_from_file(fname, channels, epoch_length_s=EPOCH_LENGTH_S, middle_seconds=MIDDLE_SECONDS):
    """
    Build epochs from an EDF file and extract features.
    
    Parameters:
    -----------
    fname : str - path to EDF file
    channels : list of str - channel names to use
    epoch_length_s : float - epoch length in seconds
    middle_seconds : float - seconds to extract from middle
    
    Returns:
    --------
    features : list of ndarray - feature vector for each epoch
    """
    raw = mne.io.read_raw_edf(fname, preload=True, verbose=False)
    raw.pick(channels)
    sfreq = raw.info["sfreq"]
    
    # Extract middle portion
    n_samples = int(middle_seconds * sfreq)
    start_sample = int((raw.n_times - n_samples) // 2)
    data, _ = raw[:, start_sample:start_sample+n_samples]
    
    # Create non-overlapping epochs
    n_epoch_samples = int(epoch_length_s * sfreq)
    features = []
    
    for start_idx in range(0, data.shape[1] - n_epoch_samples, n_epoch_samples):
        e_data = data[:, start_idx:start_idx+n_epoch_samples]
        if e_data.shape[1] == n_epoch_samples:
            feats = extract_epoch_features(e_data, sfreq, channels)
            features.append(feats)
    
    return features


# ============================================
# DATASET BUILDING
# ============================================
def load_edf_dataset(data_dir=DATA_DIR):
    """
    Load EDF files and assign labels based on filename convention.
    _1.edf = Non-Cognitive (label 0)
    _2.edf = Cognitive (label 1)
    
    Returns:
    --------
    files : list of str - paths to EDF files
    labels : list of int - corresponding labels
    common_channels : list of str - channels present in all files
    """
    print(f"Looking for EDF files in {data_dir} ...")
    edf_files = glob.glob(os.path.join(data_dir, "*.edf"))
    print(f"Found {len(edf_files)} EDF files.")
    
    files, labels = [], []
    for f in edf_files:
        if "_1" in os.path.basename(f):
            files.append(f)
            labels.append(0)  # Non-Cognitive
        elif "_2" in os.path.basename(f):
            files.append(f)
            labels.append(1)  # Cognitive
    
    print(f"Labeled files: {len(files)} (Non-Cognitive: {labels.count(0)}, Cognitive: {labels.count(1)})")
    
    # Find common channels across all files
    common_channels = None
    for f in files[:min(5, len(files))]:
        raw = mne.io.read_raw_edf(f, preload=False, verbose=False)
        chs = set(raw.ch_names)
        if common_channels is None:
            common_channels = chs
        else:
            common_channels = common_channels.intersection(chs)
    
    common_channels = sorted(list(common_channels))
    print(f"Using {len(common_channels)} common channels.")
    
    return files, labels, common_channels


def build_file_level_features(files, labels, channels):
    """
    Build file-level feature vectors by averaging epoch features.
    
    Returns:
    --------
    X : ndarray (n_files, n_features)
    y : ndarray (n_files,)
    """
    X, y = [], []
    for f, lab in zip(files, labels):
        try:
            epoch_feats = build_epochs_from_file(f, channels)
            if len(epoch_feats) > 0:
                file_feat = np.mean(epoch_feats, axis=0)
                X.append(file_feat)
                y.append(lab)
        except Exception as e:
            print(f"Warning: Could not process {f}: {e}")
    
    return np.array(X), np.array(y)


def build_epoch_level_dataset(files, labels, channels):
    """
    Build epoch-level dataset (all epochs from all files).
    
    Returns:
    --------
    X : ndarray (n_epochs, n_features)
    y : ndarray (n_epochs,)
    file_ids : list of int - file index for each epoch (for file-level CV)
    """
    X, y, file_ids = [], [], []
    
    for file_idx, (f, lab) in enumerate(zip(files, labels)):
        try:
            epoch_feats = build_epochs_from_file(f, channels)
            for feats in epoch_feats:
                X.append(feats)
                y.append(lab)
                file_ids.append(file_idx)
        except Exception as e:
            print(f"Warning: Could not process {f}: {e}")
    
    return np.array(X), np.array(y), file_ids


# ============================================
# EVALUATION FUNCTIONS
# ============================================
def evaluate_epoch_level(X, y, file_ids):
    """
    Evaluate at epoch level with proper file-level separation.
    Uses Leave-One-File-Out CV to prevent data leakage.
    """
    print("\n" + "="*60)
    print("EPOCH-LEVEL EVALUATION (Leave-One-File-Out CV)")
    print("="*60)
    
    unique_files = len(set(file_ids))
    print(f"Total epochs: {len(X)}, from {unique_files} files")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Leave-One-File-Out CV
    loo = LeaveOneOut()
    rf_preds = np.zeros(len(X))
    svm_preds = np.zeros(len(X))
    
    file_indices = {fid: np.where(np.array(file_ids) == fid)[0] for fid in set(file_ids)}
    
    print("\nRunning Leave-One-File-Out CV...")
    for train_idx, test_idx in loo.split(X):
        # Get file IDs for test set
        test_file_ids = set([file_ids[i] for i in test_idx])
        
        # Ensure train set doesn't contain any epochs from test files
        clean_train_idx = [i for i in train_idx if file_ids[i] not in test_file_ids]
        
        if len(clean_train_idx) == 0 or len(test_idx) == 0:
            continue
        
        X_train, X_test = X_scaled[clean_train_idx], X_scaled[test_idx]
        y_train, y_test = y[clean_train_idx], y[test_idx]
        
        # Random Forest
        rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_preds[test_idx] = rf.predict(X_test)
        
        # SVM
        svm = SVC(kernel="rbf", C=1, gamma="scale", random_state=RANDOM_STATE)
        svm.fit(X_train, y_train)
        svm_preds[test_idx] = svm.predict(X_test)
    
    # Calculate accuracies
    rf_acc = accuracy_score(y, rf_preds)
    svm_acc = accuracy_score(y, svm_preds)
    
    print(f"\nRandom Forest - LOFO CV Accuracy: {rf_acc*100:.2f}%")
    print(f"SVM - LOFO CV Accuracy: {svm_acc*100:.2f}%")
    
    # Confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    cm_rf = confusion_matrix(y, rf_preds)
    ConfusionMatrixDisplay(cm_rf, display_labels=["Non-Cognitive", "Cognitive"]).plot(
        ax=axes[0], cmap="Blues", colorbar=True
    )
    axes[0].set_title("Random Forest (Epoch-Level)")
    axes[0].set_ylabel("")
    
    cm_svm = confusion_matrix(y, svm_preds)
    ConfusionMatrixDisplay(cm_svm, display_labels=["Non-Cognitive", "Cognitive"]).plot(
        ax=axes[1], cmap="Greens", colorbar=True
    )
    axes[1].set_title("SVM (Epoch-Level)")
    axes[1].set_ylabel("")
    
    plt.tight_layout()
    plt.savefig("epoch_level_confusion_matrices.png", dpi=300)
    plt.show()
    print("Saved: epoch_level_confusion_matrices.png")
    
    return {
        "rf_acc": rf_acc,
        "svm_acc": svm_acc,
        "rf_cm": cm_rf,
        "svm_cm": cm_svm
    }


def evaluate_file_level(X, y):
    """
    Evaluate at file level (one feature vector per file).
    Uses 80/20 train-test split with stratification.
    """
    print("\n" + "="*60)
    print("FILE-LEVEL EVALUATION (80/20 Split)")
    print("="*60)
    print(f"Total files: {len(X)}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)
    rf_acc = accuracy_score(y_test, y_pred_rf)
    
    # Cross-validation
    rf_cv = cross_val_score(rf, X_train_scaled, y_train, cv=5).mean()
    
    # SVM
    svm = SVC(kernel="rbf", C=1, gamma="scale", random_state=RANDOM_STATE)
    svm.fit(X_train_scaled, y_train)
    y_pred_svm = svm.predict(X_test_scaled)
    svm_acc = accuracy_score(y_test, y_pred_svm)
    
    # Cross-validation
    svm_cv = cross_val_score(svm, X_train_scaled, y_train, cv=5).mean()
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Random Forest - Test Accuracy: {rf_acc*100:.2f}%, CV: {rf_cv*100:.2f}%")
    print(f"SVM - Test Accuracy: {svm_acc*100:.2f}%, CV: {svm_cv*100:.2f}%")
    
    # Classification reports
    print("\nRandom Forest Classification Report:")
    print(classification_report(y_test, y_pred_rf, target_names=["Non-Cognitive", "Cognitive"]))
    
    print("SVM Classification Report:")
    print(classification_report(y_test, y_pred_svm, target_names=["Non-Cognitive", "Cognitive"]))
    
    # Confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    ConfusionMatrixDisplay(cm_rf, display_labels=["Non-Cognitive", "Cognitive"]).plot(
        ax=axes[0], cmap="Blues", colorbar=True
    )
    axes[0].set_title("Random Forest (File-Level)")
    axes[0].set_ylabel("")
    
    cm_svm = confusion_matrix(y_test, y_pred_svm)
    ConfusionMatrixDisplay(cm_svm, display_labels=["Non-Cognitive", "Cognitive"]).plot(
        ax=axes[1], cmap="Greens", colorbar=True
    )
    axes[1].set_title("SVM (File-Level)")
    axes[1].set_ylabel("")
    
    plt.tight_layout()
    plt.savefig("file_level_confusion_matrices.png", dpi=300)
    plt.show()
    print("Saved: file_level_confusion_matrices.png")
    
    # Bar chart comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = ["RF Test", "RF CV", "SVM Test", "SVM CV"]
    scores = [rf_acc, rf_cv, svm_acc, svm_cv]
    colors = ["#4CAF50", "#81C784", "#2196F3", "#64B5F6"]
    
    bars = ax.bar(labels, [s*100 for s in scores], color=colors, edgecolor='black', linewidth=1.2)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Model Performance Comparison (File-Level)")
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("file_level_accuracy_comparison.png", dpi=300)
    plt.show()
    print("Saved: file_level_accuracy_comparison.png")
    
    return {
        "rf_acc": rf_acc,
        "rf_cv": rf_cv,
        "svm_acc": svm_acc,
        "svm_cv": svm_cv,
        "rf_cm": cm_rf,
        "svm_cm": cm_svm
    }


# ============================================
# MAIN PIPELINE
# ============================================
def main():
    """Run complete EEG MLA pipeline."""
    print("="*60)
    print("EEG COGNITIVE STATE CLASSIFICATION PIPELINE")
    print("="*60)
    
    # Load dataset
    files, labels, common_channels = load_edf_dataset(DATA_DIR)
    
    if len(files) == 0:
        print(f"\nError: No EDF files found in '{DATA_DIR}' directory.")
        print("Please place your EDF files in the 'data' folder.")
        return
    
    # Build file-level features
    print("\nBuilding file-level features...")
    X_file, y_file = build_file_level_features(files, labels, common_channels)
    print(f"File-level dataset: {X_file.shape}")
    
    # Build epoch-level dataset
    print("\nBuilding epoch-level dataset...")
    X_epoch, y_epoch, file_ids = build_epoch_level_dataset(files, labels, common_channels)
    print(f"Epoch-level dataset: {X_epoch.shape}")
    
    # Run evaluations
    epoch_results = evaluate_epoch_level(X_epoch, y_epoch, file_ids)
    file_results = evaluate_file_level(X_file, y_file)
    
    # Summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Epoch-Level (LOFO CV):")
    print(f"  Random Forest: {epoch_results['rf_acc']*100:.2f}%")
    print(f"  SVM: {epoch_results['svm_acc']*100:.2f}%")
    print(f"\nFile-Level (80/20 Split):")
    print(f"  Random Forest: {file_results['rf_acc']*100:.2f}% (CV: {file_results['rf_cv']*100:.2f}%)")
    print(f"  SVM: {file_results['svm_acc']*100:.2f}% (CV: {file_results['svm_cv']*100:.2f}%)")
    print("="*60)


if __name__ == "__main__":
    main()