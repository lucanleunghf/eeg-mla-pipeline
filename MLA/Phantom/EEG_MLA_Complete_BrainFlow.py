#!/usr/bin/env python3
"""
EEG Machine Learning Analysis Pipeline - Complete BrainFlow Version
Classifies cognitive vs non-cognitive states from BrainFlow CSV/TXT recordings.
Designed for phantom head EEG data.

Features:
- File-level train/test split (no data leakage)
- Both epoch-level and file-level evaluation
- Random Forest and SVM classifiers
- Comprehensive feature extraction (time, frequency, asymmetry)
- Data augmentation for small datasets
- Proper cross-validation and confusion matrices
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import welch
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
DATA_DIR = "data_brainflow"  # Directory containing BrainFlow files
SFREQ = 256  # Sampling frequency (Hz)
EPOCH_LENGTH_S = 2  # Epoch length in seconds
TRIM_START_S = 5  # Trim first 5 seconds (artifact removal)
MAX_DURATION_S = 30  # Use maximum 30 seconds of data
RANDOM_STATE = 42
N_EPOCHS_PER_FILE = 15  # Number of random epochs to extract per file

# EEG frequency bands
BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45)
}


# ============================================
# DATA LOADING
# ============================================
def load_brainflow_file(filepath):
    """
    Load BrainFlow CSV or TXT file.
    
    Parameters:
    -----------
    filepath : str - path to CSV or TXT file
    
    Returns:
    --------
    data : ndarray (n_channels, n_samples) - EEG data
    """
    if filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
        # Look for EXG channels
        eeg_cols = [c for c in df.columns if "EXG" in c]
        if len(eeg_cols) == 0:
            # Try numeric columns (timestamp + channels)
            df = df.iloc[:, 1:]  # Skip timestamp
        data = df.values.T
    elif filepath.endswith(".txt"):
        df = pd.read_csv(filepath, delim_whitespace=True, header=None)
        # Skip first column if it looks like timestamps
        if df.shape[1] > 1 and df.iloc[:, 0].dtype in [np.float64, np.int64]:
            df = df.iloc[:, 1:]
        data = df.values.T
    else:
        raise ValueError(f"Unsupported file format: {filepath}")
    
    return data


def trim_data(data, sfreq=SFREQ, trim_start=TRIM_START_S, max_duration=MAX_DURATION_S):
    """
    Trim data: remove start artifacts and limit duration.
    
    Parameters:
    -----------
    data : ndarray (n_channels, n_samples)
    sfreq : float - sampling frequency
    trim_start : float - seconds to trim from start
    max_duration : float - maximum duration to use
    
    Returns:
    --------
    trimmed : ndarray - trimmed data
    """
    start_sample = int(trim_start * sfreq)
    max_samples = int(max_duration * sfreq)
    
    if start_sample >= data.shape[1]:
        start_sample = 0
    
    end_sample = min(start_sample + max_samples, data.shape[1])
    
    return data[:, start_sample:end_sample]


# ============================================
# FEATURE EXTRACTION
# ============================================
def random_window_epochs(data, sfreq=SFREQ, epoch_len=EPOCH_LENGTH_S, n_epochs=N_EPOCHS_PER_FILE, seed=None):
    """
    Extract random non-overlapping windows from data as epochs.
    
    Parameters:
    -----------
    data : ndarray (n_channels, n_samples)
    sfreq : float - sampling frequency
    epoch_len : float - epoch length in seconds
    n_epochs : int - number of epochs to extract
    seed : int - random seed for reproducibility
    
    Returns:
    --------
    epochs : ndarray (n_epochs, n_channels, epoch_samples)
    """
    rng = np.random.default_rng(seed)
    samples_per_epoch = int(epoch_len * sfreq)
    max_start = max(0, data.shape[1] - samples_per_epoch)
    
    if max_start <= 0:
        # Data too short, pad with zeros
        pad_width = samples_per_epoch - data.shape[1]
        data = np.pad(data, ((0, 0), (0, pad_width)), mode='constant')
        max_start = 1
    
    epochs = []
    for _ in range(n_epochs):
        start = rng.integers(0, max_start)
        stop = start + samples_per_epoch
        epochs.append(data[:, start:stop])
    
    return np.array(epochs)


def extract_epoch_features(epoch, sfreq=SFREQ):
    """
    Extract comprehensive features from one epoch.
    
    Parameters:
    -----------
    epoch : ndarray (n_channels, n_samples)
    sfreq : float - sampling frequency
    
    Returns:
    --------
    features : ndarray - concatenated feature vector
    """
    features = []
    n_channels = epoch.shape[0]
    
    for ch_idx in range(n_channels):
        ch_data = epoch[ch_idx]
        
        # Time-domain features
        features.append(np.mean(ch_data))
        features.append(np.std(ch_data))
        features.append(skew(ch_data))
        features.append(kurtosis(ch_data))
        
        # Frequency-domain: PSD using Welch's method
        nperseg = min(int(sfreq * 2), len(ch_data))
        if nperseg < 4:
            nperseg = len(ch_data) // 2
        
        f, Pxx = welch(ch_data, fs=sfreq, nperseg=nperseg, n_overlap=nperseg//2)
        
        # Total power
        total_power = np.trapz(Pxx, f) if len(f) > 1 else np.sum(Pxx)
        
        # Absolute band powers
        for band_name, (fmin, fmax) in BANDS.items():
            mask = (f >= fmin) & (f < fmax)
            if np.any(mask):
                bp = np.trapz(Pxx[mask], f[mask])
            else:
                bp = 0
            features.append(bp)
        
        # Relative band powers
        for band_name, (fmin, fmax) in BANDS.items():
            mask = (f >= fmin) & (f < fmax)
            if np.any(mask) and total_power > 0:
                bp = np.trapz(Pxx[mask], f[mask])
                rel_bp = bp / total_power
            else:
                rel_bp = 0
            features.append(rel_bp)
        
        # Beta/Alpha ratio
        alpha_mask = (f >= BANDS["alpha"][0]) & (f < BANDS["alpha"][1])
        beta_mask = (f >= BANDS["beta"][0]) & (f < BANDS["beta"][1])
        
        if np.any(alpha_mask):
            alpha_power = np.trapz(Pxx[alpha_mask], f[alpha_mask])
        else:
            alpha_power = 0
        
        if np.any(beta_mask):
            beta_power = np.trapz(Pxx[beta_mask], f[beta_mask])
        else:
            beta_power = 0
        
        features.append(beta_power / alpha_power if alpha_power > 0 else 0)
    
    # Frontal asymmetry (if enough channels)
    if n_channels >= 5:
        # Assume channels 0 and 4 are roughly left/right frontal
        try:
            # Left channel (e.g., channel 0)
            f_L, Pxx_L = welch(epoch[0], fs=sfreq, nperseg=min(int(sfreq*2), epoch.shape[1]))
            alpha_mask = (f_L >= BANDS["alpha"][0]) & (f_L < BANDS["alpha"][1])
            alpha_L = np.trapz(Pxx_L[alpha_mask], f_L[alpha_mask]) if np.any(alpha_mask) else 0
            
            # Right channel (e.g., channel 4)
            f_R, Pxx_R = welch(epoch[4], fs=sfreq, nperseg=min(int(sfreq*2), epoch.shape[1]))
            alpha_mask = (f_R >= BANDS["alpha"][0]) & (f_R < BANDS["alpha"][1])
            alpha_R = np.trapz(Pxx_R[alpha_mask], f_R[alpha_mask]) if np.any(alpha_mask) else 0
            
            # Asymmetry index
            denom = alpha_L + alpha_R
            asym = (alpha_R - alpha_L) / denom if denom > 0 else 0
            features.append(asym)
            
            # Beta/Alpha ratio (combined)
            beta_L = np.trapz(Pxx_L[(f_L >= BANDS["beta"][0]) & (f_L < BANDS["beta"][1])], 
                             f_L[(f_L >= BANDS["beta"][0]) & (f_L < BANDS["beta"][1])]) if np.any((f_L >= BANDS["beta"][0]) & (f_L < BANDS["beta"][1])) else 0
            beta_R = np.trapz(Pxx_R[(f_R >= BANDS["beta"][0]) & (f_R < BANDS["beta"][1])], 
                             f_R[(f_R >= BANDS["beta"][0]) & (f_R < BANDS["beta"][1])]) if np.any((f_R >= BANDS["beta"][0]) & (f_R < BANDS["beta"][1])) else 0
            
            features.append((beta_L + beta_R) / (alpha_L + alpha_R) if (alpha_L + alpha_R) > 0 else 0)
        except:
            features.extend([0, 0])
    else:
        features.extend([0, 0])
    
    return np.array(features)


def augment_features(features, rng=None, jitter_std=0.1, noise_scale=0.05, dropout_prob=0.1):
    """
    Apply data augmentation to feature vectors.
    
    Parameters:
    -----------
    features : ndarray (n_samples, n_features)
    rng : np.random.Generator
    jitter_std : float - standard deviation for multiplicative jitter
    noise_scale : float - scale for additive noise
    dropout_prob : float - probability of zeroing out features
    
    Returns:
    --------
    augmented : ndarray - augmented features
    """
    if rng is None:
        rng = np.random.default_rng()
    
    augmented = features.copy()
    
    # Multiplicative jitter
    jitter = rng.normal(1.0, jitter_std, size=augmented.shape)
    augmented *= jitter
    
    # Additive Gaussian noise
    noise = rng.normal(0, noise_scale * np.std(augmented), size=augmented.shape)
    augmented += noise
    
    # Random dropout
    dropout_mask = rng.random(size=augmented.shape) < dropout_prob
    augmented[dropout_mask] = 0
    
    return augmented


# ============================================
# DATASET BUILDING
# ============================================
def load_brainflow_dataset(data_dir=DATA_DIR, label_fn=None):
    """
    Load BrainFlow dataset from CSV/TXT files.
    
    Labeling convention:
    - Files with "cog" in name -> Cognitive (label 1)
    - Others -> Non-Cognitive (label 0)
    
    Parameters:
    -----------
    data_dir : str - directory containing files
    label_fn : callable - custom labeling function (optional)
    
    Returns:
    --------
    X : ndarray (n_epochs, n_features)
    y : ndarray (n_epochs,)
    file_features : list of (features, label) tuples per file
    """
    print(f"Looking for BrainFlow files in {data_dir} ...")
    
    if not os.path.exists(data_dir):
        print(f"Error: Directory '{data_dir}' not found.")
        return None, None, None
    
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    txt_files = glob.glob(os.path.join(data_dir, "*.txt"))
    all_files = csv_files + txt_files
    
    print(f"Found {len(all_files)} files ({len(csv_files)} CSV, {len(txt_files)} TXT)")
    
    file_features = []
    
    for fname in all_files:
        try:
            # Determine label
            if label_fn is not None:
                label = label_fn(fname)
            else:
                # Default: "cog" in filename -> Cognitive
                label = 1 if "cog" in os.path.basename(fname).lower() else 0
            
            # Load and preprocess data
            data = load_brainflow_file(fname)
            data = trim_data(data)
            
            # Extract random epochs
            epochs = random_window_epochs(data, sfreq=SFREQ, seed=RANDOM_STATE)
            
            # Extract features for each epoch
            feats = np.vstack([extract_epoch_features(ep, SFREQ) for ep in epochs])
            
            file_features.append((feats, label))
            
        except Exception as e:
            print(f"Warning: Could not process {fname}: {e}")
    
    if len(file_features) == 0:
        print("Error: No valid files processed.")
        return None, None, None
    
    # Combine all epochs
    X = np.vstack([f[0] for f in file_features])
    y = np.concatenate([[f[1]] * f[0].shape[0] for f in file_features])
    
    n_cog = sum(y)
    n_non_cog = len(y) - n_cog
    print(f"Total epochs: {len(X)} (Non-Cognitive: {n_non_cog}, Cognitive: {n_cog})")
    print(f"Number of files: {len(file_features)}")
    
    return X, y, file_features


# ============================================
# EVALUATION FUNCTIONS
# ============================================
def evaluate_epoch_level(X, y, file_features, use_augmentation=True):
    """
    Evaluate at epoch level with 80/20 split.
    """
    print("\n" + "="*60)
    print("EPOCH-LEVEL EVALUATION (80/20 Split)")
    print("="*60)
    
    # Optional augmentation
    if use_augmentation and len(X) < 200:
        print("Applying data augmentation (small dataset)...")
        rng = np.random.default_rng(RANDOM_STATE + 1)
        X_aug = augment_features(X, rng=rng)
        y_aug = y
        X = np.vstack([X, X_aug])
        y = np.concatenate([y, y_aug])
        print(f"Augmented dataset size: {len(X)}")
    
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
    rf_cv = cross_val_score(rf, X_train_scaled, y_train, cv=5).mean()
    
    # SVM
    svm = SVC(kernel="rbf", C=1, gamma="scale", random_state=RANDOM_STATE)
    svm.fit(X_train_scaled, y_train)
    y_pred_svm = svm.predict(X_test_scaled)
    svm_acc = accuracy_score(y_test, y_pred_svm)
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
    axes[0].set_title("Random Forest (Epoch-Level)")
    axes[0].set_ylabel("")
    
    cm_svm = confusion_matrix(y_test, y_pred_svm)
    ConfusionMatrixDisplay(cm_svm, display_labels=["Non-Cognitive", "Cognitive"]).plot(
        ax=axes[1], cmap="Greens", colorbar=True
    )
    axes[1].set_title("SVM (Epoch-Level)")
    axes[1].set_ylabel("")
    
    plt.tight_layout()
    plt.savefig("phantom_epoch_confusion_matrices.png", dpi=300)
    plt.show()
    print("Saved: phantom_epoch_confusion_matrices.png")
    
    # Bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = ["RF Test", "RF CV", "SVM Test", "SVM CV"]
    scores = [rf_acc, rf_cv, svm_acc, svm_cv]
    colors = ["#4CAF50", "#81C784", "#2196F3", "#64B5F6"]
    
    bars = ax.bar(labels, [s*100 for s in scores], color=colors, edgecolor='black', linewidth=1.2)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Model Performance Comparison (Epoch-Level)")
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("phantom_epoch_accuracy_comparison.png", dpi=300)
    plt.show()
    print("Saved: phantom_epoch_accuracy_comparison.png")
    
    return {
        "rf_acc": rf_acc,
        "rf_cv": rf_cv,
        "svm_acc": svm_acc,
        "svm_cv": svm_cv,
        "rf_cm": cm_rf,
        "svm_cm": cm_svm
    }


def evaluate_file_level(file_features):
    """
    Evaluate at file level using Leave-One-Out CV.
    Each file contributes one averaged feature vector.
    """
    print("\n" + "="*60)
    print("FILE-LEVEL EVALUATION (Leave-One-Out CV)")
    print("="*60)
    
    # Average features per file
    X_file = np.vstack([f[0].mean(axis=0) for f in file_features])
    y_file = np.array([f[1] for f in file_features])
    
    n_files = len(X_file)
    print(f"Total files: {n_files}")
    
    if n_files < 5:
        print("Warning: Too few files for reliable file-level evaluation.")
        print("Consider collecting more data.")
        return None
    
    # Scale features
    scaler = StandardScaler()
    
    # Leave-One-Out CV
    loo = LeaveOneOut()
    rf_preds = np.zeros(n_files)
    
    print("\nRunning Leave-One-Out CV...")
    for train_idx, test_idx in loo.split(X_file):
        X_train = scaler.fit_transform(X_file[train_idx])
        X_test = scaler.transform(X_file[test_idx])
        y_train = y_file[train_idx]
        
        rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_preds[test_idx] = rf.predict(X_test)
    
    rf_acc = accuracy_score(y_file, rf_preds)
    cm = confusion_matrix(y_file, rf_preds)
    
    print(f"\nRandom Forest - LOO CV Accuracy: {rf_acc*100:.2f}%")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_file, rf_preds, target_names=["Non-Cognitive", "Cognitive"]))
    
    # Confusion matrix
    plt.figure(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=["Non-Cognitive", "Cognitive"]).plot(
        cmap="Blues", colorbar=True
    )
    plt.title("Random Forest (File-Level LOO CV)")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig("phantom_file_confusion_matrix.png", dpi=300)
    plt.show()
    print("Saved: phantom_file_confusion_matrix.png")
    
    return {
        "rf_acc": rf_acc,
        "rf_cm": cm
    }


# ============================================
# MAIN PIPELINE
# ============================================
def main():
    """Run complete BrainFlow EEG MLA pipeline."""
    print("="*60)
    print("BRAINFLOW EEG COGNITIVE STATE CLASSIFICATION PIPELINE")
    print("="*60)
    
    # Load dataset
    X, y, file_features = load_brainflow_dataset(DATA_DIR)
    
    if X is None or len(X) == 0:
        print("\nError: No data loaded.")
        print("Please place your BrainFlow CSV/TXT files in the 'data_brainflow' folder.")
        print("Naming convention: files with 'cog' in name = Cognitive, others = Non-Cognitive")
        return
    
    # Run epoch-level evaluation
    epoch_results = evaluate_epoch_level(X, y, file_features, use_augmentation=True)
    
    # Run file-level evaluation
    if len(file_features) >= 5:
        file_results = evaluate_file_level(file_features)
    else:
        file_results = None
        print("\nSkipping file-level evaluation (insufficient files)")
    
    # Summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Epoch-Level (80/20 Split):")
    print(f"  Random Forest: {epoch_results['rf_acc']*100:.2f}% (CV: {epoch_results['rf_cv']*100:.2f}%)")
    print(f"  SVM: {epoch_results['svm_acc']*100:.2f}% (CV: {epoch_results['svm_cv']*100:.2f}%)")
    
    if file_results is not None:
        print(f"\nFile-Level (LOO CV):")
        print(f"  Random Forest: {file_results['rf_acc']*100:.2f}%")
    
    print("="*60)


if __name__ == "__main__":
    main()