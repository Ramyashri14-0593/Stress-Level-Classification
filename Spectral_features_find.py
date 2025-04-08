# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8  2025

@author: Ramyashri Ramteke
"""


import numpy as np
from scipy.signal import hilbert

def compute_instfreq_batch(data, fs):
    """
    Compute instantaneous frequency for a batch of signals.

    Parameters:
        data: 2D array (n_signals, n_samples)
        fs: Sampling frequency (Hz)

    Returns:
        inst_freqs: List of 1D arrays (one per signal)
        t: Time vector (shared for all)
    """
import numpy as np
from scipy.signal import spectrogram
from scipy.stats import entropy

def compute_instfreq_and_entropy(data, fs=360, n_windows=524):
    """
    data: ndarray shape (N_signals, N_samples)
    fs: sampling rate (Hz)
    n_windows: number of time windows for spectrogram
    """
    N_signals, N_samples = data.shape
    instfreq_all = []
    pentropy_all = []

    # Calculate nperseg from desired number of time windows
    noverlap = 0
    nperseg = int(N_samples / n_windows)

    for i in range(N_signals):
        signal = data[i]

        # Compute spectrogram
        f, t, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg, noverlap=noverlap, mode='psd')
        Sxx = np.maximum(Sxx, 1e-12)  # Avoid log(0)

        # Normalize the spectrum over frequencies to get a distribution
        Sxx_norm = Sxx / np.sum(Sxx, axis=0, keepdims=True)

        # 1. Instantaneous Frequency (first moment of power spectrum)
        inst_freq = np.sum(f[:, np.newaxis] * Sxx_norm, axis=0)

        # 2. Spectral Entropy
        spec_entropy = entropy(Sxx_norm, base=2, axis=0)

        # Trim or pad to exactly 524 windows
        instfreq = instfreq[:524]
        spec_entropy = spec_entropy[:524]

        instfreq_all.append(inst_freq)
        pentropy_all.append(spec_entropy)

    instfreq_all = np.array(instfreq_all)
    pentropy_all = np.array(pentropy_all)
    combined_features = np.stack((instfreq_all, pentropy_all), axis=2)  # shape: (samples, 524, 2)

    return combined_features, t[:524]



loaded = np.load("ECG_samples.npy")
features, time_axis = compute_fixed_instfreq_entropy(loaded)

print(features.shape)  # (16, 524, 2)


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(times, instfreqs[0])
plt.title("Instantaneous Frequency")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")

plt.subplot(1, 2, 2)
plt.plot(times, pentropies[0])
plt.title("Spectral Entropy")
plt.xlabel("Time (s)")
plt.ylabel("Entropy (bits)")
plt.tight_layout()
plt.show()












