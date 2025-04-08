# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 17:36:32 2025

@author: Ramyashri Ramteke
"""

import numpy as np
from scipy.signal import find_peaks
from collections import Counter


def normalize_rr_intervals(rr_intervals, target_length=80):
    rr_intervals = np.array(rr_intervals).flatten()
    if len(rr_intervals) < target_length:
        most_common = Counter(rr_intervals).most_common(1)[0][0]
        return np.pad(rr_intervals, (0, target_length - len(rr_intervals)), constant_values=most_common)
    elif len(rr_intervals) > target_length:
        return rr_intervals[:target_length]

    else:
        return rr_intervals
    

def compute_rr_all(ecg_data, fs=360, rr_length=80, peak_height=0.3):
    rr_matrix = []

    for i, signal in enumerate(ecg_data):
        # Basic R-peak detection
        peaks, _ = find_peaks(signal, height=peak_height, distance=fs*0.417)  # at least 0.4s between peaks

        # Compute RR intervals (in ms)
        rr_intervals = np.diff(peaks) / fs 

        # Normalize to fixed length
        normalized_rr = normalize_rr_intervals(rr_intervals, rr_length)

        rr_matrix.append(normalized_rr)

    return np.array(rr_matrix)

loaded = np.load("ECG_samples.npy")
rr_features = compute_rr_all(loaded)

import matplotlib.pyplot as plt
plt.plot(rr_features[0])
plt.title("Normalized RR Intervals (One ECG Sample)")
plt.xlabel("Beat Index")
plt.ylabel("RR Interval (ms)")
plt.grid(True)
plt.show()
