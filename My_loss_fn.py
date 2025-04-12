# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 20:27:45 2025

@author: Ramyashri Ramteke
"""

import numpy as np
import tensorflow as tf

def compute_class_weights(labels, num_classes=None):
    if num_classes is None:
        num_classes = np.max(labels) + 1
    
    total_samples = len(labels)
    class_counts = np.bincount(labels, minlength=num_classes)
    
    weights = {}
    for i in range(num_classes):
        if class_counts[i] > 0:
            weights[i] = total_samples / class_counts[i]
        else:
            weights[i] = 0.0  

    return weights

def get_weighted_categorical_crossentropy(class_weights):
    weight_vector = tf.constant([class_weights[i] for i in sorted(class_weights)], dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        weights = tf.reduce_sum(weight_vector * y_true, axis=1)
        return loss * weights

    return loss_fn

# Example class labels of training samples for self-created data (e.g., Normal=91, Low=133, High=104)
y = np.array([0]*91 + [1]*133 + [2]*104)  

# Step 1: Compute weights
weights = compute_class_weights(y)
print("Computed class weights:", weights)

# Step 2: Prepare one-hot labels and predictions
y_onehot = tf.keras.utils.to_categorical(y, num_classes=3)

# Step 3: Get the weighted loss function
loss_fn = get_weighted_categorical_crossentropy(weights)

