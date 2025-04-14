# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 2025

@author: Ramyashri Ramteke
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Concatenate, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K

# ---- Custom Attention Layer ----
class my_att(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super(my_att,self).__init__(**kwargs)
 
    def build(self,input_shape):
        self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],1), 
                               initializer='random_normal', trainable=True)
        self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],1), 
                               initializer='zeros', trainable=True)        
        super(my_att, self).build(input_shape)
 
    def call(self,x):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x,self.W)+self.b)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)   
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        context = K.sum(context, axis=1)
        return context 

# ---- Model Building Function ----
def build_model(hidden_units=32):
    # Input1: RR intervals (80 samples)
    input1 = Input(shape=(80, 1), name='input_rr')
    x1 = Bidirectional(LSTM(hidden_units, return_sequences=True))(input1)
    att1 = my_att()(x1)  # (None, 64)

    # Input2: Spectrogram features (524 time steps, 2 features)
    input2 = Input(shape=(524, 2), name='input_spec')
    x2 = Bidirectional(LSTM(hidden_units, return_sequences=True))(input2)
    att2 = my_att()(x2)  # (None, 64)

    # Merge
    merged = Concatenate()([att1, att2])  # (None, 128)
    output = Dense(3, activation='softmax')(merged)

    model = Model(inputs=[input1, input2], outputs=output)
    model.compile(optimizer='adam', loss='loss_fn', metrics=['accuracy'])
    return model

# ---- Data Preprocessing Function ----
def prepare_data(rr_data, spec_data, labels, num_classes=3):
    x1 = rr_features.reshape((-1, 80, 1))
    x2 = combined_features  # already (N, 524, 2)
    y = to_categorical(Labels, num_classes=num_classes)
    return [x1, x2], y

loaded_Labels = np.load("labels.npy")
# Prepare
X, Y = prepare_data(rr_features, combined_features, loaded_Labels)    

# Step 2: First split test set (70 samples)
X_zipped = list(zip(X[0], X[1]))  # zip rr and spec together for joint splitting
X_temp, X_test, Y_temp, Y_test = train_test_split(
    X_zipped, Y, test_size=70, random_state=42, stratify=loaded_Labels)

# Step 3: Split validation set (69 samples)
X_train_zipped, X_val_zipped, Y_train, Y_val = train_test_split(
    X_temp, Y_temp, test_size=69, random_state=42, stratify=np.argmax(Y_temp, axis=1))

# Step 4: Unzip the tuples back into separate arrays
X_train = [np.array([x[0] for x in X_train_zipped]), np.array([x[1] for x in X_train_zipped])]
X_val   = [np.array([x[0] for x in X_val_zipped]),   np.array([x[1] for x in X_val_zipped])]
X_test  = [np.array([x[0] for x in X_test]),         np.array([x[1] for x in X_test])]

# Build & Train
model = build_model()
model.summary()

earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=3, mode='min')
history=model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=500, batch_size=64, callbacks=[earlystop, reduce_lr])


# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(X_test, Y_test, verbose=1)

# Print results
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

print('Training complete...\n')    

import matplotlib.pyplot as plt

# Plot training & validation accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.grid(True)

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title('model Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
