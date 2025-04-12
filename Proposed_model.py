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

# Build & Train
model = build_model()
model.summary()

earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=3, mode='min')
history=model.fit(X, Y, epochs=500, batch_size=64, callbacks=[earlystop, reduce_lr], validation_split=0.15)

print('Training complete...\n')    

