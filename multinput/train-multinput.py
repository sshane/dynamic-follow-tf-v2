import json
import os
import tensorflow as tf
from keras.models import Sequential
import keras
from keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, Activation, LeakyReLU, Flatten, Input
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
from keras import backend as K
import shutil
import functools
import operator
import os
from keras.models import Model
from keras.utils import plot_model
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'
os.chdir("C:/Git/dynamic-follow-tf-v2/multinput")

x_train = np.array([[.1], [.2], [.5], [.4], [.9], [.3], [.4]])
x_train2 = np.array([[.2], [.5], [.5], [.5], [.05], [.3], [0]])
y_train = np.array([.3, .7, 1.0, .9, .95, .6, .4])

input_a = Input(shape=(1,), name='input_a')

x = Dense(64, activation='relu')(input_a)
output_a = Dense(64, activation='relu')(x)

input_b = Input(shape=(1,), name='input_b')
x = Dense(64, activation='relu')(input_b)
output_b = Dense(64, activation='relu')(x)
x = keras.layers.concatenate([output_a, output_b])

output_c = Dense(1)(x)

#model = Model(inputs=input_a, outputs=main_output)
model = Model(inputs=[input_a, input_b], outputs=output_c)
plot_model(model, to_file='model.png')
model.compile(optimizer='adam',
              loss='mse',
              metrics=['mae'])
model.fit([x_train, x_train2], y_train, epochs=100)  # starts training