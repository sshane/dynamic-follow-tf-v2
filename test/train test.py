import json
import os
import tensorflow as tf
from keras.models import Sequential
import keras
from keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, Activation, LeakyReLU
from keras.layers.advanced_activations import ELU
from keras.activations import selu
import numpy as np
import random
from tensorflow.keras.callbacks import TensorBoard
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pickle
from keras import backend as K
import shutil
os.chdir('C:/Git/dynamic-follow-tf/test')
#np.set_printoptions(threshold=np.inf)
'''from numpy.random import seed
seed(5)
from tensorflow import set_random_seed
set_random_seed(3)'''

'''gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))'''

class Visualize(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        preds = []
        reals = []
        for random_choice in range(len(random_choices)):
            pred = model.predict([[x_train[random_choice]]])[0][0]
            real = y_train[random_choice]
            #print("Real: {}".format(real))
            #print("Prediction: {}\n".format(pred))
            reals.append(real)
            preds.append(pred)
        
        plt.clf()
        plt.plot(range(len(preds)), reals, label='ground-truth', marker='o')
        plt.plot(range(len(preds)), preds, label='prediction-{}'.format(epoch), marker='o')
        plt.legend()
        plt.pause(.1)
        plt.show()


with open('x_train', 'rb') as f:
    x_train = pickle.load(f)

with open('y_train', 'rb') as f:
    y_train = pickle.load(f)

all_vals = []
for sample in x_train:
    for i in sample:
        all_vals.append(i)

min_max = [min(all_vals), max(all_vals)]

x_train = np.array([[np.interp(i, min_max, [0, 1]) for i in sample] for sample in x_train])

random_choices = [random.randint(0, len(x_train) - 1) for i in range(50)]

#opt = keras.optimizers.Adam(lr=0.1)#, decay=1e-6)
opt = keras.optimizers.Adadelta()
#opt = keras.optimizers.SGD(lr=0.01, momentum=0.9)
#opt = keras.optimizers.RMSprop(lr=0.001, decay=1e-5)
#opt = keras.optimizers.Adagrad(lr=0.1)
#opt = 'adam'

#[12, 324]
options=[[5, 512]] # good ones: [[8, 1000], [7, 2500], [4, 2048], [4, 4096]], best so far: [[3, 8096], [2, 8096]] (adadelta)

for i in options:
    layer_num=i[0] - 1
    nodes=i[1]
    a_function="relu"
    
    model = Sequential()
    model.add(Dense(nodes, activation=a_function, input_shape=(x_train.shape[1:])))
    for i in range(layer_num):
        model.add(Dense(nodes, activation=a_function))
    model.add(Dense(1))
        
    callbacks = [Visualize()]
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
    #tensorboard = TensorBoard(log_dir="logs/{}-layers-{}-nodes-{}".format(layer_num, nodes, a_function))
    model.fit(x_train, y_train, shuffle=True, batch_size=32, epochs=10, callbacks=callbacks) #callbacks=[tensorboard])
