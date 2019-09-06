import json
import os
import tensorflow as tf
from keras.models import Sequential
import keras
from keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, Activation, LeakyReLU, Flatten
import numpy as np
import random
from normalizer_brakepred import normX
import matplotlib.pyplot as plt
import pickle
from keras import backend as K
from sklearn.model_selection import train_test_split
import shutil
import functools
import operator
from keras.models import load_model

def interp_fast(x, xp, fp=[0, 1]):  # extrapolates above range, np.interp does not
    return (((x - xp[0]) * (fp[1] - fp[0])) / (xp[1] - xp[0])) + fp[0]

os.chdir("C:/Git/dynamic-follow-tf-v2")
data_dir = "brake_pred"
norm_dir = "data/{}/normalized"
model_name = "brake_pred"

print("Loading data...")
with open("data/{}/x_train".format(data_dir), "rb") as f:
    x_train = pickle.load(f)
with open("data/{}/y_train".format(data_dir), "rb") as f:
    y_train = pickle.load(f)

print("Loading data...")
with open("data/live_tracks/x_train", "rb") as f:
    x_train_nobrake = pickle.load(f)
with open("data/live_tracks/y_train", "rb") as f:
    y_train_nobrake = np.array(pickle.load(f))

#x_train = [i for idx, i in enumerate(x_train_data) if y_train_data[idx] >= 0]
#y_train = [i for i in y_train_data if i >= 0]

#y_train_all = list(y_train_data)
#x_train_all = list(x_train_data)

print("Normalizing data...")
x_train, scales = normX(x_train)
x_train = np.array(x_train)
y_train = np.array([interp_fast(i, [-1, 1]) for i in y_train])

#x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1)

opt = keras.optimizers.Adam(lr=0.0001)
#opt = keras.optimizers.Adadelta(lr=.000375)
#opt = keras.optimizers.SGD(lr=0.008, momentum=0.9)
#opt = keras.optimizers.RMSprop(lr=0.00005)#, decay=1e-5)
#opt = keras.optimizers.Adagrad(lr=0.00025)
#opt = keras.optimizers.Adagrad(lr=0.001)
#opt = 'adam'

opt = 'rmsprop'
#opt = keras.optimizers.Adadelta()

layer_num = 4
nodes = 128
a_function = "relu"

model = Sequential()
model.add(Dense(nodes, activation=a_function, input_shape=(x_train.shape[1:])))

for i in range(layer_num - 1):
    model.add(Dense(nodes, activation=a_function))
model.add(Dense(1, activation='linear'))
    

model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
model.fit(x_train, y_train, shuffle=True, batch_size=128, epochs=10, validation_split=0.1)

#x_train_all = x_train_all[20000:25000]
#y_train_all = y_train_all[20000:25000]

'''x = range(len(x_train_all))
a = [[[[np.interp(i[0], scales['v_ego_scale'], [0, 1]), interp_fast(i[1], scales['a_ego_scale'], [0.5, 1])]]] for i in x_train_all]
y = [model.predict(i)[0][0] for i in a]
y2 = [i[1] for i in x_train_all]
plt.clf()
plt.plot(x, y, label='prediction')
plt.plot(x, y2, label='a_ego')
plt.plot(x, y_train_all, label='gas')
plt.legend()
plt.show()

#print("Gas/brake spread: {}".format(sum([model.predict([[[random.uniform(0,1) for i in range(4)]]])[0][0] for i in range(10000)])/10000)) # should be as close as possible to 0.5
'''
preds = []
for idx, i in enumerate(x_train[:10000]):
    preds.append(abs(model.predict([[i]])[0][0] - y_train[idx]))

print("Train accuracy: {}".format(1 - sum(preds) / len(preds)))

'''for c in np.where(y_test==.5)[0][:20]:
    #c = random.randint(0, len(x_test))
    print('Ground truth: {}'.format(y_test[c]))
    print('Prediction: {}'.format(model.predict([[x_test[c]]])[0][0]))
    print()'''

for i in range(20):
    c = random.randint(0, len(x_train))
    print('Ground truth: {}'.format(y_train[c]))
    print('Prediction: {}'.format(model.predict([[x_train[c]]])[0][0]))
    print()


showed = 0
while showed <= 20:
    c = random.randint(0, len(x_train_nobrake))
    if x_train_nobrake[c]['v_ego'] > 8.9 and y_train_nobrake[c] >= 0.0:
        showed+=1
        print('Ground truth: {}'.format(y_train_nobrake[c]))
        to_pred = [interp_fast(x_train_nobrake[c]['v_ego'], scales['v_ego_scale'], [0, 1]), interp_fast(x_train_nobrake[c]['a_ego'], scales['a_ego_scale'], [0, 1])]
        print('Prediction: {}'.format(interp_fast(model.predict([[to_pred]])[0][0], [0, 1], [-1, 1])))
        print()


save_model = True
tf_lite = False
if save_model:
    model.save("models/h5_models/"+model_name+".h5")
    print("Saved model!")
    if tf_lite:
        # convert model to tflite:
        converter = tf.lite.TFLiteConverter.from_keras_model_file("models/h5_models/"+model_name+".h5")
        tflite_model = converter.convert()
        open("models/lite_models/"+model_name+".tflite", "wb").write(tflite_model)