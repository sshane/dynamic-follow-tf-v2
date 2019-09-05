import json
import os
import tensorflow as tf
from keras.models import Sequential
import keras
from keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, Activation, LeakyReLU, Flatten
import numpy as np
import random
from normalizer import normX
import matplotlib.pyplot as plt
import pickle
from keras import backend as K
from sklearn.model_selection import train_test_split
import shutil
import functools
import operator
from keras.models import load_model

def interp_fast(x, xp, fp):  # extrapolates above range, np.interp does not
    return (((x - xp[0]) * (fp[1] - fp[0])) / (xp[1] - xp[0])) + fp[0]

os.chdir("C:/Git/dynamic-follow-tf-v2")
data_dir = "brake_pred"
norm_dir = "data/{}/normalized"
model_name = "brake_pred"

try:
    shutil.rmtree("models/h5_models/{}".format(model_name))
except:
    pass


print("Loading data...")
with open("data/{}/x_train".format(data_dir), "rb") as f:
    x_train_data = pickle.load(f)
with open("data/{}/y_train".format(data_dir), "rb") as f:
    y_train_data = pickle.load(f)

y_train = [i[0] for idx, i in enumerate(y_train_data) if x_train_data[idx]['a_ego'] >= 0 and i[1] == 0]
x_train = [[i['v_ego'], i['a_ego']] for idx, i in enumerate(x_train_data) if i['a_ego'] >= 0 and y_train_data[idx][1] == 0]

y_train_all = [i[0] for idx, i in enumerate(y_train_data)]
x_train_all = [[i['v_ego'], i['a_ego']] for idx, i in enumerate(x_train_data)]

print("Normalizing data...")
x_train, scales = normX(x_train)
x_train = np.array(x_train)
y_train = np.array(y_train)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.05)

opt = keras.optimizers.Adam(lr=0.0001)
#opt = keras.optimizers.Adadelta(lr=.000375)
#opt = keras.optimizers.SGD(lr=0.008, momentum=0.9)
#opt = keras.optimizers.RMSprop(lr=0.00005)#, decay=1e-5)
#opt = keras.optimizers.Adagrad(lr=0.00025)
#opt = keras.optimizers.Adagrad(lr=0.001)
#opt = 'adam'

opt = 'rmsprop'
#opt = keras.optimizers.Adadelta()

layer_num = 2
nodes = 64
a_function = "relu"

model = Sequential()
model.add(Dense(nodes, activation=a_function, input_shape=(x_train.shape[1:])))

for i in range(layer_num - 1):
    model.add(Dense(nodes, activation=a_function))
model.add(Dense(1, activation='linear'))
    

model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
model.fit(x_train, y_train, shuffle=True, batch_size=16, epochs=50, validation_data=(x_test, y_test))

x_train_all = x_train_all[20000:25000]
y_train_all = y_train_all[20000:25000]

x = range(len(x_train_all))
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

preds = []
for idx, i in enumerate(x_test):
    preds.append(abs(model.predict([[i]])[0][0] - y_test[idx]))

'''print("Test accuracy: {}".format(1 - sum(preds) / len(preds)))

for c in np.where(y_test==.5)[0][:20]:
    #c = random.randint(0, len(x_test))
    print('Ground truth: {}'.format(y_test[c]))
    print('Prediction: {}'.format(model.predict([[x_test[c]]])[0][0]))
    print()

for i in range(20):
    c = random.randint(0, len(x_test))
    print('Ground truth: {}'.format(y_test[c]))
    print('Prediction: {}'.format(model.predict([[x_test[c]]])[0][0]))
    print()

preds = []
for idx, i in enumerate(x_train):
    preds.append(abs(model.predict([[i]])[0][0] - y_train[idx]))

print("Train accuracy: {}".format(1 - sum(preds) / len(preds)))'''

save_model = False
tf_lite = False
if save_model:
    model.save("models/h5_models/"+model_name+".h5")
    print("Saved model!")
    if tf_lite:
        # convert model to tflite:
        converter = tf.lite.TFLiteConverter.from_keras_model_file("models/h5_models/"+model_name+".h5")
        tflite_model = converter.convert()
        open("models/lite_models/"+model_name+".tflite", "wb").write(tflite_model)