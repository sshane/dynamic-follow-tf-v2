import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, Activation
import numpy as np
import random
#np.set_printoptions(threshold=np.inf)

#mnist = tf.keras.datasets.mnist
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

with open("/Git/dynamic-follow-tf/data/x_train", "r") as f:
    x_train = json.load(f)
x_train = np.asarray([i for i in x_train])


with open("/Git/dynamic-follow-tf/data/y_train", "r") as f:
    y_train = np.asarray([i for i in json.load(f)])

print(len(x_train))
#print(y_train)


#x_train = tf.keras.utils.normalize(x_train)
#y_train = tf.keras.utils.normalize(y_train)
#print(y_train)

#x_train = np.asarray([[[5, 5, 5], [5, 5, 5]], [[1.5, 1.5, 1.5], [1.5, 1.5, 1.5]]])
#y_train = np.asarray([[1.6], [1.5]])

model = Sequential()

model.add(CuDNNLSTM(32, return_sequences=True))
#model.add(Dropout(0.2))

model.add(Dense(16, activation='relu'))
#model.add(Dropout(0.2))


model.add(Dense(1, activation='relu'))
#model.add(Activation("sigmoid"))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-5)
#opt = tf.keras.optimizers.RMSprop(0.001)

model = Sequential([
    CuDNNLSTM(128, input_shape=(x_train.shape[1:]), return_sequences=True),
    CuDNNLSTM(128),
    Dense(64, activation=tf.nn.relu),
    Dense(64, activation=tf.nn.relu),
    Dense(1)
  ])

model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_squared_error'])


model.fit(x_train, y_train, epochs=50)

#print(scaler_y.inverse_transform([[float(model.predict(np.asarray([[[0.0, 6.840000152587891, 0.0]]]))[0][0][0])]]))
#print(model.predict(np.asarray([[[1.5, 1.5, 1.5], [1.5, 1.5, 1.5]]])))
model.save("/Git/dynamic-follow-tf/dynamic_follow_model")