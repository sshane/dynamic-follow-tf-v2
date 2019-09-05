import json
import os
import tensorflow as tf
from keras.models import Sequential
import keras
from keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, Activation, BatchNormalization, LeakyReLU, Flatten, GlobalMaxPooling1D, GlobalAveragePooling1D, Conv1D, MaxPooling1D, Reshape
import numpy as np
import random
from normalizer import norm
from normalizer import get_min_max
from sklearn.preprocessing import normalize
from tensorflow.keras.callbacks import TensorBoard
from keras import backend as K
import time
#np.set_printoptions(threshold=np.inf)

#v_ego, v_lead, d_lead
os.chdir("C:/Git/dynamic-follow-tf")

with open("data/LSTM/x_train", "r") as f:
    x_train = json.load(f)

with open("data/LSTM/y_train", "r") as f:
    y_train = json.load(f)

NORM = False
if NORM:
    x_train_copy = list(x_train)
    v_ego_scale = get_min_max(x_train, 0)
    a_ego_scale = get_min_max(x_train, 1)
    v_lead_scale = get_min_max(x_train, 2)
    x_lead_scale = get_min_max(x_train, 3)
    a_lead_scale = get_min_max(x_train, 4)
    x_train = []
    for idx, i in enumerate(x_train_copy):
        x_train.append([])
        for x in i:
            x_train[idx].append([norm(x[0], v_ego_scale), norm(x[1], a_ego_scale), norm(x[2], v_lead_scale), norm(x[3], x_lead_scale), norm(x[4], a_lead_scale)])
    #x_train = x_train * 2
    #y_train = y_train * 2
    x_train = np.asarray(x_train)
    y_train = np.asarray([np.interp(i, [-1, 1], [0, 1]) for i in y_train])
else:
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)

'''for idx,i in enumerate(y_train):
    if i < -.5 and x_train[idx][0] > 8.9:
        print(i)
        print(idx)
        print(x_train[idx])
        break'''

opt = keras.optimizers.Adam(lr=0.01, decay=1e-4)
#opt = keras.optimizers.Adadelta()
#opt = keras.optimizers.RMSprop(0.001)

'''model = Sequential([
    Dense(5, activation="tanh", input_shape=(x_train.shape[1:])),
    Dense(8, activation="tanh"),
    Dense(16, activation="tanh"),
    Dense(300, activation="tanh"),
    Dense(300, activation="tanh"),
    Dense(300, activation="tanh"),
    Dense(300, activation="tanh"),
    Dense(300, activation="tanh"),
    Dense(300, activation="tanh"),
    Dense(300, activation="tanh"),
    Dense(300, activation="tanh"),
    Dense(300, activation="tanh"),
    Dense(300, activation="tanh"),
    Dense(16, activation="tanh"),
    Dense(8, activation="tanh"),
    Dense(1),
  ])'''

model = Sequential()
model.add(CuDNNLSTM(20, input_shape=(x_train.shape[1:]), return_sequences=True))
#model.add(Dropout(0.2))
#model.add(BatchNormalization(moving_variance_initializer='ones', momentum=.99, epsilon=0.001))

for i in range(10):
    model.add(CuDNNLSTM(20, input_shape=(x_train.shape[1:]), return_sequences=True))
    #model.add(Dropout(0.2))
    #model.add(BatchNormalization(moving_variance_initializer='ones', momentum=.99, epsilon=0.001))

model.add(CuDNNLSTM(10, input_shape=(x_train.shape[1:])))
#model.add(Dropout(0.2))
#model.add(BatchNormalization(moving_variance_initializer='ones', momentum=.99, epsilon=0.001))

model.add(Dense(10, activation="tanh"))
#model.add(Dropout(0.2))

model.add(Dense(10, activation="tanh"))
#model.add(Dropout(0.2))

model.add(Dense(5, activation="tanh"))
#model.add(Dropout(0.2))

model.add(Dense(1))


'''model.add(LSTM(64, return_sequences=True, activation="tanh", input_shape=(x_train.shape[1:])))
model.add(Flatten())
for i in range(20):
    model.add(Dense(40, activation="tanh"))
model.add(Dense(1, activation="linear"))'''

model.compile(loss='mean_absolute_error', optimizer=opt, metrics=['mean_squared_error'])
tensorboard = TensorBoard(log_dir="logs/test-{}".format("333"))
model.fit(x_train, y_train, shuffle=True, batch_size=32, validation_split=.05, epochs=10, callbacks=[tensorboard])

data = [[0.60201864, 0.46189996, 0.23465932, 0.21173397, 0.12633373],
       [0.60153158, 0.46204853, 0.2347616 , 0.21144353, 0.12641934],
       [0.60106121, 0.46102761, 0.23455393, 0.21129831, 0.12637026],
       [0.6005899 , 0.46001266, 0.23446031, 0.21100786, 0.12631119],
       [0.6000443 , 0.45806538, 0.23429521, 0.21100786, 0.12618268],
       [0.59923277, 0.45556536, 0.23415194, 0.21100786, 0.12601838],
       [0.59884671, 0.45799175, 0.23393889, 0.21086263, 0.12576658],
       [0.59846424, 0.45863915, 0.23375225, 0.21086263, 0.12546996],
       [0.59830796, 0.46402394, 0.23368725, 0.21086263, 0.12524679],
       [0.59777   , 0.46328059, 0.23351745, 0.21100786, 0.12499559],
       [0.59736276, 0.46297952, 0.23351672, 0.21100786, 0.12487207],
       [0.59678228, 0.46029385, 0.23381708, 0.21158874, 0.12511619],
       [0.59621012, 0.45937497, 0.2331753 , 0.21129831, 0.12483759],
       [0.5957651 , 0.45896896, 0.23364012, 0.21216963, 0.12509869],
       [0.59527392, 0.4579213 , 0.23403499, 0.21216963, 0.12573533],
       [0.59492696, 0.4606251 , 0.23332487, 0.21246008, 0.12570857],
       [0.59430232, 0.45899378, 0.23365961, 0.21231486, 0.12604952],
       [0.59366072, 0.45717183, 0.23392519, 0.21100786, 0.12662529],
       [0.59324315, 0.45578083, 0.23291179, 0.21057219, 0.12627258],
       [0.59270519, 0.45427143, 0.23332096, 0.21013652, 0.12640437]]
prediction=model.predict(np.asarray([data]))[0][0]

#print((prediction - 0.5)*2.0) if NORM else print(prediction)
print(prediction)


#accur = list([list(i) for i in x_train])

try:
    accuracy=[]
    for i in range(500):
        choice = random.randint(0, len(x_train - 2))
        real=y_train[choice]
        to_pred = list(list(x_train)[choice])
        pred = model.predict(np.asarray([to_pred]))[0][0]
        accuracy.append(abs(real-pred))
        #print("Real: "+str(real))
        #print("Prediction: "+str(pred))
        #print()
    avg = sum(accuracy) / len(accuracy)
    if NORM:
        print("Accuracy: "+ str(abs(avg-1)))
    else:
        print("Accuracy: "+ str(np.interp(avg, [0, 1], [1, 0])))
except:
    pass
    


#test_data = [[norm(15, v_ego_scale), norm(0, a_ego_scale), norm(15, v_lead_scale), norm(18, x_lead_scale), norm(0, a_lead_scale)]]

#print(model.predict(np.asarray(test_data)))

save_model = True
tf_lite = False
if save_model:
    model_name = "LSTM"
    model.save("models/h5_models/"+model_name+".h5")
    if tf_lite:
        # convert model to tflite:
        converter = tf.lite.TFLiteConverter.from_keras_model_file("models/h5_models/"+model_name+".h5")
        tflite_model = converter.convert()
        open("models/lite_models/"+model_name+".tflite", "wb").write(tflite_model)