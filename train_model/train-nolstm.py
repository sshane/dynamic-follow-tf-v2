'''from numpy.random import seed
seed(255)
from tensorflow import set_random_seed
set_random_seed(255)'''
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
import os
import load_brake_pred_model as lbpm

def interp_fast(x, xp, fp=[0, 1], ext=False):  # extrapolates above range when ext is True
    interped = (((x - xp[0]) * (fp[1] - fp[0])) / (xp[1] - xp[0])) + fp[0]
    return interped if ext else min(max(fp[0], interped), fp[1])

brake_model, brake_scales = lbpm.get_brake_pred_model()

def pad_tracks(tracks, max_tracks):
    to_add = max_tracks - len(tracks)
    to_add_left = to_add - (to_add // 2)
    to_add_right = to_add - to_add_left
    to_pad = [[0, 0, 0]]
    #return tracks + (to_add * to_pad)
    return (to_pad * to_add_left) + tracks + (to_pad * to_add_right)

os.chdir("C:/Git/dynamic-follow-tf-v2")
data_dir = "live_tracks"
norm_dir = "data/{}/normalized"
model_name = "live_tracks"

try:
    shutil.rmtree("models/h5_models/{}".format(model_name))
except:
    pass

'''for idx, i in enumerate(x_train):
    if y_train[idx] < 0.0:
        y_train[idx] = -0.2'''


if os.path.exists("data/{}/x_train_normalized".format(data_dir)):
    print('Loading normalized data', flush=True)
    with open("data/{}/x_train_normalized".format(data_dir), "rb") as f:
        tracks_normalized, car_data_normalized, scales = pickle.load(f)
    with open("data/{}/y_train_normalized".format(data_dir), "rb") as f:
        y_train = pickle.load(f)
    max_tracks = scales['max_tracks']
else:
    print("Loading data...", flush=True)
    with open("data/{}/x_train".format(data_dir), "rb") as f:
        x_train = pickle.load(f)
    with open("data/{}/y_train".format(data_dir), "rb") as f:
        y_train = pickle.load(f)
    
    max_tracks = max([len(i['live_tracks']['tracks']) for i in x_train])  # max number of tracks in all samples
    tracks = [line['live_tracks']['tracks'] for line in x_train] # only tracks
    
    # get relevant training car data to normalize
    car_data = [[line['v_ego'], line['steer_angle'], line['steer_rate'], line['a_lead'], line['set_speed'], line['left_blinker'], line['right_blinker'], line['status']] for line in x_train]
    
    print("Normalizing data...", flush=True)  # normalizes track dicts into [yRel, dRel, vRel trackStatus (0/1)] lists for training
    tracks_normalized, car_data_normalized, scales = normX(tracks, car_data)  # normalizes data and adds blinkers
    scales['max_tracks'] = max_tracks
    
    print("Predicting brake data...", flush=True)
    for idx, i in enumerate(x_train):  # use brake model to predict what the brake value is from vego and aego, above 2 mph
        if y_train[idx] < 0.0:  # if brake sample
            to_pred = [interp_fast(i['v_ego'], brake_scales['v_ego_scale']), interp_fast(i['a_ego'], brake_scales['a_ego_scale'])]
            predicted_brake = interp_fast(brake_model.predict([[to_pred]])[0][0], [0, 1], [-1, 1])
            if predicted_brake <= 0.0: # if prediction is to accel, default to coast (might want to choose arbitrary brake value)
                y_train[idx] = (predicted_brake - 0.02) * 1.05  # increase predicted brake to add weight
            else:
                y_train[idx] = -0.1
    
    y_train = np.array([interp_fast(i, [-1, 1]) for i in y_train])
    
    with open("data/{}/x_train_normalized".format(data_dir), "wb") as f:
        pickle.dump([tracks_normalized, car_data_normalized, scales], f)
    
    with open("data/{}/y_train_normalized".format(data_dir), "wb") as f:
        pickle.dump(y_train, f)
#print(''+1)

#Format data
print("Sorting tracks...")
tracks_sorted = [sorted(line, key=lambda track: track[0]) for line in tracks_normalized]  # sort tracks by yRel

# pad tracks to max_tracks length so the shape is correct for training (keeps data in center of pad)
tracks_padded = [line if len(line) == max_tracks else pad_tracks(line, max_tracks) for line in tracks_sorted]  # tracks_sorted

# flatten tracks to 1d array
flat_tracks = [[item for sublist in sample for item in sublist] for sample in tracks_padded]

# combine into one list
x_train = np.array([i[0] + i[1] for i in zip(car_data_normalized, flat_tracks)])

#y_train = np.array([i if i >= 0 else 0.0 for i in y_train])  # pick some constant arbitrary negative value so we know when to warn user

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.05)
print(x_train.shape)

try:
    os.mkdir("models/h5_models/{}".format(model_name))
except:
    pass

#opt = keras.optimizers.Adam(lr=0.01, decay=1.75e-4)
opt = keras.optimizers.Adadelta() #lr=.000375)
#opt = keras.optimizers.SGD(lr=0.008, momentum=0.9)
#opt = keras.optimizers.RMSprop(lr=0.00005)#, decay=1e-5)
#opt = keras.optimizers.Adagrad(lr=0.00025)
#opt = keras.optimizers.Adagrad()
#opt = 'adam'

opt = 'rmsprop'
#opt = keras.optimizers.Adadelta()

layer_num = 5
nodes = 256
a_function = "relu"

model = Sequential()
model.add(Dense(x_train.shape[1], activation=a_function, input_shape=(x_train.shape[1:])))

for i in range(layer_num):
    model.add(Dense(nodes, activation=a_function))
model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
model.fit(x_train, y_train, shuffle=True, batch_size=300, epochs=200, validation_data=(x_test, y_test))
#model = load_model("models/h5_models/{}.h5".format('live_tracksv6'))

#print("Gas/brake spread: {}".format(sum([model.predict([[[random.uniform(0,1) for i in range(4)]]])[0][0] for i in range(10000)])/10000)) # should be as close as possible to 0.5

seq_len = 100
plt.clf()
rand_start = random.randint(0, len(x_train) - seq_len)
x = range(seq_len)
y = y_train[rand_start:rand_start+seq_len]
y2 = [model.predict([[i]])[0][0] for i in x_train[rand_start:rand_start+seq_len]]
plt.plot(x, y, label='ground truth')
plt.plot(x, y2, label='prediction')
plt.legend()
plt.show()


'''preds = []
for idx, i in enumerate(x_test):
    preds.append(abs(model.predict([[i]])[0][0] - y_test[idx]))

print("Test accuracy: {}".format(1 - sum(preds) / len(preds)))

for c in np.where(y_test==0.5)[0][:20]:
    #c = random.randint(0, len(x_test))
    print('Ground truth: {}'.format(interp_fast(y_test[c], [0, 1], [-1, 1])))
    print('Prediction: {}'.format(interp_fast(model.predict([[x_test[c]]])[0][0], [0, 1], [-1, 1])))
    print()

for c in np.where(y_test>0.5)[0][:20]:
    #c = random.randint(0, len(x_test))
    print('Ground truth: {}'.format(interp_fast(y_test[c], [0, 1], [-1, 1])))
    print('Prediction: {}'.format(interp_fast(model.predict([[x_test[c]]])[0][0], [0, 1], [-1, 1])))
    print()

for c in np.where(y_test<0.5)[0][:20]:
    #c = random.randint(0, len(x_test))
    print('Ground truth: {}'.format(interp_fast(y_test[c], [0, 1], [-1, 1])))
    print('Prediction: {}'.format(interp_fast(model.predict([[x_test[c]]])[0][0], [0, 1], [-1, 1])))
    print()'''

'''preds = []
for idx, i in enumerate(x_train):
    preds.append(abs(model.predict([[i]])[0][0] - y_train[idx]))

print("Train accuracy: {}".format(1 - sum(preds) / len(preds)))'''

'''preds = [[]]*16  # gernby acc code
for idx, i in enumerate(x_train):
    pred = model.predict([[i]])[0]
    acc = [abs(y_train[idx][idi] - i) for idi, i in enumerate(pred)]
    for idi, i in enumerate(acc):
        preds[idi].append(i)

for i in preds:
    print(sum(i)/len(i))'''

tf_lite = False
def save_model():
    model.save("models/h5_models/"+model_name+".h5")
    print("Saved model!")
    if tf_lite:
        # convert model to tflite:
        converter = tf.lite.TFLiteConverter.from_keras_model_file("models/h5_models/"+model_name+".h5")
        tflite_model = converter.convert()
        open("models/lite_models/"+model_name+".tflite", "wb").write(tflite_model)
#save_model()