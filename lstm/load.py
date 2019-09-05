import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from normalizer import norm
import itertools
os.chdir("C:/Git/dynamic-follow-tf/models/h5_models")
scales = {'v_ego_scale': [0.0, 39.129379272461],
 'v_lead_scale': [0.0, 44.459167480469],
 'x_lead_scale': [0.375, 146.375]}
model = tf.keras.models.load_model("LSTM-epoch-1.h5")

data = [[0.3148282177454169, 0.300430513661197, 0.0913162361862313],
 [0.3146124942623468, 0.29999850440203135, 0.0913162361862313],
 [0.31439124840786936, 0.29999850440203135, 0.0913162361862313],
 [0.31419850360582485, 0.29999850440203135, 0.0913162361862313],
 [0.3140565599407523, 0.29999850440203135, 0.0913162361862313],
 [0.3139184538558104, 0.2992213934104919, 0.0913162361862313],
 [0.3137209120786027, 0.2992213934104919, 0.0913162361862313],
 [0.313532566453781, 0.2992213934104919, 0.0913162361862313],
 [0.3133691884996867, 0.2992213934104919, 0.0913162361862313],
 [0.3132575944774759, 0.2992213934104919, 0.0913162361862313],
 [0.3131318903283217, 0.2992213934104919, 0.0913162361862313],
 [0.3130310836502584, 0.29986623611729624, 0.09186325073242188],
 [0.31291286746233454, 0.29986623611729624, 0.09186325073242188],
 [0.312785104123842, 0.29986623611729624, 0.09186325073242188],
 [0.3125670640527663, 0.29986623611729624, 0.09186325073242188],
 [0.3123764252398178, 0.29986623611729624, 0.09186325073242188],
 [0.3121874478182673, 0.2999742919993502, 0.09213675474509214],
 [0.3119915674324576, 0.2999742919993502, 0.09213675474509214],
 [0.31179989902483995, 0.2999742919993502, 0.09213675474509214],
 [0.31162872891109034, 0.2999742919993502, 0.09213675474509214]]

data = [i for x in data for i in x]
prediction=model.predict([[data]])
print(prediction)

data = [[0.5356867808281186, 0.4540627726038027, 0.43082622299846424],
 [0.535675408487, 0.4540627726038027, 0.43082622299846424],
 [0.5355506403329997, 0.4540627726038027, 0.43082622299846424],
 [0.5354325645443488, 0.4541867915301022, 0.4302792019314236],
 [0.5353268438917281, 0.4541867915301022, 0.4302792019314236],
 [0.5352616518292665, 0.4541867915301022, 0.4302792019314236],
 [0.5351366964762354, 0.4541867915301022, 0.4302792019314236],
 [0.5349955952068013, 0.4541867915301022, 0.4302792019314236],
 [0.5348462571800139, 0.4541867915301022, 0.4302792019314236],
 [0.5348256652866304, 0.4546296428032118, 0.42973218086438303],
 [0.5348081153774968, 0.4546296428032118, 0.42973218086438303],
 [0.5346404318456951, 0.4546296428032118, 0.42973218086438303],
 [0.5344651667531477, 0.4546296428032118, 0.42973218086438303],
 [0.5343606160944692, 0.4547369487435898, 0.4296410193810096],
 [0.5342507302634141, 0.4547369487435898, 0.4296410193810096],
 [0.5341528719700851, 0.4547369487435898, 0.4296410193810096],
 [0.5340428457397569, 0.4547369487435898, 0.4296410193810096],
 [0.0970380924755715, -0.0011691376459236603, 0.15408546904213408],
 [0.09676130700865594, -0.0011691376459236603, 0.15408546904213408],
 [0.09649931026517028, -0.0011691376459236603, 0.15408546904213408]]

data = [i for x in data for i in x]
prediction=model.predict([[data]])
print(prediction)