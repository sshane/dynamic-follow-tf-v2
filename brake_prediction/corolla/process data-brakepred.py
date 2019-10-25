import json
import ast
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import csv
import time

os.chdir("C:\Git\dynamic-follow-tf-v2\data")
'''with open("traffic-highway/df-data", "r") as f:
    d_data = f.read().split("\n")'''

data_dir = "D:\Resilio Sync\dfv2"
driving_data = []

print("Loading data...")
for folder in os.listdir(data_dir):
    if any([sup_car in folder for sup_car in ["ShaneSmiskol-TOYOTA COROLLA 2017"]]):
        for filename in os.listdir(os.path.join(data_dir, folder)):
            if 'brake-data' in filename:
                print(filename)
                with open(os.path.join(os.path.join(data_dir, folder), filename), "r") as f:
                    data = f.read().split("\n")

                data_parsed = []
                for line in data:
                    try:
                        data_parsed.append(ast.literal_eval(line))
                    except:
                        continue

                keys = data_parsed[0]
                data = data_parsed[1:]

                data = [dict(zip(keys, i)) for i in data]
                for line in data:
                    if line['brake'] >= 0 and line['gas'] == 0:
                        driving_data.append(line)


print("Total samples: {}".format(len(driving_data)))
y_train = [line['brake'] for line in driving_data]

average_y = [i for i in y_train]
average_y = sum(average_y) / len(average_y)
print('Average of samples: {}'.format(average_y))

remove_keys = ['gas', 'brake', 'time']  # remove these unneeded keys in training
save_data = True
if save_data:
    print("Saving data...")
    save_dir = "brake_pred-Corolla"
    x_train = [{key: line[key] for key in line if key not in remove_keys} for line in driving_data]
    with open(save_dir+"/x_train", "wb") as f:
        pickle.dump(np.array(x_train), f)
    with open(save_dir+"/y_train", "wb") as f:
        pickle.dump(np.array(y_train), f)
    print("Saved data!")
