import json
import ast
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import csv
import time

os.chdir("C:/Git/dynamic-follow-tf-v2/data")
data_dir = "D:/Resilio Sync/dfv2"
driving_data = []

print("Loading data...")
for folder in os.listdir(data_dir):
    for filename in os.listdir(os.path.join(data_dir, folder)):
        print(os.path.join(os.path.join(data_dir, folder), filename))
        with open(os.path.join(os.path.join(data_dir, folder), filename), 'r') as f:
            data = [ast.literal_eval(line) for line in f.read().split('\n') if line != '']  # load and parse data to list of dicts
            
        gas_list = [line['gas'] for line in data]
        min_gas = min(gas_list)
        max_gas = max(gas_list)
        
        for line in data:  # map gas values to appropriate 0 to 1 range and append to driving_data
            new_gas = (line['gas'] - min_gas) / max_gas
            new_gas = new_gas if new_gas > 0.005 else 0  # if gas is less than half a percent, assume 0
            #new_brake = line['brake'] if line['brake'] > 600 else 0  # if brake pressure less than 600 assume no brake
            #new_brake = 0.0 if new_gas > 0 else line['brake'] / 4047.0  # if gas, assume no brake

            line.update({'gas': new_gas})
            #line.update({'brake': new_brake})
            driving_data.append(line)

print("Total samples: {}".format(len(driving_data)))
y_train = [[line['gas'], line['brake']] for line in driving_data]


save_data = True
if save_data:
    print("Saving data...")
    save_dir="brake_pred"
    x_train = [{key: line[key] for key in line if key not in ['gas', 'brake']} for line in data]  # remove gas/brake from x_train
    with open(save_dir+"/x_train", "wb") as f:
        pickle.dump(x_train, f)
    with open(save_dir+"/y_train", "wb") as f:
        pickle.dump(y_train, f)
    try:
        os.remove(save_dir+"/normalized")
    except:
        pass
    print("Saved data!")