import json
import ast
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import csv
import time

def interp_fast(x, xp, fp):  # extrapolates above range, np.interp does not
    return (((x - xp[0]) * (fp[1] - fp[0])) / (xp[1] - xp[0])) + fp[0]

os.chdir("C:/Git/dynamic-follow-tf-v2/data")
data_dir = "D:/Resilio Sync/dfv2"
driving_data = []

print("Loading data...")
for folder in os.listdir(data_dir):
    for filename in os.listdir(os.path.join(data_dir, folder)):
        if 'old' not in filename and '.txt' not in filename and filename == 'df-data.12':
            filepath = os.path.join(os.path.join(data_dir, folder), filename)
            print('Processing: {}'.format(filepath))
            with open(filepath, 'r') as f:
                data = [ast.literal_eval(line) for line in f.read().split('\n') if line != '' and line[-1] == '}']  # load and parse data to list of dicts
            
            pedal_gas = [line['gas'] for line in data]  # data from comma pedal
            car_gas = [line['car_gas'] for line in data]  # data from car's can
            
            max_car_gas = max(car_gas)  # this is most accurate way to map pedal gas to car gas range (0 to 1)
            max_pedal_gas = pedal_gas[car_gas.index(max_car_gas)]
            
            min_car_gas = [i for i in car_gas if i > 0.0][0]
            min_pedal_gas = pedal_gas[car_gas.index(min_car_gas)]
            
            pedal_gas = [interp_fast(i, [min_pedal_gas, max_pedal_gas], [min_car_gas, max_car_gas]) for i in pedal_gas]
            pedal_gas = [i if i >= 0 else 0 for i in pedal_gas]
            
            #gas_list = [line['gas'] for line in data]
            #min_gas = min(gas_list)
            #max_gas = max(gas_list)
            
            for line in data:  # map gas and brake values to appropriate 0 to 1 range and append to driving_data
                new_gas = interp_fast(line['gas'], [min_pedal_gas, max_pedal_gas], [min_car_gas, max_car_gas])
                new_gas = new_gas if new_gas >= 0 else 0  # if remapped gas is negative, no gas
                
                #new_gas = (line['gas'] - min_gas) / max_gas
                #new_gas = new_gas if new_gas > 0.005 else 0  # if gas is less than half a percent, assume 0
                new_brake = line['brake'] / 4047.0 if line['brake'] > 512 and new_gas == 0.0 else 0  # throw out brake when gas is applied or pressure less than or equal to 512
                #new_brake = line['brake'] if line['brake'] > 600 else 0  # if brake pressure less than 600 assume no brake
                #new_brake = 0.0 if new_gas > 0 else line['brake'] / 4047.0  # if gas, assume no brake
    
                line.update({'gas': new_gas})
                line.update({'brake': new_brake})
                driving_data.append(line)

print("Total samples: {}".format(len(driving_data)))
y_train = [line['gas'] - line['brake'] for line in driving_data]
print("Gas samples: {}".format(len([i for i in y_train if i > 0])))
print("Coast samples: {}".format(len([i for i in y_train if i == 0])))
print("Brake samples: {}".format(len([i for i in y_train if i < 0])))

average_y = [i for i in y_train]
average_y = sum(average_y) / len(average_y)
print('Average of samples: {}'.format(average_y))

save_data = True
if save_data:
    print("Saving data...")
    save_dir = "live_tracks"
    x_train = [{key: line[key] for key in line if key not in ['gas', 'brake']} for line in driving_data]  # remove gas/brake from x_train
    with open(save_dir+"/x_train", "wb") as f:
        pickle.dump(x_train, f)
    with open(save_dir+"/y_train", "wb") as f:
        pickle.dump(y_train, f)
    try:
        os.remove(save_dir+"/normalized")
    except:
        pass
    print("Saved data!")