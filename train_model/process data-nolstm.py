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
supported_users = ['ShaneSmiskol-TOYOTA COROLLA 2017']
consider_set_speed = False  # removing set_speed for now
use_pedal = False

print("Loading data...")
for folder in [i for i in os.listdir(data_dir) if i in supported_users]:
    for filename in os.listdir(os.path.join(data_dir, folder)):
        if 'old' not in filename and '.txt' not in filename:
            file_path = os.path.join(os.path.join(data_dir, folder), filename)
            print('Processing: {}'.format(file_path))
            with open(file_path, 'r') as f:
                df_data = f.read().replace("'", '"').replace('False', 'false').replace('True', 'true')

            data = []
            for sample in df_data.split('\n'):
                try:
                    data.append(json.loads(sample))
                except:
                    pass

            new_format = type(data[0]) == list  # new space saving format, this will convert it to list of dicts
            if new_format:
                keys = data[0]  # gets keys and removes keys from data
                if 'track_data' in keys:
                    keys[keys.index('track_data')] = 'live_tracks'
                if 'lead_status' in keys:
                    keys[keys.index('lead_status')] = 'status'
                if 'time.time()' in keys:
                    keys[keys.index('time.time()')] = 'time'
                data = data[1:]
                data = [dict(zip(keys, i)) for i in data]
            
            #pedal = any([True if i['car_gas'] > 0.0 else False for i in data])
            #if pedal:
            if use_pedal:
                pedal_gas = [line['gas'] for line in data]  # data from comma pedal
                car_gas = [line['car_gas'] for line in data]  # data from car's can

                max_car_gas = max(car_gas)  # this is most accurate way to map pedal gas to car gas range (0 to 1)
                max_pedal_gas = pedal_gas[car_gas.index(max_car_gas)]  # comma pedal has more resolution than car's sensor

                min_car_gas = [i for i in car_gas if i > 0.0][0]
                min_pedal_gas = pedal_gas[car_gas.index(min_car_gas)]

                pedal_gas = [interp_fast(i, [min_pedal_gas, max_pedal_gas], [min_car_gas, max_car_gas]) for i in pedal_gas]
                pedal_gas = [i if i >= 0 else 0 for i in pedal_gas]
            #else:
                #print('Not pedal!')

            for line in data:  # map gas and brake values to appropriate 0 to 1 range and append to driving_data
                if (line['set_speed'] == 0.0 or (line['set_speed'] > line['v_ego'] and line['car_gas'] > .15)) and consider_set_speed:
                    continue
                if use_pedal:
                    new_gas = (interp_fast(line['gas'], [min_pedal_gas, max_pedal_gas], [min_car_gas, max_car_gas]) * 0.4) + (line['car_gas'] * 0.6)  # makes it so that it's closer to the car's gas sensor while still preserving some extra accuracy (noise?) from pedal sensor
                    new_gas = new_gas if new_gas >= 0 and line['car_gas'] != 0 else 0  # if remapped gas is negative, no gas
                    line.update({'gas': new_gas})
                else:
                    line['gas'] = float(line['car_gas'])

                new_brake = line['brake'] / 4047.0 if line['brake'] >= 256 and line['gas'] == 0.0 else 0  # throw out brake when gas is applied or pressure less than or equal to 512
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
#plt.plot(range(len(driving_data)), [i['decel_for_model'] for i in driving_data])

remove_keys = ['gas', 'brake', 'v_lat', 'car_gas', 'path_curvature', 'decel_for_model', 'a_rel', 'y_lead']  # remove these unneeded keys in training
save_data = True
if save_data:
    print("Saving data...")
    save_dir = "live_tracks"
    x_train = [{key: line[key] for key in line if key not in remove_keys} for line in driving_data]  # remove gas/brake from x_train
    with open(save_dir+"/x_train", "wb") as f:
        pickle.dump(x_train, f)
    with open(save_dir+"/y_train", "wb") as f:
        pickle.dump(y_train, f)
    
    to_remove = ["/x_train_normalized", "/y_train_normalized"]
    for i in to_remove:
        try:
            os.remove(save_dir + i)
        except:
            pass
    print("Saved data!")