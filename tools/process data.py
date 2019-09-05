import json
import ast
import os
import matplotlib.pyplot as plt
import numpy as np

os.chdir("C:/Git/dynamic-follow-tf/data")
with open("traffic-highway/df-data", "r") as f:
    d_data = f.read().split("\n")

split_leads=False

driving_data = []
for i in d_data:
    if i != "":
        line = ast.literal_eval(i) #remove time for now
        if line[0] < -0.22352 or sum(line) == 0: #or (sum(line[:3]) == 0):
            continue
        if line[4] > 5 or line[4] < -5: # filter out crazy acceleration
            continue
        '''if line[6] > 0: # gas only
            continue'''
        line[0] = max(line[0], 0)
        line[2] = max(line[2], 0)
        line[3] = max(line[3], 0)
        line[6] = 0.0 # remove all brake data
        #line[-1] = line[-1] / 4047.0  # only for corolla
        driving_data.append(line)

if split_leads:
    split_data = [[]]
    counter = 0
    for idx, i in enumerate(driving_data):
        if idx != 0:
            if abs(i[3] - driving_data[idx-1][3]) > 2.5: # if new lead, split
                counter+=1
                split_data.append([])
        split_data[counter].append(i)
    print([len(i) for i in split_data])
    
    new_split_data = []
    for i in split_data:
        if len(i) >= 20: # remove small amounts of data
            new_split_data.append(i)
    
    x_train = []
    y_train = []
    n = 20
    # the following further splits the data into 20 sample sections for training
    for lead in new_split_data:
        '''if lead[0][0] > 4.4704:
            continue'''
        if len(lead) == n: # don't do processing if list is just 20
            x_train.append([i[:5] for i in lead])
            y_train.append(lead[-1][5] - lead[-1][6]) # append last gas/brake values
            continue
        
        lead_split = [lead[i * n:(i + 1) * n] for i in range((len(lead) + n - 1) // n)]
        if lead_split[-1] != 20: # if last section isn't 20 samples, remove
            del lead_split[-1]
        lead_split_x = [[x[:5] for x in i] for i in lead_split] # remove gas and brake from xtrain
        lead_split_y = [(i[-1][5] - i[-1][6]) for i in lead_split] #only (last) gas/brake
        for lead in lead_split_x:
            x_train.append(lead)
        y_train.extend(lead_split_y)
    
    for i in x_train:
        for x in i:
            if len(x) != 5:
                print("uh oh")
        if len(i) != 20:
            print("Bad") # check for wrong sizes in array
    
    save_data = False
    if save_data:
        with open("LSTM/x_train-gbergman", "w") as f:
            json.dump(x_train, f)
        with open("LSTM/y_train-gbergman", "w") as f:
            json.dump(y_train, f)
    
even_out=False
if even_out:
    driving_data_equalized=[]
    counter=0
    for i in driving_data: # even out gas and brake sample numbers
        if i[5] > 0:
            if counter == 2:
                driving_data_equalized.append(i)
                counter=0
            counter+=1
            continue
        else:
            driving_data_equalized.append(i)
    driving_data=driving_data_equalized

print(len(driving_data))
print()
y_train = [i[5] - i[6] for i in driving_data]
print(len([i for i in y_train if i > 0]))
print(len([i for i in y_train if i < 0]))
print(len([i for i in y_train if i == 0]))


save_data=True

if save_data:
    save_dir="brake_to_none"
    x_train = [i[:5] for i in driving_data]
    with open(save_dir+"/x_train", "w") as f:
        json.dump(x_train, f)
    
    with open(save_dir+"/y_train", "w") as f:
        json.dump(y_train, f)

'''x = [i for i in range(len(driving_data))]
y = [i[0] for i in driving_data]
plt.plot(x, y)
plt.show()'''