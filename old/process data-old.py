import json
with open("/Git/dynamic-follow-tf/data/driving_data.json", "r") as f:
    d_data = json.loads(f.read())



# split data into different leads
last_time = d_data[0][3]
last_distance = d_data[0][1]
data_lead_split = []

lead_counter=0
current_lead = []
for i in d_data:
    if i[0] > 0:
        current_TR = i[1] / i[0]
    else:
        current_TR = None
    current_lead.append([max(i[0], 0), max(i[1], 0), max(i[2], 0), i[3], current_TR]) # make sure v_ego, drel and vlead aren't below 0
    if (i[3] - last_time) > 1 or abs(i[1] - last_distance) > 6:
        data_lead_split.append(current_lead)
        current_lead = []
        lead_counter+=1
        #print("New lead, time since last lead: "+str(i[3]-last_time))
    last_time=i[3]
    last_distance=i[1]

print(len(data_lead_split))


# remove leads under set amount time, useless
new_data_lead_split = []
minimum_lead_time = 5 # seconds
for i in data_lead_split:
    if i[-1][3] - i[0][3] > minimum_lead_time and len(i) > minimum_lead_time * 10:
        list_builder = []
        for x in i:
            list_builder.append({"v_ego": x[0], "dRel": x[1], "vLead": x[2], "time": x[3],"realTR": x[4]})  # convert list to dictionary for ease of reading
        new_data_lead_split.append(list_builder)

#print(len(data_lead_split))
#print([len(i) for i in data_lead_split])

data_lead_split = []
for i in new_data_lead_split:
    tmp_split = []
    for point in i:
        tmp_split.append(point)
        if len(tmp_split) == 200 and sum([this_tmp['v_ego'] for this_tmp in tmp_split]) > 0:
            data_lead_split.append(tmp_split)
            tmp_split = []
#print(len(data_lead_split))
#print([len(i) for i in data_lead_split])

x_train = []
y_train = []

for idx, i in enumerate(data_lead_split):
    x_train.append([])
    for idi, x in enumerate(i):
        try:
            future_data = i[idi+50]
            x_train[idx].append([max(x["v_ego"], 0), max(x["dRel"], 0), max(x["vLead"], 0)])
        
        except:
            pass
    if i[-1]["v_ego"] > 0:
        TR=max(i[-1]["dRel"], 0) / max(i[-1]["v_ego"], 0)
        if i[-1]["v_ego"] < 4.4704 and TR > 1.6:
            TR = 1.6
    else:
        TR=1.6
    if TR>6.0:
        TR=6.0
    y_train.append(TR)
print(len(y_train))
print(x_train[4])
print(y_train[4])
#print([len(i) for i in x_train])

with open("/Git/dynamic-follow-tf/data/x_train", "w") as f:
    json.dump(x_train, f)

with open("/Git/dynamic-follow-tf/data/y_train", "w") as f:
    json.dump(y_train, f)

'''
x_train = [[max(i[0], 0), max(i[1], 0), max(i[2], 0)] for i in data_lead_split]
y_train = []
for i in d_data:
    if i[0] > 0:
        TR=i[1] / i[0]
    else:
        TR=1.4
    if TR>10:
        TR=10.
    y_train.append(TR)

with open("/Git/dynamic-follow-tf/data/x_train", "w") as f:
    json.dump(x_train, f)

with open("/Git/dynamic-follow-tf/data/y_train", "w") as f:
    json.dump(y_train, f)'''

'''
print(len(data_lead_split))

find_data = [{'vLead': 0.309262752532959, 'dRel': 45.880001068115234, 'realTR': 10.346703303025393, 'v_ego': 4.434262752532959}]

diffs_example = [[{'vLead': 0.309262752532959, 'dRel': 45.880001068115234, 'realTR': 10.346703303025393, 'v_ego': 4.434262752532959}, {...}], [{...}, {...}]]
diffs = []

for point in find_data:
    tmp_tmp = []
    for lead_section in data_lead_split:
        tmp = []
        for lead_point in lead_section:
            tmp_builder = {}
            tmp_builder["v_ego"] = abs(lead_point["v_ego"] - point["v_ego"])
            tmp_builder["dRel"] = abs(lead_point["dRel"] - point["dRel"])
            tmp_builder["vLead"] = abs(lead_point["vLead"] - point["vLead"])
            tmp.append(tmp_builder)
        tmp_tmp.append(tmp)
    diffs.append(tmp_tmp)

for lead_section in data_lead_split:
    for lead_point in lead_section:
        point_builder = []
        for point in find_data:
            tmp_builder = {}
            tmp_builder["v_ego"] = abs(lead_point["v_ego"] - point["v_ego"])
            tmp_builder["dRel"] = abs(lead_point["dRel"] - point["dRel"])
            tmp_builder["vLead"] = abs(lead_point["vLead"] - point["vLead"])
            print(point)
            break
        break
    break
'''

'''

for i in diffs[0]:
    print(i)
    print(len(i))
    print("\n\n")
    he=input()'''

'''
for section in data_lead_split:
    section_builder = []
    for point in section:
        point_builder = []
        for find_point in find_data:
            print(abs(point["v_ego"] - find_point["v_ego"]))
            print(abs(point["dRel"] - find_point["dRel"]))
            print(abs(point["vLead"] - find_point["vLead"]))
            if point["realTR"] is not None and find_point["realTR"] is not None:
                print(abs(point["realTR"] - find_point["realTR"]))
            else:
                print("None")
            
            print()
    diff_list.append()
'''

    
'''
labels = ["speed", "relative velocity", "lead acceleration"]
dataset = [[[40, -1.765, -0.5], [35, -1.765, -0.5], [34, -1.765, -0.5], [30, -1.765, -0.5]], [[40, 0, -0.5], [42, 1, -0.5], [45, 2, -0.5], [47, 3, -0.5]]]

sample = [[40, -1.765, -0.5], [35, -1.765, -0.5], [34, -1.765, -0.5], [30, -1.765, -0.5]]

sum_diffs = []

for times_section in dataset:
    diffs = [[] for i in range(len(sample))]
    for idx, values in enumerate(times_section):
        for idx2, value in enumerate(values):
            diffs[idx].append(abs(value-sample[idx][idx2]))
    diff_sum = 0
    for i in diffs:
        diff_sum += sum(i)
    sum_diffs.append(diff_sum)

print(sum_diffs)'''