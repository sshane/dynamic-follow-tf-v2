import ast
import matplotlib.pyplot as plt

with open('D:/Resilio Sync/dfv2/ShaneSmiskol-TOYOTA COROLLA 2017/df-data.29', 'r') as f:
    data = []
    for i in f.read().split('\n'):
        try:
            data.append(ast.literal_eval(i))
        except:
            pass
keys, data = data[0], data[1:]
data = [i for i in data if i[-1] is not None]

plt.plot(range(len(data)), [i[-2] for i in data], label='new_accel')  # late by 5 samples?
plt.plot(range(len(data)), [i[1] for i in data])
plt.legend()
plt.show()
