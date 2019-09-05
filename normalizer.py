import numpy as np
scale = [0, 1] # wider scale might improve accuracy
def norm(data, min_max=[]):
    if min_max==[]:
        d_min = min(data)
        d_max = max(data)
        
        x = [d_min, d_max]
        return [np.interp(i, x, scale) for i in data], [d_min, d_max]
        #return [(i - d_min) / (d_max - d_min) for i in data], [d_min, d_max]
    else:
        y = [min_max[0], min_max[1]]
        return np.interp(data, y, scale)
        #return (data - min_max[0]) / (min_max[1] - min_max[0])

def get_3d_min_max(data, idx):
    #all_vals = [j for i in [[h[idx] for h in p] for p in data] for j in i]
    all_vals = [i[idx] for i in data]
    all_vals = [j for i in all_vals for j in i]  #get all numbers from 3d array
    
    return [min(all_vals), max(all_vals)]

'''def denorm(data, min_max):
    return [(min_max[0] - i) * (min_max[0] - min_max[1]) for i in data]'''