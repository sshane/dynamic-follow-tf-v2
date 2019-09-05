def norm(data, min_max=[]):
    if min_max==[]:
        d_min = min(data)
        d_max = max(data)
        return [(i - d_min) / (d_max - d_min) for i in data], [d_min, d_max]
    else:
        return (data - min_max[0]) / (min_max[1] - min_max[0])

def get_min_max(data, idx):
    all_vals = [j for i in [[h[idx] for h in p] for p in data] for j in i]
    #all_vals = [j for i in data for j in i]
    return [min(all_vals), max(all_vals)]

'''def denorm(data, min_max):
    return [(min_max[0] - i) * (min_max[0] - min_max[1]) for i in data]'''