import numpy as np
import functools
import operator


def normX(data, data_scale=None, scale=[0, 1]):
    if data_scale is None:
        # v_ego = [i[0] for i in data]
        v_ego = [i['v_ego'] for i in data]
        a_ego = [i['a_ego'] for i in data]
        
        scales = {'v_ego_scale': [min(v_ego), max(v_ego)],
                  'a_ego_scale': [min(a_ego), max(a_ego)]}

        normalized = [[np.interp(i['v_ego'], scales['v_ego_scale'], [0, 1]), np.interp(i['a_ego'], scales['a_ego_scale'], [0, 1])] for i in data]
        return normalized, scales
    else:
        y = [data_scale[0], data_scale[1]]
        return np.interp(data, y, scale)
