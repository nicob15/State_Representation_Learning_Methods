import numpy as np
import torch
try:
    import cPickle as pickle
except:
    import pickle

def save_pickle(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def load_pickle(file):
    if not file[-3:] == 'pkl' and not file[-3:] == 'kle':
        file = file+'pkl'

    with open(file, 'rb') as f:
        data = pickle.load(f)

    return data

def logvar2var(log_var):
    return torch.clip(torch.exp(log_var), min=1e-5)

def add_gaussian_noise(data, noise_level=0.0, clip=False, clip_level=(0, 1)):
    if clip:
        return (data + np.random.normal(0.0, noise_level, size=data.shape)).clip(clip_level[0], clip_level[1])
    else:
        return data + np.random.normal(0.0, noise_level, size=data.shape)
