import numpy as np

def softMax1d(z):
    z = np.exp(z - np.max(z))
    return z / np.sum(z)

def softMax2d(z):
    z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return z / np.sum(z, axis=1, keepdims=True)