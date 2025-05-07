import numpy as np
def load_file(name):
    with open(name,"rb") as f:
        return np.load(f)
def save_file(name,array):
    with open(name,"wb") as f:
        return np.save(f,array)