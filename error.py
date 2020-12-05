import pandas as pd
import numpy as np



def rmse(prediction,target):
    return np.sqrt(np.sum(np.square(prediction-target)))

def mse(prediction,target):
    return np.sum(np.square(prediction-target))