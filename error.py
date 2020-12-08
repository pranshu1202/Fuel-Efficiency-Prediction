import pandas as pd
import numpy as np
import math


def rmse(prediction,target):
    return np.sqrt(np.sum(np.square(prediction-target)))

def mse(prediction,target):
    return np.sum(np.square(prediction-target))

def cross_entrophy(prediction,target):
    return (sum([prediction[i]*math.log2(target[i]) for i in range(len(prediction))])[0]/1000)