import pandas as pd
import numpy as np

class error:
    def __init__(self):
        pass

    def rmse(self,prediction,target):
        return np.sqrt(np.sum(np.square(prediction-target)))

    def mse(self,prediction,target):
        return np.sum(np.square(prediction-target))