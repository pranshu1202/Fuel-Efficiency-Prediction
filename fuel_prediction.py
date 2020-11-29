import pandas as pd
import numpy as np
from linear_regression import linear_regression 
from kmeans import kmeans

column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight','Acceleration', 'Model Year', 'Origin'] 
data = pd.read_csv('./Data/auto-mpg.data.csv',names=column_names,index_col=False).iloc[1:].values

input_data = data[:,1:] 
output_data = data[:,0] 

obj_linear_regression = linear_regression(input_data,output_data)

