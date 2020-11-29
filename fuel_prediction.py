import pandas as pd
import numpy as np

column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight','Acceleration', 'Model Year', 'Origin'] 
data = pd.read_csv('./Data/auto-mpg.data.csv',names=column_names,index_col=False).iloc[1:].values

print(data.shape)