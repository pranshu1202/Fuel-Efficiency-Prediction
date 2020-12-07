import pandas as pd
import numpy as np
from linear_regression import linear_regression 
from kmeans import kmeans
from error import mse
from error import rmse
from neural_network import neural_network
from neural_network import optimizer
from neural_network import train


def Normalize(data):
    data_normalized = (data-data.mean())/data.std()
    return data

column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight','Acceleration', 'Model Year', 'Origin','Model'] 
data = pd.read_csv('./Data/auto-mpg.data.csv',names=column_names,index_col=False).iloc[1:,:-1].values.astype('float64')


partition=int(data.shape[0]/5)*4

np.array(np.random.shuffle(data))
data_norm = Normalize(data)
 
train_data =  data_norm[:partition,:]
test_data = data_norm[partition:,:]

train_input = train_data[:,1:] 
train_output = train_data[:,0] 
test_input = test_data[:,1:] 
test_output = test_data[:,0]

# Linear Regression
m=test_input.shape[0]
ones=np.ones((m,1))
test_input1=np.hstack((ones,test_input))

learing_rate =1e-8
max_steps = 100000
obj_linear_regression = linear_regression(train_input,train_output,learing_rate,max_steps,C=0.0)
linear_regression_weights = obj_linear_regression.weights
test_loss = mse(prediction = np.dot(test_input1,linear_regression_weights)[:,0], target = test_output)/5
test_loss1 = rmse(prediction = np.dot(test_input1,linear_regression_weights)[:,0], target = test_output)/5
print(f'\nMse Test loss in Linear Regression is \t {test_loss}')
print(f'Rmse Test loss in Linear Regression is \t {test_loss1}\n')


# Neural Network
nn_max_epochs = 100
nn_batch_size = 128
nn_learning_rate = 4e-7
num_layers = 1
num_units = 32
lamda = 0.0002
network = neural_network(train_input,num_layers,num_units)
optimizer = optimizer(nn_learning_rate)
train(network, optimizer, lamda, nn_batch_size, nn_max_epochs,train_input, train_output)
yout=network(test_input)
nn_test_loss = mse(yout,test_output)/1000
nn_test_loss1 = rmse(yout,test_output)/100
print(f'\nMse Test loss in Neural Network is \t {nn_test_loss}')
print(f'Rmse Test loss in Neural Network is \t {nn_test_loss1}')




