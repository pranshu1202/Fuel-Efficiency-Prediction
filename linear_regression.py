import pandas as pd
import numpy as np
from error import mse
from error import rmse


class linear_regression:
    def __init__(self,input_data,output_data,learning_rate,max_steps,C):
        def initialize(n):
            return np.random.uniform(0,0.01,(n,1)) 

        def gradient(input_data,weights,output_data,C=0.0):
            m,n=self.input_data.shape
            yout = np.dot(self.input_data,weights)
            grad1 = np.dot(self.input_data.T,(yout-self.output_data)) 
            grad2 = 2*weights*C 
            grad=grad1+grad2
            return (2*grad)/m
        
        def weights_modify(weights,gradient,learning_rate):
            weights=weights-learning_rate*gradient
            return weights

        def early(train_loss0,train_loss):
            if abs(train_loss-train_loss0)<0.5:
                return True
            return False

        def gradient_descent():
            n = self.input_data.shape[1]
            weights = initialize(n)
            train_loss = mse(prediction = np.dot(self.input_data,weights), target= self.output_data)
            train_loss0=100.0
            print("step {} \t train loss: {}".format(0,train_loss))
            for iteration in range(1,self.max_steps+1):
            
                gradients = gradient(self.input_data, weights, self.output_data, self.C)
                weights = weights_modify(weights, gradients, self.learning_rate)

                if iteration%10 == 0:
                    train_loss = mse(prediction = np.dot(self.input_data,weights), target = self.output_data)
                    print("step {} \t train loss: {}".format(iteration,train_loss))
                    if early(train_loss0,train_loss) == True :
                        print('Stopping Early at step: {}'.format(iteration))
                        break
                    train_loss0=train_loss
            return weights

        m=input_data.shape[0]
        ones=np.ones((m,1))
        input_data=np.hstack((ones,input_data))
        self.input_data = input_data
        self.output_data = output_data
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.C=C
        self.weights = gradient_descent()
        
    