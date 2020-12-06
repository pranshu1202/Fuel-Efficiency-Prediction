import pandas as pd
import numpy as np
from error import mse
from error import rmse
NUM_FEATS = 7

class neural_network:
    def __init__(self,train_input,num_layers,num_units):
        weights = np.random.rand(NUM_FEATS,num_units)
        biases = np.random.rand(num_units)
        weights_output= np.random.rand(num_units ,1)
        biases_output = np.random.rand(1)
        self.weights=weights
        self.biases=biases
        self.weights_output=weights_output
        self.biases_output=biases_output
        self.m=0
        self.n=num_units
        self.activation_output=(0,0,0)
        self.input_data = train_input
        self.num_layers = num_layers
        self.num_units = num_units

        return 

    def __call__(self, X):
        def relu(a,b):
            return(np.fmax(a,b))
        W1 = self.weights
        b1 = self.biases
        W2 = self.weights_output
        b2 = self.biases_output
        b = np.zeros((1,self.n))
        z1 = np.dot(X,W1)+b1
        a1 = relu(b,z1)
        y_ = np.dot(a1,W2)+b2
        self.activation_output=(a1,y_)
        m=X.shape[0]
        n=X.shape[1]
        self.m=m
        return y_

    def backward(self, X, y, lamda):
        W1 = self.weights
        b1 = self.biases
        W2 = self.weights_output
        b2 = self.biases_output
        m=X.shape[0]
        a1,y_=self.activation_output
        def relu_derivative(z):
            m=z.shape[0]
            n=z.shape[1]
            a=np.zeros((m,n))
            for i in range(0,m-1):
                for j in range(0,n-1):
                    if z[i][j] > 0:
                        a[i][j]=1
                    else:
                        a[i][j]=0
            return a
        delta2 =  y_-y
        deltaW2 =  (1.0/m)*np.dot(np.transpose(a1),delta2)+(lamda/m)*W2
        deltab2 = (1.0/m) * np.sum(delta2, axis=0, keepdims=True)
        delta1 =np.dot(delta2,np.transpose(W2))*(relu_derivative(a1))
        deltaW1 = (1.0/m)*np.dot(np.transpose(X),delta1)+(lamda/m)*W1
        deltab1 = (1.0/m) * np.sum(delta1, axis=0, keepdims=True)
        del_W=[]
        del_b=[]
        del_W.append(deltaW1)
        del_W.append(deltaW2)
        del_b.append(deltab1)
        del_b.append(deltab2)
        return del_W,del_b

class optimizer:
    def __init__(self, nn_learning_rate):
        self.learning_rate=nn_learning_rate
        return
    def step(self, weights, biases, delta_weights, delta_biases):
        W1=weights[0]
        W2=weights[1]
        b1=biases[0]
        b2=biases[1]
        dW1= delta_weights[0]
        dW2=delta_weights[1]
        db1=delta_biases[0]
        db2=delta_biases[1]
        learning_rate=self.learning_rate
        W1 = W1 - learning_rate * dW1
        b1 = b1 - learning_rate * db1
        W2 = W2 - learning_rate * dW2
        b2 = b2 - learning_rate * db2
        Weights=[]
        Weights.append(W1)
        Weights.append(W2)
        Biases=[]
        Biases.append(b1)
        Biases.append(b2)
        return Weights,Biases

def loss_regularization(weights, biases):
    W1=weights[0]
    W2=weights[1]
    b1=biases[0]
    b2=biases[1]
    L2_regularization_cost = (np.sum(np.square(W1)+b1) + np.sum(np.square(W2)+b2))
    return L2_regularization_cost

def loss_fn(y, y_hat, weights, biases, lamda):
    m=y.shape[0]
    loss = mse(y,y_hat)+(lamda)*loss_regularization(weights,biases)
    return loss

def train(net, optimizer, lamda, batch_size, max_epochs,train_input, train_target):
    net(train_input)
    m=train_input.shape[0]
    train_target=np.array(train_target)
    train_target=train_target.reshape(m,1)
    W1 = net.weights
    b1 = net.biases
    W2 = net.weights_output
    b2 = net.biases_output
    a1,y_=net.activation_output
    w=[]
    b=[]
    w.append(W1)
    w.append(W2)
    b.append(b1)
    b.append(b2)
    costs_train= []
    costs_dev= []
    for i in range(0, 200*max_epochs):
        for j in range(0,m-batch_size-1,batch_size):
            W1 = net.weights
            b1 = net.biases
            W2 = net.weights_output
            b2 = net.biases_output 
            w=[]
            b=[]
            w.append(W1)
            w.append(W2)
            b.append(b1)
            b.append(b2)
            input1=train_input[j:j+batch_size]
            output1=train_target[j:j+batch_size]
            net(input1)
            a1,y_=net.activation_output
            if y_.shape != (128,1):
                y_out=y_[j:j+batch_size]
            else:
                y_out=y_
            cost = loss_fn(output1,y_out,w,b,lamda)/5
            costs_train.append(cost)
            del_W,del_b = net.backward(input1,output1, lamda)
            wt,bias = optimizer.step(w,b,del_W,del_b)
            wt[0]=np.array(wt[0])
            wt[1]=np.array(wt[1])
            bias[0]=np.array(bias[0])
            bias[1]=np.array(bias[1])
            net.weights=wt[0]
            net.weights_output=wt[1]
            net.biases=bias[0]
            net.biases_output=bias[1]
            #skip this for Part 1b
        if (i+1)%100==0:
            print("Iteration %i: Train cost :-  %f"  %(i+1, cost))
       
    return

