#!/usr/bin/env python
# coding: utf-8

# In[122]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from sklearn.preprocessing import StandardScaler
from my_util import load_dataset_csv


# In[123]:


#load my dataset
csv_file_path = 'data.csv'
X_train, X_test,Y_train, Y_test, classes = load_dataset_csv(csv_file_path) 


# In[124]:


m_train = X_train.shape[0]
m_test = X_test.shape[0]
num_columns = X_train.shape[1]
print("Number of training examples: " + str(m_train))
print("Number of testing examples: " + str(m_test))
print("Number of columns/features: " + str(num_columns))
print("X_train shape : " + str(X_train.shape))
print("Y_train shape : " + str(Y_train.shape))
print("X_test shape : " + str(X_test.shape))
print("Y_test shape : " + str(Y_test.shape))


# In[125]:


# Reshape Y_train and Y_test
Y_train_reshaped = Y_train.reshape(1, -1)
Y_test_reshaped = Y_test.reshape(1, -1)

# Transpose X_train and X_test
X_train_transposed = X_train.T
X_test_transposed = X_test.T

# Standardize the data separately for training and testing
scaler_train = StandardScaler()
X_train_scaled = scaler_train.fit_transform(X_train_transposed)

scaler_test = StandardScaler()
X_test_scaled = scaler_test.fit_transform(X_test_transposed)


# In[126]:


# Shapes after preprocessing
print("X_train_scaled shape:", X_train_scaled.shape)  # (31, 455)
print("Y_train_reshaped shape:", Y_train_reshaped.shape)  # (1, 455)

print("X_test_scaled shape:", X_test_scaled.shape)  # (31, 114)
print("Y_test_reshaped shape:", Y_test_reshaped.shape)  # (1, 114)


# In[127]:


#sigmoid function - shows how sigmoid calculation works
def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s


# In[137]:


#initializing weights and bias with zeros
def initialize_with_random(dim):
    w = np.random.randn(dim,1 ) * 0.01
    print("Shape of initialized w:", w.shape)
    b = 0
    
    assert(w.shape == (dim,1))
    assert(isinstance(b,float) or isinstance(b,int))
    return w,b


# In[138]:


def propagate(w, b, X, Y):
    
    m = X.shape[1]

##Forward Propagation 
#What it does: This is the process of making predictions by moving forward through the neural network, layer by layer.
#How it works: Input data is passed through each layer of the network, and computations are performed to generate an output. Each layer applies a set of weights to the input and applies an activation function. This continues until the final output is obtained.
   # print("Shape of w:", w.shape)
   # print("Shape of X:", X.shape)
    

    A = sigmoid(np.dot(w.T, X) + b)
    
    #print("Shape of A:", A.shape)
   # print("Shape of Y:", Y.shape)
    cost = np.sum(((-np.log(A))*Y + (-np.log(1-A)) * (1-Y)))/m


##Backward Propagation
#What it does: This is the process of updating the model's parameters (weights and biases) to improve its accuracy.
#How it works: After making predictions, the model compares its output to the actual target values (labels). The difference (error) is calculated, and then the algorithm works backward through the network to adjust the weights and biases, aiming to minimize the error.
 
    dw = (np.dot(X,(A-Y).T))/m
    db = (np.sum(A-Y))/m

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw" : dw,
            "db" : db}
    return grads,cost


# In[139]:


w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,3.],[4.,2.,-2]]), np.array([[1,0,1]])
grads, cost = propagate(w, b, X, Y)
print("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print("cost = " + str(cost))


# In[140]:


# Optimizing
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    
    costs = []
    
    for i in range(num_iterations):
        #calculates the cost and gradient 
        grads, cost = propagate(w, b, X, Y)
        
        dw = grads["dw"]
        db = grads["db"]
        
       
        #updates the parameters 
        w = w - (learning_rate*dw)
        b = b - (learning_rate*db)
        
        if i % 100 == 0 :
            costs.append(cost)
            
        if print_cost and i % 100 == 0:
            print("C0st after interation %i : %f " %(i,cost))
            
    params = {"w": w,
              "b": b}
    grads = {"dw": dw,
            "db": db}
    
    return params, grads, costs
            


# In[141]:


params, grads, costs = optimize(w, b, X, Y, num_iterations = 100, learning_rate = 0.009, print_cost = False)

print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))


# In[142]:


def predict ( w, b, X):
    
    m = X.shape[1]
   
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0],1)
    A = sigmoid(np.dot(w.T,X) + b)
    
    
    Y_prediction = (A >= 0.5) * 1.0
    
    assert (Y_prediction.shape == (1,m))
    return  Y_prediction


# In[143]:


w = np.array([[0.1124579],[0.23106775]])
b = -0.3
X = np.array([[1.,-1.1,-3.2],[1.2,2.,0.1]])
print ("predictions = " + str(predict(w, b, X)))


# In[154]:


def model(features_train, label_train, features_test, label_test, num_iterations=3000, learning_rate=0.5, print_cost=False):
    w, b = initialize_with_random(features_train.shape[0])

    params, grads, costs = optimize(w, b, features_train, label_train, num_iterations, learning_rate, print_cost=False)

    w = params["w"]
    b = params["b"]

    Y_prediction_test = predict(w, b, features_test)
    Y_prediction_train = predict(w, b, features_train)

    train_accuracy = np.mean(Y_prediction_train == label_train) * 100
    test_accuracy = np.mean(Y_prediction_test == label_test) * 100

    d = {
        "costs": costs,
        "Y_prediction_test": Y_prediction_test,
        "Y_prediction_train": Y_prediction_train,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy
    }

    return d


# In[155]:


d = model(X_train_scaled, Y_train_reshaped, X_test_scaled, Y_test_reshaped, num_iterations = 2000, learning_rate = 0.05, print_cost = False)

print (d)


# In[156]:


costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()


# In[ ]:





# In[ ]:




