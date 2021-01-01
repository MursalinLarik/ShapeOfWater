#!/usr/bin/env python
# coding: utf-8

# ## Logistic Regression with a Neural Network for GPR Radargarms

# ### Import Packages

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
import os
import cv2
import random
import pickle
#from lr_utils import load_dataset

get_ipython().run_line_magic('matplotlib', 'inline')


# ### Creating Dataset

# #### Define class names

# In[2]:


DATADIR = r"C:\Users\narji\Documents\Semester 8\Logistic_reg"


# In[3]:


CATEGORIES = ["no_pipe", "pipe"]


# In[4]:


classes = np.asarray(CATEGORIES)


# #### Create Training Data

# In[5]:


training_data = []


# Creating a function that reads images from DATADIR and creates a 2d list, coupling image with class label

# In[6]:


def create_training_data():
    for category in CATEGORIES :
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try :
                img_array = cv2.imread(os.path.join(path, img))#, cv2.IMREAD_GRAYSCALE)
                #new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([img_array, class_num])
            except Exception as e:
                pass


# In[7]:


create_training_data()


# In[8]:


random.shuffle(training_data)


# In[9]:


train_set_x_orig = [] #features
train_set_y= [] #labels


# We added "_orig" at the end of image datasets (train and test) because we are going to preprocess them. After preprocessing, we will end up with train_set_x

# In[10]:


for features, label in training_data:
    train_set_x_orig.append(features)
    train_set_y.append(label)


# In[11]:


train_set_x_orig = np.asarray(train_set_x_orig)


# In[12]:


train_set_x_orig = train_set_x_orig.reshape(-1, 128, 128, 3)


# In[13]:


train_set_y = np.asarray(train_set_y).reshape(-1, 1)
train_set_y = train_set_y.T


# Each line of your train_set_x_orig  is an array representing an image. You can visualize an example by:

# In[14]:


# Example of a picture
index = 489
plt.imshow(train_set_x_orig[index])
print ("y = " + str(train_set_y[:, index]) + ", it's a " + classes[np.squeeze(train_set_y[:, index])] + " picture") #.decode("utf-8") +  "' picture.")


# #### Creating Test Dataset

# In[15]:


TEST_CATEGORIES = ["no_pipe_test", "pipe_test"]


# In[16]:


testing_data = []

def create_testing_data():
    for category in TEST_CATEGORIES :
        path = os.path.join(DATADIR, category)
        class_num = TEST_CATEGORIES.index(category)
        for img in os.listdir(path):
            try :
                img_array = cv2.imread(os.path.join(path, img))#, cv2.IMREAD_GRAYSCALE)
                #new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                testing_data.append([img_array, class_num])
            except Exception as e:
                pass


# In[17]:


create_testing_data()


# In[18]:


random.shuffle(testing_data)


# In[19]:


test_set_x_orig = [] #features
test_set_y= [] #labels


# In[20]:


for features, label in testing_data:
    test_set_x_orig.append(features)
    test_set_y.append(label)


# In[21]:


test_set_x_orig = np.asarray(test_set_x_orig)
test_set_x_orig = test_set_x_orig.reshape(-1, 128, 128, 3)

test_set_y = np.asarray(test_set_y).reshape(-1, 1)
test_set_y = test_set_y.T


# #### Dataset details

# In[22]:


m_train = train_set_x_orig.shape[0]
m_test =  test_set_x_orig.shape[0]
num_px = test_set_x_orig.shape[1]

print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))


# #### Pre-processing on train and test set

# In[23]:


# Reshape the training and test examples

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))


# In[24]:


## Standardize the training and test examples

train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.


# ### General Architecture of the learning algorithm
# 
# We are building a Logistic Regression, using a Neural Network. The following Figure explains why Logistic Regression is actually a very simple Neural Network!
# 
# <img src="LR_pipe.png" style="width:500px;height:330px;">
# 
# **Mathematical expressions:**:
# 
# For one example $x^{(i)}$:
# 
# ($ i $ is the training example number, and 
# $ a^{(i)} $ is the predicted class for $i$th example)
# 
# $$z^{(i)} = w^T x^{(i)} + b \tag{1}$$
# $$\hat{y}^{(i)} = a^{(i)} = sigmoid(z^{(i)})\tag{2}$$ 
# 
# Loss function:
# $$ \mathcal{L}(a^{(i)}, y^{(i)}) =  - y^{(i)}  \log(a^{(i)}) - (1-y^{(i)} )  \log(1-a^{(i)})\tag{3}$$
# 
# The cost is then computed by summing over all training examples:
# $$ J = \frac{1}{m} \sum_{i=1}^m \mathcal{L}(a^{(i)}, y^{(i)})\tag{6}$$

# ### Building the parts of our algorithm

# #### Sigmoid function

# In[25]:


def sigmoid(z):
    s = 1/(1 + np.exp(-z))
    return s


# #### Initializing parameters

# In[26]:


def initialize_with_zeros(dim):
    w = np.zeros([dim,1], dtype=int)
    b = 0
    
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b


# #### Forward and Backward propagation
# 
# Forward Propagation:
# - We get X
# - We compute $A = \sigma(w^T X + b) = (a^{(1)}, a^{(2)}, ..., a^{(m-1)}, a^{(m)})$ i.e. predict for all training examples
# - Calculate the cost function: $J = -\frac{1}{m}\sum_{i=1}^{m}y^{(i)}\log(a^{(i)})+(1-y^{(i)})\log(1-a^{(i)})$
# 
# For gradient descen, we use:
# 
# $$ \frac{\partial J}{\partial w} = \frac{1}{m}X(A-Y)^T\tag{7}$$
# $$ \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (a^{(i)}-y^{(i)})\tag{8}$$

# In[27]:


def propagate(w, b, X, Y): #Implement the cost function and its gradient for the propagation explained above

    m = X.shape[1]
    
    # FORWARD PROPAGATION (FROM X TO COST)
    z = np.dot(w.T, X) + b
    A = sigmoid(z)  # compute activation
    cost = (-1) * (1/m) * np.sum((Y * np.log(A)) + ((1-Y) * (np.log(1-A)))) # compute cost
    
    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = (1/m) * np.dot(X, (A-Y).T)
    db = (1/m) * np.sum(A - Y)

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost #This function returns:
                       #cost -- negative log-likelihood cost for logistic regression
                       #dw -- gradient of the loss with respect to w, thus same shape as w
                       #db -- gradient of the loss with respect to b, thus same shape as b


# #### Optimization function

# In[28]:


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
#This function optimizes w and b by running a gradient descent algorithm
    costs = []
    cost_per_it = []
    for i in range(num_iterations):
        
        
        # Cost and gradient calculation
        grads, cost =  propagate(w, b, X, Y)
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update rule
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 10 training iterations
        if print_cost and i % 10 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
            cost_per_it.append(cost)
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs, cost_per_it
    #This function returns:
    #params -- dictionary containing the weights w and bias b
    #grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    #costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.


# #### Prediction function
# 
# It calculates
# 
# $\hat{Y} = A = \sigma(w^T X + b)$
# 
# and convert the entries of a into 0 (if activation <= 0.5) or 1 (if activation > 0.5).

# In[29]:


def predict(w, b, X): #Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    z = np.dot(w.T, X) + b
    A = sigmoid(z)
    
    for i in range(A.shape[1]):
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if(A[0][i] <= 0.5):
            Y_prediction[0][i] = 0
        else:
            Y_prediction[0][i] = 1
    
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction


# ### Create Model
# 
# Building the logistic regression model by merging and calling the functions we've implemented previously

# In[30]:


# GRADED FUNCTION: model

def model(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate, print_cost = True):
        
    # initialize parameters with zeros
    w, b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent
    parameters, grads, costs, cost_per_it =  optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost = True)
    
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    
    # Predict test/train set examples
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"cost_per_it": cost_per_it,
         "costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d


# In[31]:


my_model = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 500, learning_rate = 0.000625, print_cost = True)


# In[32]:


# Example of a picture that was wrongly classified.
index = 136
plt.imshow(test_set_x[:,index].reshape((num_px, num_px, 3)))
#print ("y = " + str(train_set_y[:, index]) + ", it's a " + classes[np.squeeze(train_set_y[:, index])] + " picture") #.decode("utf-8") +  "' picture.")
print ("y = " + str(test_set_y[0,index]) )#+ ", you predicted that it is a " + classes[d["Y_prediction_test"][0,index]] +  "picture")


# In[33]:


# Plot learning curve (with costs)
costs = np.squeeze(my_model['cost_per_it'])
My_list = [*range(0, my_model['num_iterations'], 10)]
#print(costs)
#print(My_list)
plt.plot(My_list, costs)
plt.ylabel('cost')
plt.xlabel('iterations')
plt.title("Learning rate =" + str(my_model["learning_rate"]))
plt.show()

