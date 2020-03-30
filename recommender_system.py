# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 21:25:38 2020

@author: kv83821
"""
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

movies = pd.read_csv('ml-1m/movies.dat',sep = '::',header = None, engine = 'python' , encoding = 'latin-1')
#separator = :: since movie names are separated by ::
#header = None since no header present
#encoding = latin-1 since movie name contain some special characters not given in utf-8
users = pd.read_csv('ml-1m/users.dat',sep = '::',header = None, engine = 'python' , encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat',sep = '::',header = None, engine = 'python' , encoding = 'latin-1')

training_set = pd.read_csv('ml-100k/u1.base',delimiter = '\t')
#values are separated by a tab hence \t delimiter
#u1.base consist of 80000 rows 
training_set = np.array(training_set, dtype = 'int')
#pytorch tensor works on array rather than dataframes thus we convert dtaframe to numpy array
test_set = pd.read_csv('ml-100k/u1.test',delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

#get the total number of users and the movies 
#this will be heplful when we make the matrix of users as row and movie as column and ratings cell values
#column 0 corresponds to users and 1 corresponds to movies
t_users = int(max(max(training_set[:,0]),max(test_set[:,0])))
t_movies = int(max(max(training_set[:,1]),max(test_set[:,1])))

#Conversion of data
def convert(data):
    actual_data = []
    for id_users in range(1,t_users+1):
        #select concerned users movies and its corresponding ratings
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(t_movies)
        #Since index atarts with 1 we need to subtract 1
        ratings[id_movies-1] = id_ratings
        actual_data.append(list(ratings))
    return actual_data

training_set = convert(training_set)
test_set = convert(test_set)

# Convert into torch tensors(Array containing single data types)
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

#convert the rating into binary form
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >=3] = 1
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >=3] = 1

#defining the class for the Reduced Boltzmann Machine to define its behavior

class RBM():
    def __init__(self, nh, nv):
        #nh corresponds to the hidden nodes
        #nv to the visible nodes
        self.W = torch.randn(nh, nv)
        #we require 2 dimensions in torch tensor since the function we are going to use 
        #do not accept 1d vectors but 2d torch tensors
        #First dimension corresponds to the batch size not the actual size but defining one dimension
        #Second to the number of nodes in given layer
        #randn initialises with a normal distribution i.e. mean 0 and variance 1
        self.bias_hidden = torch.randn(1, nh)
        self.bias_visible = torch.randn(1, nv)
       
    def sample_h(self, x):
        #x corresponds to the visible nodes in the probability h given v
        #probability of h given v is sigmoid activation 
        #for gibbs sampling werequire this
        #we use gibbs sampling for log likelihood gradient
        wx = torch.mm(x,self.W.t())
        #mm for multiply
        #we take transpose of W since W was defined as wt. from hidden to visible
        #the weigths will be same for visible to hidden but shape will be transpose of W
        activation = wx + self.bias_hidden.expand_as(wx)
        #this is input to the sigmoid function
        #expand_as used so that wx and bias_hidden have same dimension
        #since we will not have one user but more in a single batch so we will expand bias vector
        #according to the wx which contains all users in the given batch
        prob_h_given_v = torch.sigmoid(activation)
        #prob_h_given_v is the probability that hidden node lights up given the visible nodes
        #lets say h is the drama movie genre node
        #all the visible nodes are drama movies
        #thus the probability of h to light up is very high
        return prob_h_given_v,torch.bernoulli(prob_h_given_v)
        # bernoulli converts the probabilities into into samples of 0 or 1 
        #prob_h_given_v consist of all prob. of hidden nodes given the visible nodes
    def sample_v(self, y):
        wy = torch.mm(y,self.W)
        #note here we do not have transpose of W
        activation = wy + self.bias_visible.expand_as(wy)
        prob_v_given_h = torch.sigmoid(activation)
        return prob_v_given_h,torch.bernoulli(prob_v_given_h)
    def train(self,v0, vk, ph0, phk):
        #v0 corresponding to the input vector
        #vk corresponding to the visible layer after after k steps
        #ph0 prob. h given v0
        #phk prob. h given vk
        self.W += torch.mm(ph0.t() , v0) - torch.mm(phk.t(), vk)
        #change in weight = alpha(<v0h0> - <vkhk>) is what we learned computed from log likelihood gradient
        self.bias_visible = torch.sum((v0 - vk), 0)
        #change in visible layer bias = v0 - vk
        #0 is the axis
        self.bias_hidden = torch.sum((ph0 - phk), 0)
        #change in hidden layer bias = ph0 - phk
        
nv = len(training_set[0])
nh = 100
batch_size = 100
#Now we create object of our class
rbm = RBM(nh, nv)      

epochs = 10
for epoch in range(1,epochs + 1):
    train_loss = 0 #this is the training loss
    s = 0. #this is for normalising the train_loss
    for id_users in range(0, t_users-batch_size, batch_size):
        #the third arguement in the range is the stepping size
        #this for loop is for all the users since the class is defined for one user
        #so we need to loop for all the users
        vk = training_set[id_users:id_users + batch_size]
        v0 = training_set[id_users:id_users + batch_size]
        ph0,_ = rbm.sample_h(v0)
        for k in range(10):
        #k contrastive divergence
            _,hk = rbm.sample_h(vk)
            #actually we input v0 which is equal to vk in the first case
            #subsequently vk changes ,we compare v0 and vk
            _,vk = rbm.sample_v(hk)
            #updating vk
            vk[v0<0] = v0[v0<0]
            #do not update where rating is -1 or the user has not seen the movie
            #for training we do not require this
        phk,_ = rbm.sample_h(vk)
        #we need phk in training i.e. updating the weights
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0]))
        #v0 >= 0 is because we trained the for grater equal to 0 and not for -1
        s += 1.
        #update the counter for normalisation
    print("epoch :" + str(epoch) + "loss :" + str(train_loss/s))
    #print the epoch numner and normalised training loss
        
#testing the model


test_loss = 0 #this is the training loss
s = 0. #this is for normalising the train_loss
for id_users in range(t_users):
    #we need to loop for all the users
    v = training_set[id_users:id_users + 1]
    vt = test_set[id_users:id_users + 1]
    if len(vt[vt >= 0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt >= 0] - v[vt >= 0]))
        s += 1.
    #update the counter for normalisation
print("loss :" + str(test_loss/s))
#print the epoch numner and normalised training loss
        
        
 
        
        
        
        
        
        
        
        
        
        
        
        
        




















