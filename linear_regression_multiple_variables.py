# -*- coding: utf-8 -*-
"""
Created on Thu May  8 16:19:38 2014

@author: dchen
"""
from numpy import *
from pylab import *
import matplotlib

def normalize_feature(feature_vector):
    """scale a feature to be in the -1 to 1 range and 
    make sure the mean is 0. returns a new feature vector"""
    original_range = max(feature_vector) - min(feature_vector)
    average = feature_vector.sum()/len(feature_vector)
    normalized_features = (feature_vector - average)*1.0/original_range
    return normalized_features
    
def normalize_dataset(dataset):
    """takes a matrix where each column is the same feature
    for lots of different training examples
    and normalizes values, returning a normalized matrix for faster
    convergence with gradient descent"""
    m = dataset.shape[1] #number of training examples
    copied_dataset = dataset.astype('float_')
    for i in range(m):
        copied_dataset[:,i] = normalize_feature(dataset[:,i])
    return copied_dataset
            
def normalize_feature_test():
    feature_vec = array([[1],[2],[3]])    
    new_vec = normalize_feature(feature_vec)
    print "original vector: "+str(feature_vec)
    print "normalized vector: "+str(new_vec)
    print "expected: [[-0.5]   [0]   [0.5]]"

def normalize_dataset_test():
    feature_array = array([[1,2,4],[2,4,8],[3,6,12]])
    new_features = normalize_dataset(feature_array)
    print "normalized dataset: "+str(new_features)

def calculate_cost(thetas,x,y):
    """returns cost function or error of the hypothesis. theta is a column
    vector and x stores training examples in rows"""
    m = x.shape[1] #number of training examples    
    errors_squared = (x.dot(thetas) - y)**2
    cost = 1/(2*m)*errors_squared.sum()
    return cost
    
def run_gradient_descent(x,y,initial_theta,alpha,num_iters):    
    error_history = zeros(shape=(num_iters,1))
    for i in range(num_iters):
        predictions = x.dot(theta)
        errors = predictions - y
        errors_times_x = []
        feature_num = x.shape[1]
        training_example_num = x.shape[0]
        for j in range(feature_num):
            errors_x = errors * x[i,:]
            errors_times_x.append(errors_x)
        for k in range(feature_num):
            theta[0][i] = theta[0][i] - alpha*(1.0/len(population))*errors_x1.sum()
        if error_history[i-1] - error < 1e-12: #if error is barely changing it has been minimized and the loop can stop
            break
        error = calculate_cost(thetas,x,y)
        error_history[i,0] = error 
        
    return theta, error_history
        
    
def run_multiple_lin_reg():
    """run the whole shebang"""
    alpha = 0.01
    num_iters = 1500
    data = loadtxt('ex1data2.txt', delimiter=',') 
    x = ones(shape=(len(data[:,1]),3))
    x[:,1] = data[:,0]
    x[:,2] = data[:,1]
    norm_x = normalize_dataset(x)
    y = data[:,2]
    initial_theta = [[0],[0]]
    run_gradient_descent(norm_x,y,initial_theta,alpha,num_iters)
    
    