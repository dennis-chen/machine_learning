# -*- coding: utf-8 -*-
"""
Created on Thu May  1 13:29:50 2014

@author: dchen

simple single variable linear regression
using the tutorial and data from : http://aimotion.blogspot.com/2011/10/machine-learning-with-python-linear.html
which in turn apparently draws from Andrew Ng's fantastic ML coursera course

"""

from numpy import *
from pylab import *
import matplotlib


def load_and_plot(file_name):
    """loads and plots the housing data information provided by Andrew Ng's coursera course"""
    data = loadtxt(file_name, delimiter=',') 
    #Plot the data
    scatter(data[:, 0], data[:, 1], marker='o', c='b')
    title('Profits distribution')
    xlabel('Population of City in 10,000s')
    ylabel('Profit in $10,000s')
    show()
    return data

def calculate_cost(theta_values, data):
    """given theta0 and theta1 values of size 1 by 2 and a dataset, h(theta_values) = theta0 + theta1*x1,
    computes the squared error cost function with the given housing dataset"""
    population = data[:,0]
    prices = data[:,1]
    total_error = 0
    for i in range(0,len(population)):
        x = array([[1],[population[i]]])
        hypothesis = theta_values.dot(x).flatten() 
        squared_error = (hypothesis - prices[i])**2
        total_error += squared_error
    return .5*total_error/len(population) #division by m is just a scaling factor since we're only interested in whether this function is minimized

def run_gradient_descent(data,theta,alpha,num_iters):
    """perform gradient descent for specified
    number of iterations, theta is a 1 by 2 row vector"""
    population = data[:,0]
    prices = data[:,1]
    x = ones(shape=(len(population),2)) #add ones for theta0 
    x[:,1] = population
    x = transpose(x)
    error_history = zeros(shape=(num_iters,1))
    
    for i in range(num_iters):
        predictions = theta.dot(x)
        errors_x1 = (predictions - prices) * x[0,:]
        errors_x2 = (predictions - prices) * x[1,:]
        theta[0][0] = theta[0][0] - alpha*(1.0/len(population))*errors_x1.sum()
        theta[0][1] = theta[0][1] - alpha*(1.0/len(population))*errors_x2.sum()
        error_history[i,0] = calculate_cost(theta,data)
        
    return theta, error_history
    
    
data = load_and_plot('ex1data1.txt')

theta_init = zeros(shape=(1,2))

iterations = 1500
alpha = 0.01

print "cost function for initial theta vector: " + str(calculate_cost(theta_init,data))

theta,error_history = run_gradient_descent(data,theta_init,alpha,iterations)

print "theta values predicted by gradient descent: "+ str(theta)

#plot best fit line calculated by gradient descent
x = ones(shape=(len(data[:,0]),2)) #add ones for theta0 
x[:,1] = data[:,0]
result = theta.dot(x.T).flatten()
plot(data[:, 0], result)
show()



        
    
        