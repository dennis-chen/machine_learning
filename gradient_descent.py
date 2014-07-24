import csv
import numpy as np
import math
from matplotlib import pyplot as plt

class GDSolver(object):
    """works with linear and logistic regression. hypothesis
    func is used to calculate the update rule while cost func is
    used to evaluate convergence"""

    def __init__(self,error_func,hypothesis_func,training_examples,learning_rate):
        self.error_func = error_func
        self.hypothesis_func = hypothesis_func
#training examples are formatted as an array, where the first row of each column is the label, and the rest of the column is the feature vector
        self.training_examples = training_examples
        self.error_history = None
        self.learned_theta = None
        self.last_num_iterations_run = None
        self.learning_rate = learning_rate

    def run_gd(self,init_theta,iterations,convergence_bound):
        """theta is a column vector"""
        theta = init_theta
        error_history = []
        for i in xrange(iterations):
            print "iteration %s/%s"%(i+1,iterations)
            error = self._calculate_error(theta)
            error_history.append(error)
            if abs(error) < convergence_bound:
                break
            theta = theta - self.learning_rate*self._calculate_update_deriv(theta)
        self.learned_theta = theta
        self.error_history = error_history
        self.last_num_iterations_run = iterations
        return theta

    def _calculate_update_deriv(self,theta):
        """returns column vector that describes update"""
        labels = self.training_examples[0,:]
        feature_vectors = self.training_examples[1:,:]
        num_examples = labels.size
        #include an array of bias terms in the training set (a vector of ones)
        feature_vectors = np.vstack((np.ones(num_examples),feature_vectors))
        num_features = feature_vectors.shape[0] # in reality, num_features + 1 because of the bias term
        update_sum = np.zeros((num_features,1))
        for i in xrange(num_examples):
            y = labels[i]
            x = feature_vectors[:,i:i+1]
            hypothesis = self.hypothesis_func(x,theta)
            update = (hypothesis - y)*x
            update_sum += update
        update_sum = update_sum/num_examples
        return update_sum

    def _calculate_error(self,theta):
        """calculates summed error function for given parameters theta"""
        labels = self.training_examples[0,:]
        feature_vectors = self.training_examples[1:,:]
        num_examples = labels.size
        feature_vectors = np.vstack((np.ones(num_examples),feature_vectors))
        total_error = 0
        for i in xrange(num_examples):
            y = labels[i]
            x = feature_vectors[:,i]
            hypothesis = self.hypothesis_func(x,theta)
            total_error += self.error_func(hypothesis,y)
        return total_error

    def plot_error_history(self):
        """shows history of error of gradient descent algorithm
        in the last job that was run"""
        iterations = self.last_num_iterations_run
        iteration_array = np.linspace(1,iterations,iterations)
        plt.plot(iteration_array,self.error_history)
        plt.show()

def linear_regression_hyp(features,theta):
    """returns scalar output that results from using theta as coefficients
    and features as 'x' values in an equation"""
    return np.asscalar(np.dot(np.transpose(theta),features))

def linear_regression_error(hypothesis,y):
    """returns squared error"""
    return (hypothesis - y)**2

def load_csv_file(path):
    return np.genfromtxt(path, delimiter=',')

def normalize_row(feature):
    """normalizes a feature row vector, returning a normalized row vector"""
    feature = (feature - np.mean(feature))/ np.std(feature)
    return feature

def format_and_normalize(ex2_data):
    """formats and normalizes data from andrew ng's ML course"""
    transposed_data = np.flipud(np.transpose(ex2_data))
    normed_data = np.empty([transposed_data.shape[0],transposed_data.shape[1]])
    normed_data[0,:] = transposed_data[0,:]
    for i in xrange(1,transposed_data.shape[0]):
        normed_data[i,:] = normalize_row(transposed_data[i,:])
    return normed_data
