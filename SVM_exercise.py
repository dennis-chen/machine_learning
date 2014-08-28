from scipy.io import loadmat
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.stem.porter import *
import string
import re

def plot_dataset_data(X,y):
    """plots the example data, which has two features and two classes"""
    num_examples = X.shape[0]
    for i in xrange(num_examples):
        x1 = X[i,0]
        x2 = X[i,1]
        hypothesis = y[i]
        if hypothesis == 0:
            plt.plot(x1,x2,'ro')
        else:
            plt.plot(x1,x2,'go')

def plot_decision_boundary(svm,X,border_size):
    """colors decision boundaries for two classes"""
    min_x0 = X[:,0].min()
    max_x0 = X[:,0].max()
    min_x1 = X[:,1].min()
    max_x1 = X[:,1].max()
    num_samples = 100
    x0 = np.linspace(min_x0,max_x0,num_samples)
    x1 = np.linspace(min_x1,max_x1,num_samples)
    xx0,xx1 = np.meshgrid(x0,x1)
    xx0_flat = xx0.ravel()
    xx1_flat = xx1.ravel()
    hypotheses = np.empty(xx0.shape).ravel()
    for i in xrange(len(xx0_flat)):
        x0 = xx0_flat[i]
        x1 = xx1_flat[i]
        hypotheses[i] = svm.predict([x0,x1])
    hypotheses = np.reshape(hypotheses, (num_samples,num_samples) )
    plt.contourf(xx0,xx1,hypotheses, cmap=plt.cm.Paired, alpha=0.8)
    plt.axis([min_x0 - border_size,max_x0 + border_size,min_x1 - border_size,max_x1 + border_size])

def test_on_dataset_1():
    """run our svm on a simple dataset to see how tuning C and sigma affect the decision boundary"""
    data = loadmat("/home/dchen/projects/ML/training_data/mlclass-ex6/ex6data1.mat")
    X = data['X']
    y = data['y']
    linear_svc = svm.SVC(kernel='linear',C = 100)
    linear_svc.fit(X,y)
    plot_dataset_data(X,y)
    plot_decision_boundary(linear_svc,X,1)
    plt.title('Decision boundary with C = 100')
    plt.show()
    linear_svc = svm.SVC(kernel='linear',C = 1)
    linear_svc.fit(X,y)
    plot_dataset_data(X,y)
    plot_decision_boundary(linear_svc,X,1)
    plt.title('Decision boundary with C = 1')
    plt.show()

def test_on_dataset_2():
    """try fitting a gaussian kernel function to the 2nd dataset"""
    data = loadmat("/home/dchen/projects/ML/training_data/mlclass-ex6/ex6data2.mat")
    X = data['X']
    y = data['y']
    rbf_svc = svm.SVC(kernel='rbf',C = 200,gamma = 5)
    rbf_svc.fit(X,y)
    plot_dataset_data(X,y)
    plot_decision_boundary(rbf_svc,x,0.3)
    plt.title('decision boundary with c = 100 and gaussian kernel')
    plt.show()

def test_accuracy(svm,x,y):
    """determines the accuracy of a svm classifier on validation set"""
    hypothesis = svm.predict(x)
    flat_y = y.ravel()
    misclassification_count = 0
    for i in xrange(len(flat_y)):
        if not( hypothesis[i] == flat_y[i] ):
            misclassification_count += 1
    return misclassification_count

def find_best_params_for_dataset_3():
    """searches solution space for the optimal c and gamma for dataset 3"""
    data = loadmat("/home/dchen/projects/ML/training_data/mlclass-ex6/ex6data3.mat")
    x = data['X']
    y = data['y']
    X_val = data['Xval']
    y_val = data['yval']
    values_to_try = [0.01,0.03,0.1,0.3,1,3,10,30,100,300]
    accuracy_matrix = np.empty( [len(values_to_try), len(values_to_try)] )
    for i, c_val in enumerate(values_to_try):
        for j, gamma_val in enumerate(values_to_try):
            rbf_svc = svm.SVC(kernel='rbf',C = c_val,gamma = gamma_val)
            rbf_svc.fit(X_val,y_val)
            accuracy = test_accuracy(rbf_svc,X_val,y_val)
            accuracy_matrix[i][j] = accuracy
    print accuracy_matrix
    print "plotting decision boundary of most accurate SVM"
    best_val_indices = np.where(accuracy_matrix==accuracy_matrix.min())
    #take the first configuration that gives us the best results, even if there are ties among configurations
    best_val_index = [best_val_indices[0][0],best_val_indices[1][0]]
    best_C = values_to_try[best_val_index[0]]
    best_gamma = values_to_try[best_val_index[1]]
    rbf_svc = svm.SVC(kernel='rbf',C = best_C,gamma = best_gamma)
    rbf_svc.fit(X_val,y_val)
    plot_dataset_data(X_val,y_val)
    plot_decision_boundary(rbf_svc,X_val,0.3)
    print "best parameters are C = %s and gamma = %s"%(best_C,best_gamma)
    plt.show()

def load_text_file(file_name):
    """loads text file and returns contents as a string"""
    with open (file_name, "r") as myfile:
        data = myfile.read()
    return data

def is_english_word(string):
    d = enchant.Dict("en_US")
    return d.check(string)

def preprocess_email(email):
    """lower cases, strips html, normalizes urls, emails, dollars, numbers, and dollars, reduces words to stem forms, and replaces all whitespaces with a single space"""
    text_file = email.lower()
    text_file = re.sub('<[^<>]+>', ' ',text_file)
    text_file = re.sub('[0-9]+', 'number', text_file)
    text_file = re.sub('(http|https)://[^\s]*', 'httpaddr', text_file)
    text_file = re.sub('[^\s]+@[^\s]+', 'emailaddr', text_file)
    text_file = re.sub('[$]+', 'dollar', text_file)
    no_punc = ''
    for char in text_file:
        if char not in string.punctuation:
            no_punc += char
    tokenized = nltk.word_tokenize(no_punc)
    stemmed_word_list = []
    stemmer = PorterStemmer()
    for word in tokenized:
        stemmed_word_list.append(stemmer.stem(word))
    return stemmed_word_list

def load_vocab(vocab_list):
    """loads vocab text file and returns a vocabulary dictionary, where the keys are words and the vals are the indice of the vocab word in the feature vector"""
    text = load_text_file(vocab_list)
    text = re.sub('[0-9]+','',text)
    text = text.split()
    vocab_dict = {}
    for i, word in enumerate(text):
        vocab_dict[word] = i
    return vocab_dict

def extract_feature_vector(file_name):
    """extracts a feature vector from an email"""
    text = load_text_file(file_name)
    vocab_dict = load_vocab("/home/dchen/projects/ML/training_data/mlclass-ex6/vocab.txt")
    preprocessed_text = preprocess_email(text)
    feature_vec = np.zeros( [1, len(vocab_dict)] )
    for word in preprocessed_text:
        if word in vocab_dict:
            feature_index = vocab_dict[word]
            feature_vec[0][feature_index] = 1
    return feature_vec

def train_spam_classifier():
    data = loadmat("/home/dchen/projects/ML/training_data/mlclass-ex6/spamTrain.mat")
    X = data['X']
    y = data['y']
    rbf_svc = svm.SVC(kernel='rbf',C = 3,gamma = 5)
    rbf_svc.fit(X,y)
    return rbf_svc

def test_spam_classifier(classifier):
    data = loadmat("/home/dchen/projects/ML/training_data/mlclass-ex6/spamTest.mat")
    X_test= data['Xtest']
    y_test= data['ytest']
    misclassified_examples = test_accuracy(classifier,X_test,y_test)
    print "misclassified examples in test set: "
    print str(misclassified_examples) + "/%s"%(y_test.shape[0])

    #now try it on an example email!
    feature_vec = extract_feature_vector("/home/dchen/projects/ML/training_data/mlclass-ex6/notSpam.txt")
    print classifier.predict(feature_vec)

if __name__ == "__main__":
    spam_classifier = train_spam_classifier()
    test_spam_classifier(spam_classifier)
