import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

def load_mat(file_name):
    """returns .mat file as a dictionary"""
    return scipy.io.loadmat(file_name)

def scale_normalize(data):
    """scales and normalizes features"""
    num_examples,num_features = data.shape
    normalized_data = np.empty( data.shape )
    for i in xrange(num_features):
        feature = data[:,i]
        mean = np.mean(feature)
        std = np.std(feature)
        normalized_data[:,i] = (feature - mean)/std
    return normalized_data

def plot_2d(data):
    """plots and displays 2 features"""
    x0 = data[:,0].tolist()
    x1 = data[:,1].tolist()
    plt.plot(x0,x1,'ro')

def plot_2d_eigenvecs(reduction_matrix):
    """plots eigenvectors determined by PCA"""
    features, num_eigenvecs = reduction_matrix.shape
    for i in xrange(num_eigenvecs):
        eigenvec = reduction_matrix[:,i].tolist()
        plt.plot([0,eigenvec[0]],[0,eigenvec[1]],linewidth=2.0)

def get_cov_matrix(data):
    """returns the covariance matrix (n by n) for the data (m by n)"""
    num_examples,num_features = data.shape
    cov_matrix = np.dot(np.transpose(data),data) / (1.0 * num_examples)
    return cov_matrix

def num_components_needed(S,variance_retained):
    """calculates number of eigenvectors needed to represent X based on the variance user wishes to retain. Uses the diagonalized eigenvalues to do so."""
    total_num_eigenvals = len(S)
    variance_goal = 1.0 - variance_retained
    sum_eigenvals = np.sum(S)
    current_variance = 1.0
    eigenval_counter = 0
    while current_variance > variance_goal:
        eigen_val = S[eigenval_counter]
        current_variance -= eigen_val/sum_eigenvals
        eigenval_counter += 1
    return eigenval_counter

def get_reduction_matrix(data,variance_retained=None,num_components=None):
    """returns matrix that can be multiplied by data to yield a lower dimensional representation. Automatically chooses # of principal components based on variance that the caller wishes to retain"""
    assert (variance_retained is None) ^ (num_components is None)
    if variance_retained is not None:
        assert 0 <= variance_retained <= 1
        covariance_matrix = get_cov_matrix(data)
        U,S,V = np.linalg.svd(covariance_matrix,full_matrices=False)
        num_components = num_components_needed(S,variance_retained)
        reduction_matrix = U[:,:num_components]
    if num_components is not None:
        covariance_matrix = get_cov_matrix(data)
        U,S,V = np.linalg.svd(covariance_matrix,full_matrices=False)
        reduction_matrix = U[:,:num_components]
    return reduction_matrix

def demo_pca_toy_dataset():
    """runs pca on a toy dataset and plots the results. Note that the results are correct, but the eigenvectors don't look orthogonal because of the aspect ratio."""
    data = load_mat("/home/dchen/projects/ML/training_data/mlclass-ex7/ex7data1.mat")
    X = scale_normalize(data['X'])
    reduction_matrix = get_reduction_matrix(X,variance_retained=.85)
    plot_2d_eigenvecs(reduction_matrix)
    plot_2d(X)
    plt.title("plot of original dataset and the vector we project our data onto""")
    plt.show()
    #Z is the dataset after dimension reduction
    Z = np.dot(X,reduction_matrix)
    reconstructed_X = np.dot(Z,np.transpose(reduction_matrix))
    plt.title("plot of data after being reduced to 1 dimension and being reconstructed")
    plot_2d_eigenvecs(reduction_matrix)
    plot_2d(reconstructed_X)
    plt.show()

def display_faces(X,plot_title=None):
    """displays a 10 by 10 grid of faces"""
    num_faces, features = X.shape
    assert num_faces == 100
    f, subplots = plt.subplots(10, 10, sharex='col', sharey='row', figsize=(32,32))
    for i in xrange(num_faces):
        subplot_row = ((i+1)/10 - 1)
        subplot_col = ((i+1)%10 - 1)
        subplot = subplots[subplot_row][subplot_col]
        face = np.transpose(np.reshape(X[i,:], [32,32]))
        subplot.imshow(face, cmap=cm.Greys_r)
    plt.suptitle(plot_title)
    plt.show()

def demo_eigenfaces():
    """displays original face images, eigenfaces, and faces reconstructed from eigenfaces"""
    data = load_mat("/home/dchen/projects/ML/training_data/mlclass-ex7/ex7faces.mat")
    X = scale_normalize(data['X'])
    #let's only look at the first 100 faces
    X = X[:100,:]
    display_faces(X,"Original Faces")
    reduction_matrix = get_reduction_matrix(X,num_components=100)
    display_faces(np.transpose(reduction_matrix),"First 100 Eigen Faces")
    reduced_faces = np.dot(X,reduction_matrix)
    reconstructed_faces = np.dot(reduced_faces,np.transpose(reduction_matrix))
    display_faces(reconstructed_faces,"Reconstructed Faces")

if __name__ == "__main__":
    demo_pca_toy_dataset()
    demo_eigenfaces()
