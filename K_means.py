import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy import misc
import scipy.io

#when change in kmeans error < EPSILON, convergence occured
EPSILON = 10e-6

def load_mat(file_name):
    """loads .mat files into a dictionary"""
    data = scipy.io.loadmat(file_name)
    return data

def initialize_centroids(X,k):
    """returns coordinates of k centroids, randomly initialized to be equal to some example from X"""
    example_num, feature_num = X.shape
    indices = range(example_num)
    random_indices = random.sample(indices, k)
    random_centroids = []
    for index in random_indices:
        random_centroid = X[index,:]
        random_centroids.append(random_centroid)
    return random_centroids

def get_squared_dist(x,centroid):
    """returns the squared norm of the distance between two vectors."""
    squared_dist = abs(centroid - x)**2
    return np.sum(squared_dist)

def find_closest_centroid_index(x,centroids):
    """returns index of centroid that a single training example x is closest to"""
    dist_to_centroids = []
    for centroid_index, centroid in enumerate(centroids):
        dist_to_centroids.append(get_squared_dist(x,centroid))
    closest_centroid_index = dist_to_centroids.index(min(dist_to_centroids))
    return closest_centroid_index

def get_clusters(centroids, X):
    """X is a n by m set of points. Returns a n by 1 set of indices that indicate which centroid a point is closest to."""
    example_num, feature_num = X.shape
    centroid_indices = np.empty( [example_num, 1] )
    for i in xrange(example_num):
        x = X[i,:]
        closest_centroid_index = find_closest_centroid_index(x,centroids)
        centroid_indices[i][0] = closest_centroid_index
    return centroid_indices

def plot_2d(centroids, centroid_indices, X, K):
    """visualizes centroid and clusters (assumes there are only 2 features and won't plot more) and that K is not larger than the number of named matplotlib colors"""
    example_num, feature_num = X.shape
    color_list = matplotlib.colors.cnames.keys()
    random.shuffle(color_list)
    for i in xrange(example_num):
        x = X[i,:]
        centroid_index = int(centroid_indices[i][0])
        plt.plot(x[0],x[1],color=color_list[centroid_index],marker='o',markersize=10)
    for j, centroid in enumerate(centroids):
        plt.plot(centroid[0],centroid[1],color="black",marker='x',markersize=25)
    plt.show()

def update_centroids(cluster_indices,X,k):
    """returns k new centroids at the location of the mean of each cluster"""
    example_num, feature_num = X.shape
    cluster_sums = [np.zeros([feature_num]) for i in xrange(k)]
    example_counter = [0] * k
    for i in xrange(example_num):
        x = X[i,:]
        cluster_index = int(cluster_indices[i][0])
        cluster_sums[cluster_index] += x
        example_counter[cluster_index] += 1
    cluster_avgs = [sums/example_num for sums, example_num in zip(cluster_sums,example_counter)]
    return cluster_avgs

def calc_cost(centroids,cluster_indices,X,k):
    """determines distortion cost function that K-means is trying to minimize"""
    example_num, feature_num = X.shape
    sum_squared_dist = 0
    for i in xrange(example_num):
        x = X[i,:]
        closest_centroid = centroids[int(cluster_indices[i][0])]
        squared_dist = get_squared_dist(closest_centroid,x)
        sum_squared_dist += squared_dist
    return sum_squared_dist/example_num

def run_k_means(X,k):
    """runs k means on the dataset X and returns the final centroid and cluster information, as well as the cost function minima reached"""
    example_num, feature_num = X.shape
    centroids = initialize_centroids(X,k)
    cluster_indices = get_clusters(centroids, X)
    #initialize cost change to be larger than epsilon. The loop below tests for convergence of the algorithm and exits once updates won't change centroid locations anymore.
    cost_change = EPSILON + 1
    while cost_change > EPSILON:
        old_cost = calc_cost(centroids,cluster_indices,X,k)
        centroids = update_centroids(cluster_indices,X,k)
        cluster_indices = get_clusters(centroids, X)
        new_cost = calc_cost(centroids,cluster_indices,X,k)
        cost_change = old_cost - new_cost
    final_cost = calc_cost(centroids,cluster_indices,X,k)
    return (centroids,cluster_indices,final_cost)

def search_global_optima(X,k,iterations):
    """runs k means specified number of times and saves the distortion cost function of each clustering to find the global (or at least the best found local) minimum"""
    run_results = []
    for i in xrange(iterations):
        run_results.append(run_k_means(X,k))
    #sort k means results by final cost function value
    sorted_results = sorted(run_results, key=lambda results: results[2])
    return sorted_results[0]

def demo_kmeans_toy_dataset():
    """finds best clustering for a simple toy dataset and plots the solution"""
    data = load_mat("/home/dchen/projects/ML/training_data/mlclass-ex7/ex7data2.mat")
    X = data['X']
    cluster_num = 3
    iterations = 50
    centroids,cluster_indices,final_cost = search_global_optima(X,cluster_num,iterations)
    print "final error: " + str(final_cost)
    plot_2d(centroids,cluster_indices,X,cluster_num)

def convert_img_to_training_set(img_file):
    """converts an image into a numpy array that k means will take"""
    img = misc.imread(img_file)
    height, width, depth = img.shape
    num_pixels = height * width
    pixel_vals = []
    for i in xrange(depth):
        pixel_val = np.ravel(img[:,:,i]).tolist()
        pixel_vals.append(pixel_val)
    X = np.array(zip(*pixel_vals))
    return X,img.shape

def convert_set_to_img(X, original_shape):
    """converts X back to a format that can be saved as an image"""
    height, width, depth = original_shape
    img = np.empty(original_shape)
    for i in xrange(depth):
        img[:,:,i] = np.reshape( X[:,i], (height,width) )
    return img

def set_to_closest_centroid(X,centroids):
    """converts all examples in X to have the same value as the closest centroid in the centroids list"""
    example_num, feature_num = X.shape
    for i in xrange(example_num):
        x = X[i,:]
        closest_centroid_index = find_closest_centroid_index(x, centroids)
        X[i,:] = centroids[closest_centroid_index]
    return X

def compress_image(img_file, num_colors):
    """uses kmeans clustering to choose num_colors colors to represent the image with"""
    X, original_shape = convert_img_to_training_set(img_file)
    iterations = 1
    centroids,cluster_indices,final_cost = search_global_optima(X,num_colors,iterations)
    recolored_X = set_to_closest_centroid(X,centroids)
    new_img = convert_set_to_img(recolored_X, original_shape)
    #chop off old filename extension in order to append _compressed to new file name
    new_img_name = img_file[:-4] + "_compressed.jpg"
    misc.imsave(new_img_name, new_img)

def demo_img_compression():
    """compresses an image using kmeans to represent the data with less colors. saves the new image"""
    num_colors = 16
    img_file_name = "/home/dchen/projects/ML/training_data/mlclass-ex7/demo.png"
    compressed_img = compress_image(img_file_name,num_colors)

if __name__ == "__main__":
    demo_kmeans_toy_dataset()
    demo_img_compression()
