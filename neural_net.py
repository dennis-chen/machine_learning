import math
import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy.optimize as op
import scipy.io

class Neuron(object):
    """a single neuron unit, with a logistic activation function"""
    def __init__(self):
        self.activation_value = None
        self.error_value = None

    @staticmethod
    def sigmoid_activation(z):
        """returns sigmoid activation value for input z"""
        return 1/(1+math.e**-z)

class Layer(object):
    """a layer of neurons"""
    def __init__(self, size):
        self.size = size
        self.neurons = []
        self.activation_func = Neuron.sigmoid_activation
        for i in xrange(size):
            self.neurons.append(Neuron())

    def __str__(self):
        layer_representation = np.empty( (self.size,1) )
        for i, neuron in enumerate(self.neurons):
            layer_representation[i][0] = neuron.activation_value
        return str(layer_representation)

    def set_activation(self, activation_values):
        assert activation_values.size == self.size
        for i in xrange(activation_values.size):
            self.neurons[i].activation_value = activation_values[i][0]

    def get_activation(self):
        activations = np.empty( (len(self.neurons),1) )
        for i, neuron in enumerate(self.neurons):
            activations[i][0] = neuron.activation_value
        return activations

    def set_error(self, error_values, output_layer = False):
        if output_layer:
            assert error_values.size == self.size
            for i in xrange(error_values.size):
                self.neurons[i].error_value = error_values[i][0]
        else:
            assert error_values.size == self.size + 1
            for i in xrange(error_values.size - 1):
                self.neurons[i].error_value = error_values[i+1][0]

    def get_error(self):
        errors = np.empty( (len(self.neurons),1) )
        for i, neuron in enumerate(self.neurons):
            errors[i][0] = neuron.error_value
        return errors

class NeuralNet(object):
    """Holds neurons and weights."""
    def __init__(self, architecture_specs, reg_term, training_examples):
        self.RANDOM_INIT_RANGE = 0.12
        self.architecture_specs = architecture_specs
        self._build_architecture(architecture_specs)
        self.training_examples = self._format_training_data(training_examples)
        self.num_training_examples = len(self.training_examples[0])
        self.reg_term = reg_term

    def _build_architecture(self,architecture_specs):
        """creates internal data structures to hold weight info"""
        self.architecture = []
        self.weight_matrices = []
        self.update_accumulators = []
        for num_neurons in architecture_specs:
            self.architecture.append(Layer(num_neurons))
        for i in xrange(len(architecture_specs)-1):
            #initialize to random values for neural net to work
            random_matrix = np.random.random( (architecture_specs[i+1],(architecture_specs[i]+1)) )
            weight_matrix = (random_matrix - 0.5) * self.RANDOM_INIT_RANGE/0.5
            accumulator = np.empty( (architecture_specs[i+1],(architecture_specs[i]+1)) )
            self.weight_matrices.append(weight_matrix)
            self.update_accumulators.append(accumulator)

    def run_training(self):
        """trains the neural net on the MNIST handwritten digits set. Uses conjugate gradient descent to optimize the error function. Saves resulting neural net weights in a pickle file"""
        #randomly initialize weight matrices with doubles between 0 and 1
        initial_thetas = []
        for matrix in self.weight_matrices:
            random_matrix = ( np.random.random( matrix.shape ) - 0.5 ) * self.RANDOM_INIT_RANGE/0.5
            initial_thetas.append(random_matrix)
        bias_thetas, norm_thetas = self._unroll_matrices(initial_thetas)
        initial_thetas_flattened = np.hstack([bias_thetas,norm_thetas])
        #use scipy's built in conjugate gradient descent function to minimize the cost func
        solution = op.fmin_cg(self.evaluate_cost, initial_thetas_flattened, fprime = self.evaluate_cost_gradient,maxiter = 400)
        solution = self._reshape_to_matrices(solution)
        pickle_info = {"solution":solution,"architecture":self.architecture_specs,"reg_term":self.reg_term}
        pickle.dump( pickle_info, open( "trained_net.p", "wb" ) )

    def test_net(self):
        """displays images and prints the predicted image class"""
        test_set = self.training_examples
        Y = test_set[0]
        X = test_set[1]
        num_test_examples = len(Y)
        for i in xrange(num_test_examples):
            x = X[i]
            y = Y[i]
            self.forward_prop(x)
            output_layer = self.architecture[-1]
            hypothesis_vec = output_layer.get_activation()
            hypothesis = hypothesis_vec.argmax() + 1
            print "hypothesis: "+str(hypothesis)
            original_image = np.reshape(x,(20,20) )
            original_image = original_image.T
            plt.imshow(original_image,cmap = plt.get_cmap('gray'))
            plt.show()

    def evaluate_cost(self,thetas):
        """returns cost function for specified thetas"""
        self._set_weight_matrices(thetas)
        training_examples = self.training_examples
        Y = training_examples[0]
        X = training_examples[1]
        num_training_examples = len(Y)
        total_log_cost = 0
        for i in xrange(num_training_examples):
            x = X[i]
            y = Y[i]
            self.forward_prop(x)
            output_layer = self.architecture[-1]
            hypothesis = output_layer.get_activation()
            log_cost = y*np.log(hypothesis) + (np.ones(y.shape) - y)*np.log(np.ones(hypothesis.shape) - hypothesis)
            total_log_cost += np.sum(log_cost)
        #extract norm weights that are encoded in the thetas vector, and ignore the bias weights, since the regularization term should not be counting bias weights
        bias_weights, norm_weights = self._separate_bias_weights(thetas)
        reg_term = self.reg_term*1.0/(2*num_training_examples) * np.sum(norm_weights ** 2)
        total_cost = np.asscalar(-1.0*total_log_cost/num_training_examples + reg_term)
        #scipy's minimization function needs a scalar cost value or weird errors about ambiguous truth values show up
        print "Cost function error: " + str(total_cost)
        return total_cost

    def evaluate_cost_gradient(self,thetas):
        """runs forward/backprop to determine gradient of cost func"""
        self._set_weight_matrices(thetas)
        self.run_forward_back_prop()
        gradient = self.evaluate_gradient()
        return gradient

    def run_forward_back_prop(self):
        """runs forward back prop once for the whole training set"""
        self._zero_accumulated_gradient() #zeros accumulated updates from the last forward/backprop iteration
        training_examples = self.training_examples
        Y = training_examples[0]
        X = training_examples[1]
        for  i in xrange(len(Y)):
            y = Y[i]
            x = X[i]
            self.forward_prop(x)
            self.backward_prop(y)

    def evaluate_gradient(self):
        """uses accumulated updates to calculate derivatives"""
        bias_weights, weights = self._unroll_matrices(self.weight_matrices)
        bias_updates, updates = self._unroll_matrices(self.update_accumulators)
        regularized_norm_updates = (1.0/self.num_training_examples) * updates + (1.0/self.num_training_examples) * self.reg_term * weights
        regularized_bias_updates = (1.0/self.num_training_examples) * bias_updates
        gradient = np.hstack([regularized_bias_updates,regularized_norm_updates])
        return gradient

    def forward_prop(self, x):
        """runs forward propagation once"""
        input_layer = self.architecture[0]
        input_layer.set_activation(x)
        for i in xrange(len(self.architecture) - 1):
            prev_layer = self.architecture[i]
            prev_layer_activations = self._add_bias_unit(prev_layer.get_activation())
            weight_matrix = self.weight_matrices[i]
            next_layer_vals = np.dot(weight_matrix,prev_layer_activations)
            next_layer = self.architecture[i+1]
            next_layer_activations = next_layer.activation_func(next_layer_vals)
            next_layer.set_activation(next_layer_activations)

    def backward_prop(self, y):
        """runs backwards propagation once and updates accumulation matrices"""
        output_layer = self.architecture[-1]
        output_layer_activation = output_layer.get_activation()
        output_error = output_layer_activation - y
        output_layer.set_error(output_error,output_layer = True)
        for i in xrange(len(self.architecture) - 2):
            prev_layer_index = -(i+1)
            prev_layer = self.architecture[prev_layer_index]
            next_layer = self.architecture[prev_layer_index - 1]
            prev_layer_error = prev_layer.get_error()
            next_layer_activation = self._add_bias_unit(next_layer.get_activation())
            weights = np.transpose(self.weight_matrices[-(i+1)])
            ones = np.ones([len(next_layer_activation), 1])
            next_layer_error = np.dot(weights,prev_layer_error)*next_layer_activation*(ones - next_layer_activation )
            next_layer.set_error(next_layer_error)
        update_matrices = []
        for i, accumulator in enumerate(self.update_accumulators):
            prev_layer = self.architecture[i]
            next_layer = self.architecture[i+1]
            prev_layer_activation = self._add_bias_unit(prev_layer.get_activation())
            update = np.dot(next_layer.get_error(),np.transpose(prev_layer_activation))
            accumulator += update

    def test_forward_prop(self):
        """runs one iteration of forward prop"""
        x = self.training_data[:,0:1]
        self._print_activations()
        self.forward_prop(x)
        self._print_activations()

    def test_backward_prop(self):
        """runs one iteration of back prop"""
        y = np.array( [[1],[0]] )
        self._print_errors()
        self.backward_prop(y)
        self._print_errors()

    def _zero_accumulated_gradient(self):
        """resets accumulated gradient values to zero"""
        for accumulator in self.update_accumulators:
            accumulator[:] = 0

    def _set_weight_matrices(self, thetas):
        """reshapes theta vector to matrices and sets the neural net parameters"""
        matrices = self._reshape_to_matrices(thetas)
        self.weight_matrices = matrices

    def _unroll_matrices(self, matrices):
        """unrolls weight and update matrices to look like vectors for processing,
        breaking it into seperate bias and normal neural unit vectors. Each weight matrix is broken up into a matrix of bias weights and a matrix of weights for normal units. We then flatten these 2 matrices and string them together with the flattened matrices from other weight matrices to form two vectors, one for bias units and one for norm units. Then we string the bias units and norm units together and return one long string."""
        result_vec = []
        bias_vec = []
        for matrix in matrices:
            biases = matrix[:,0:1]
            norm_neurons = matrix[:,1:]
            flattened_biases = np.ravel(biases)
            flattened_neurons = np.ravel(norm_neurons)
            result_vec = np.hstack([result_vec,flattened_neurons])
            bias_vec = np.hstack([bias_vec,flattened_biases])
        return bias_vec,result_vec

    def _reshape_to_matrices(self, vector):
        """reshapes weight and update vectors to list of matrices"""
        specs = self.architecture_specs
        bias_unit_count = 0
        original_matrix_dimensions = []
        original_bias_dimensions = []
        for i in xrange(len(specs) - 1):
            bias_unit_count += specs[i+1]
            original_matrix_dimensions.append([specs[i+1],specs[i]])
            original_bias_dimensions.append([specs[i+1],1])
        bias_units = vector[:bias_unit_count]
        norm_units = vector[bias_unit_count:]
        reconstructed_matrices = []
        for i in xrange(len(specs) - 1):
            original_matrix_dim = original_matrix_dimensions[i]
            original_matrix_len = original_matrix_dim[0]*original_matrix_dim[1]
            original_bias_dim = original_bias_dimensions[i]
            original_bias_len = original_bias_dim[0]*original_bias_dim[1]
            unshaped_matrix = norm_units[:original_matrix_len]
            unshaped_bias = bias_units[:original_bias_len]
            norm_units = norm_units[original_matrix_len:]
            bias_units = bias_units[original_bias_len:]
            orig_matrix = np.reshape(unshaped_matrix,original_matrix_dim)
            orig_bias = np.reshape(unshaped_bias,original_bias_dim)
            reconstructed_matrix = np.hstack( [orig_bias,orig_matrix] )
            reconstructed_matrices.append(reconstructed_matrix)
        return reconstructed_matrices

    def _separate_bias_weights(self, vector):
        """separates bias and normal weights and returns them as two vectors"""
        specs = self.architecture_specs
        bias_unit_count = 0
        for i in xrange(len(specs) - 1):
            bias_unit_count += specs[i+1]
        bias_units = vector[:bias_unit_count]
        norm_units = vector[bias_unit_count:]
        return bias_units, norm_units

    def _format_training_data(self, data):
        """formats training data to fit neural net inputs"""
        Y = data['y']
        X = data['X']
        num_examples = Y.size
        #categories = np.unique(Y).size
        categories = 10 #10 categories for 10 digits
        new_Y = []
        new_X = []
        for i in xrange(num_examples):
            y = np.zeros( [categories,1] )
            digit = Y[i][0]
            y[digit - 1] = 1 #convert the digit category to a length ten vector with zeros for all digits except the positive example. ex 3 is encoded as [0 0 1 0 0 0 0 0 0 0] and 10 (which actually represents 0) is encoded in the 10th position with [0 0 0 0 0 0 0 0 0 1]
            x = np.transpose(X[i:i+1,:])
            new_Y.append(y)
            new_X.append(x)
        return(new_Y,new_X)

    def _add_bias_unit(self, activations):
        """appends a 1 to activation vector to represent the bias unit"""
        new_activations = np.ones( (len(activations) + 1, 1) )
        new_activations[1:,:] = activations
        return new_activations

    def _print_activations(self):
        """prints out last activation values of neurons"""
        architecture_rep = []
        for layer in self.architecture:
            layer_rep = []
            for neuron in layer.neurons:
                layer_rep.append(neuron.activation_value)
            architecture_rep.append(layer_rep)
        print architecture_rep

    def _print_errors(self):
        """prints out last errors values of neurons"""
        architecture_rep = []
        for layer in self.architecture:
            layer_rep = []
            for neuron in layer.neurons:
                layer_rep.append(neuron.error_value)
            architecture_rep.append(layer_rep)

    def gradient_check(self):
        """numerically estimates gradient and prints for user to check against backprop"""
        initial_thetas = []
        for matrix in self.weight_matrices:
            random_matrix = ( np.random.random( matrix.shape ) - 0.5 ) * self.RANDOM_INIT_RANGE/0.5
            initial_thetas.append(random_matrix)
        bias_thetas, norm_thetas = self._unroll_matrices(initial_thetas)
        #we put the randomized initial theta into backprop and numerical gradient estimator
        initial_thetas_flattened = np.hstack([bias_thetas,norm_thetas])
        numerical_sol =  self.estimate_gradient(initial_thetas_flattened)
        backprop_sol =  self.evaluate_cost_gradient(initial_thetas_flattened)
        return numerical_sol,backprop_sol

    def estimate_gradient(self,thetas):
        epsilon = 10e-4
        grad_approx = np.empty( thetas.shape )
        for i in xrange(len(thetas)):
            larger_theta = np.empty( thetas.shape )
            smaller_theta = np.empty( thetas.shape )
            np.copyto(larger_theta,thetas)
            np.copyto(smaller_theta,thetas)
            larger_theta[i] += epsilon
            smaller_theta[i] += -epsilon
            grad_approx[i] = ( self.evaluate_cost(larger_theta) - self.evaluate_cost(smaller_theta) )/(2.0*epsilon)
        return grad_approx

def load_coursera_data_file():
    """loads Andrew Ng's coursera class handwritten digit data file and returns numpy matrix"""
    data = scipy.io.loadmat("/home/dchen/projects/ML/training_data/ex4data1.mat")
    y = data['y']
    X = data['X']
    all_data = np.hstack([y,X])
    np.random.shuffle(all_data) #randomly shuffle data so test and training sets will see all examples
    num_examples = all_data.shape[0]
    split_set_index = int(num_examples * 0.8)
    training_y = all_data[:split_set_index,0:1]
    training_X = all_data[:split_set_index,1:]
    training_set = { 'y':training_y,'X':training_X }
    test_y = all_data[split_set_index:,0:1]
    test_X = all_data[split_set_index:,1:]
    test_set = { 'y':test_y,'X':test_X }
    return training_set, test_set

def test_matrices_vector_conversion():
    """does it unravel correctly for scipy?"""
    data = load_coursera_data_file()
    architecture_specs = [2,3,5,2]
    REGULARIZATION_TERM = 0.001
    net = NeuralNet(architecture_specs, REGULARIZATION_TERM, data)
    test_matrices = [np.ones([3,3]),np.zeros([5,4]),np.ones([2,6])]
    bias, normal = net._unroll_matrices(test_matrices)
    vector = np.hstack([bias,normal])
    matrices = net._reshape_to_matrices(vector)
    print matrices

def test_gradient_checking():
    """numerically solves for gradient and compares backprop results"""
    x = np.array( [ [0,0],[1,1] ] )
    y = np.array( [ [1],[2] ] )
    data = {'X':x,'y':y}
    architecture_specs = [2,2,2]
    REGULARIZATION_TERM = 1
    net = NeuralNet(architecture_specs, REGULARIZATION_TERM, data)
    numerical_sol, backprop_sol = net.gradient_check()
    print net._reshape_to_matrices(numerical_sol)
    print net._reshape_to_matrices(backprop_sol)

def load_trained_net(pickle_file, test_set):
    """loads a net with learned weight params from a pickle file that contains the info."""
    net_info = pickle.load( open( pickle_file, "rb" ) )
    weights = net_info["solution"]
    architecture = net_info["architecture"]
    REGULARIZATION_TERM = net_info["reg_term"]
    net = NeuralNet(architecture, REGULARIZATION_TERM, test_set)
    net.weight_matrices = weights
    return net

def test_net(test_set):
    """test the trained network"""
    net = load_trained_net("trained_net.p", test_set)
    net.test_net()

def train_net(training_set):
    """train the network on the MNIST set"""
    architecture_specs = [400,25,10]
    REGULARIZATION_TERM = 1
    net = NeuralNet(architecture_specs, REGULARIZATION_TERM, training_set)
    net.run_training()

if __name__ == "__main__":
    training_set,test_set = load_coursera_data_file()
    train_net(training_set)
    test_net(test_set)
