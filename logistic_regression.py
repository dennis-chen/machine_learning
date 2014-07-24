from gradient_descent import *

def logistic_regression_hyp(features, theta):
    """returns scalar probability that the example is of class 1.
    feed me 2 column vectors."""
    linear_hyp = np.asscalar(np.dot(np.transpose(theta),features))
    e = math.e
    return 1/(1+e**-linear_hyp)

def test_logistic_regression_hyp():
    """should plot the right half of a logistic curve"""
    h_theta = []
    hypothesis = []
    for i in range(-10,10):
        j = i/5.0
        features = np.array([[j]])
        theta = np.array([[j]])
        h_theta.append(j**2)
        hypothesis.append(logistic_regression_hyp(features,theta))
    plt.plot(h_theta,hypothesis,'ro')
    plt.show()

def logistic_regression_error(hypothesis,y):
    """calculates error for given hypothesis and true label"""
    return -float(y)*math.log(hypothesis) - (1-float(y))*math.log(1-hypothesis)

def test_logistic_regression_error():
    """shows two convex curves between 0 and 1. Intuition behind why we choose
    the error function we do: it's easy to optimize."""
    epsilon = 10e-9 #output of h(theta) can't == 0 or 1
    x = np.linspace(0+epsilon,1-epsilon,11).tolist()
    print x
    error_when_y_is_1 = []
    error_when_y_is_0 = []
    for elem in x:
        error_when_y_is_1.append(logistic_regression_error(elem,1))
        error_when_y_is_0.append(logistic_regression_error(elem,0))
    plt.plot(x,error_when_y_is_0,'ro')
    plt.plot(x,error_when_y_is_1,'bo')
    plt.show()

def plot_ex2data(formatted_data):
    """takes numpy data array and plots it for ex2 in the ML exercise"""
    classification = formatted_data[0,:]
    test_1_scores = formatted_data[1,:]
    test_2_scores = formatted_data[2,:]
    for i in range(len(classification)):
        x = test_1_scores[i]
        y = test_2_scores[i]
        c = classification[i]
        if c == 1:
            plt.plot(x,y,'ro')
        else:
            plt.plot(x,y,'bo')
    plt.xlabel('Test 1 Scores')
    plt.ylabel('Test 2 Scores')

def plot_ex2_condition_bound(thetas):
    """plots condition boundary predicted by logistic regression"""
    x1 = np.linspace(-2,2,100)
    bias = np.empty(100)
    theta1 = np.empty(100)
    theta2 = np.empty(100)
    bias.fill(thetas[0][0])
    theta1.fill(thetas[1][0])
    theta2.fill(thetas[2][0])
    x2 = np.divide((-bias - np.multiply(theta1,x1)) , theta2)
    plt.plot(x1,x2,'gx')

def train_log_reg(data):
    """trains on dataset with and returns a gradient descent solver object"""
    gd_solver = GDSolver(logistic_regression_error,logistic_regression_hyp,data,.001)
    init_theta = np.transpose(np.array([[0,0,0]]))
    iterations = 100
    convergence_bound = 10e-9
    gd_solver.run_gd(init_theta,iterations,convergence_bound)
    return gd_solver

def label_prediction(features,theta):
	"""outputs a binary prediction for classification"""
	score = logistic_regression_hyp(features,theta)
	if score >= 0.5:
		return 1
	else:
		return 0

def run_training():
	data = load_csv_file("/home/dchen/projects/ML/training_data/ex2data1.txt")
	formatted_data = format_and_normalize(data)
	solver = train_log_reg(formatted_data)
	solver.plot_error_history()
	theta = solver.learned_theta
	plot_ex2_condition_bound(theta)
	plot_ex2data(formatted_data)
	plt.show()
	return solver

def evaluate_trained_model(solver):
	"""takes a GD solver and sees how it does when classifying a held-out validation set"""
	data = load_csv_file("/home/dchen/projects/ML/training_data/ex2data1.txt")
	formatted_data = format_and_normalize(data)
	theta = solver.learned_theta
	total_examples = formatted_data.shape[1]
	correctly_classified_ex = 0
	for i in xrange(total_examples):
		label = formatted_data[0,i]
		features = formatted_data[1:,i:i+1]
		features = np.vstack(([1],features))
		prediction = label_prediction(features,theta)
		if prediction == int(label):
			correctly_classified_ex += 1
	print "%s/%s correctly classified!"%(correctly_classified_ex,total_examples)
	plot_ex2_condition_bound(theta)
	plot_ex2data(formatted_data)
	plt.show()

if __name__ == "__main__":
	solver = run_training()
	evaluate_trained_model(solver)
