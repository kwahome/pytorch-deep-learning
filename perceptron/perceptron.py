import numpy as np

# Setting the random seed, feel free to change it and see different solutions.
np.random.seed(42)


def step_function(t):
    if t >= 0:
        return 1
    return 0


def prediction(p_input, weight, bias):
    return step_function((np.matmul(p_input, weight) + bias)[0])


# TODO: Fill in the code below to implement the perceptron trick.
# The function should receive as inputs the data X, the labels y,
# the weights W (as an array), and the bias b,
# update the weights and bias W, b, according to the perceptron algorithm,
# and return W and b.
def perceptron_step(inputs, output, weights, bias, learn_rate=0.01):
    for i in range(len(inputs)):
        y_hat = prediction(inputs[i], weights, bias)
        if output[i] - y_hat == 1:
            weights[0] += inputs[i][0] * learn_rate
            weights[1] += inputs[i][1] * learn_rate
            bias += learn_rate
        elif output[i] - y_hat == -1:
            weights[0] -= inputs[i][0] * learn_rate
            weights[1] -= inputs[i][1] * learn_rate
            bias -= learn_rate
    return weights, bias


# This function runs the perceptron algorithm repeatedly on the dataset,
# and returns a few of the boundary lines obtained in the iterations,
# for plotting purposes.
# Feel free to play with the learning rate and the num_epochs,
# and see your results plotted below.
def train_perceptron_algorithm(X, y, learn_rate=0.01, num_epochs=25):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2, 1))
    b = np.random.rand(1)[0] + x_max
    # These are the solution lines that get plotted below.
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptron_step(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0] / W[1], -b / W[1]))
    return boundary_lines
