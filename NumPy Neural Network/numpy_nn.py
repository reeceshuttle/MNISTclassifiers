import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time

# --------------------
# helper functions:
def convert_to_grid(datapoint):
    datapoint = np.squeeze(datapoint)
    im = []
    for i in range(28):
        im.append([datapoint[28*i+j] for j in range(28)])
    return np.array(im).T

def plot_examples():
    rows, cols = 6, 14
    figure, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(14, 6))
    for i in range(rows):
        for j in range(cols):
            axes[i][j].imshow(convert_to_grid(training_datapoints[cols*i+j]), cmap='gray')
            axes[i][j].set(title=f'label:{training_labels[cols*i+j][0]%10}', yticks=[], xticks=[])
    plt.tight_layout()
    plt.show()

def convert_label_to_vect(label: int):
    """
    Note that 0 is labelled as 10. 
    We will make the 0th index correspond to zero.
    ***this function has been checked manually, lgtm***
    """
    vect = np.zeros((1,10))
    vect[0][label%10] = 1
    return vect.T # we want a column vector
        
def get_metrics(current_network, points, labels):
    """will return classification error and total loss
    for both training and testing sets."""
    total_loss = 0
    total_incorrect = 0
    for point, label_ in zip(points, labels):
        label = label_[0]%10 # since 10 is 0's label
        point_in = np.array([point]).T
        prediction = forward_pass(current_network, point_in)
        loss, _ = MeanSquared(prediction, convert_label_to_vect(label))
        total_loss += loss
        predicted_class = np.argmax(prediction)
        if predicted_class != label:
            total_incorrect += 1
    return total_loss, total_incorrect/len(points)

# --------------------
# activation functions:

def Identity(vect):
    return vect

def ReLU(vect):
    return vect * (vect>0)

def Sigmoid(vect):
    return 1/(1+np.exp(-vect))

def Softmax(vect):
   v_sum = np.sum(np.exp(vect))
   return np.exp(vect)/v_sum

# --------------------
# activation derivative functions:

def dSigmoid(vect):
    """
    Vect here would be y, or what the Sigmoid function 
    was applied to during the forward pass.
    ***returns the matrix for the derivative 
    aka along the diagonal***
    """
    diagonal_vals = Sigmoid(vect)*(1-Sigmoid(vect))
    return np.diag(np.squeeze(diagonal_vals))

# --------------------
# loss functions:

def MeanSquared(estimate, actual):
    """
    This will return both the loss value and the gradient, dLdz.
    """
    Loss = np.sum((estimate-actual)**2)
    dLdz = 2*(estimate-actual)
    return Loss, dLdz

# --------------------


def initialize_network(hidden_layer_size=20):
    """
    initializes a nn with a single hidden layer.
    returns in form [{'W':W1,'activation':ReLU}, {'W':W2,'activation':Softmax}]
    """
    W1 = np.array([[(random.random()-0.5) for i in range(hidden_layer_size)] for j in range(784)])
    offset1 = np.array([[(random.random()-0.5)] for i in range(hidden_layer_size)])
    W2 = np.array([[(random.random()-0.5) for i in range(10)] for j in range(hidden_layer_size)])
    offset2 = np.array([[(random.random()-0.5)] for i in range(10)])
    return [{'W': W1, 'offset': offset1, 'activation': Sigmoid, 'd_activation': dSigmoid}, {'W': W2, 'offset': offset2, 'activation': Sigmoid, 'd_activation': dSigmoid}]



def forward_pass(network, datapoint):
    """gets list that represents the network and gets the output"""
    x = datapoint
    for layer in network:
        layer['layer_input'] = np.copy(x)
        y = layer['W'].T@x + layer['offset']
        layer['pre_activation_output'] = np.copy(y)
        x = layer['activation'](y)
    return x


def back_propagation(network, dLdz, lr=0.001):
    for i in range(len(network)-1, -1, -1):
        layer = network[i]
        dLdy = layer['d_activation'](layer['pre_activation_output'])@dLdz
        dLdW = layer['layer_input']*(dLdy.T)
        if i > 0: dLdz = layer['W']@dLdy # note that we dont have to do this when i = 0(saves 60 sec with 1000000 steps)
        # update:
        layer['W'] -= lr*dLdW
        layer['offset'] -= lr*dLdy


def train(network, training_data, training_labels, testing_data, testing_labels, stochastic_steps, edit_global_graphs=False):
    training_start = time.time()
    for step_num in range(stochastic_steps):
        if step_num%100000==0 and step_num!=0:
            print(f'reached step {step_num} at {time.time()-training_start} sec')

        if step_num%10000==0 and edit_global_graphs:
            total_testing_loss, testing_error_proportion = get_metrics(network, testing_data, testing_labels)
            testing_loss.append(total_testing_loss)
            testing_errors.append(testing_error_proportion)
            total_training_loss, training_error_proportion = get_metrics(network, training_data, training_labels)
            training_loss.append(total_training_loss)
            training_errors.append(training_error_proportion)
            corresponding_step.append(step_num)

        i = random.randint(0, 3000-1)
        point, label = np.array([training_data[i]]).T, training_labels[i][0]
        estimate = forward_pass(network, point)
        Loss, dLdz = MeanSquared(estimate, convert_label_to_vect(label))
        back_propagation(network, dLdz)
    
    print(f'total training time:{time.time()-training_start} sec')





if __name__ == "__main__":
    t1 = time.time()
    training_data = loadmat('Numpy Neural Network/mnist_training.mat')
    training_datapoints, training_labels = training_data['X_train'], training_data['y_train']
    testing_data = loadmat('Numpy Neural Network/mnist_test.mat')
    testing_datapoints, testing_labels = testing_data['X_test'], testing_data['y_test']
    print(f'data loading time:{time.time()-t1} sec')



    t2 = time.time()
    network = initialize_network()
    print(f'network initialization time:{time.time()-t2} sec')
    stochastic_steps = 500000
    corresponding_step = []
    testing_loss = []
    training_loss = []
    testing_errors = []
    training_errors = []

    train(network, training_datapoints, training_labels, testing_datapoints, testing_labels, stochastic_steps, edit_global_graphs=True)

    final_testing_loss, final_test_error_proportion = get_metrics(network, testing_datapoints, testing_labels)
    
    fig, ax = plt.subplots(2)
    ax[0].plot(corresponding_step, testing_errors, label='testing')
    ax[0].plot(corresponding_step, training_errors, label='training')
    ax[0].set(title=f'graph of classification error\nfinal testing error: {final_test_error_proportion}', xlabel='stochastic step number', ylabel='classification error')
    # plt.legend()
    
    ax[1].plot(corresponding_step, testing_loss, label='testing')
    ax[1].plot(corresponding_step, training_loss, label='training')
    ax[1].set(title='graph of total loss', xlabel='stochastic step number', ylabel='total loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # receptive fields aka the maximal input:
    # rows, cols = 4, 5
    # figure, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(14, 6))
    # for i in range(rows):
    #     for j in range(cols):
    #         axes[i][j].imshow(convert_to_grid(network[0]['W'].T[cols*i+j]), cmap='gray')
    #         axes[i][j].set(title=f'Weights for Y_{cols*i+j+1} Visualized', yticks=[], xticks=[])
    # plt.tight_layout()
    # plt.show()
