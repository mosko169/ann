# code base taken from https://iamtrask.github.io/2015/07/12/basic-python-network/

import numpy as np

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

def split_tests_to_matrices(tests):
    inputs = []
    outputs = []
    for test in tests:
        inputs.append(test[0])
        outputs.append(test[1])
    return np.array(inputs), np.array([outputs]).T

def run_network(tests):
    # input dataset
    X, y = split_tests_to_matrices(tests)

    # seed random numbers to make calculation
    # deterministic (just a good practice)
    np.random.seed(1)

    # initialize weights randomly with mean 0
    syn0 = 2*np.random.random((X.shape[1], 1)) - 1

    for iter in xrange(10000):

        # forward propagation
        l0 = X
        l1 = nonlin(np.dot(l0,syn0))

        # how much did we miss?
        l1_error = y - l1

        # multiply how much we missed by the
        # slope of the sigmoid at the values in l1
        l1_delta = l1_error * nonlin(l1,True)

        # update weights
        syn0 += np.dot(l0.T,l1_delta)

    print "Output After Training:"
    print l1

def main():

if __name__ == '__main__':
    main()