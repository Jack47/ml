import numpy as np
# Reference: http://iamtrask.github.io/2015/07/12/basic-python-network/
# A simple 3 inputs and 1 output neural network

# sigmod function
def sigmod(x, derive=False):
    if derive==True: # input x is already sigmod, we calculate sigmod's derive based on calculated sigmod
        return x*(1-x)
    return 1/(1+np.exp(-x))

# deterministic
np.random.seed(1)

# input dataset
# contains 4 training examples, each has three input nodes
# we will process all of the 4 training examples at the same time(full batch training)
X = np.array([[0, 0, 1],
             [0, 1, 1],
             [1, 0, 1],
             [1, 1, 1]])

# output dataset
y = np.array([[0, 0, 1, 1]]).T
print("real output: \n",y)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((3, 1)) - 1 # [ -1~1 ]
print("start syn0: \n", syn0)
for i in range(0, 10):
    # forward propagation
    l0 = X # [4, 3]
    # l1 generate guess for the 4 examples
    l1 = sigmod(np.dot(l0, syn0)) # [4, 1]

    # how much did we miss
    # it's a vector of positive and negative numbers reflecting
    # how much the network missed
    l1_pred_err = y-l1

    print("l1_pred_err\n", l1_pred_err)
    # multiply how much we missed by the
    # slope of the sigmod at the values in l1
    # multiplying them "elementwise"
    ## derivation is always between 0 and 1
    l1_delta = l1_pred_err*sigmod(l1, derive=True) # [4, 1]
    print("l1_delta\n", l1_delta)

    # back propagation: update weights
    # all of learning is store in syn0 matrix
    syn0 += np.dot(l0.T, l1_delta) # [ 3, 1]

print("After training\n")
print("syn0: \n", syn0)
print("output: \n", l1)
