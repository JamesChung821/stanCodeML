from __future__ import print_function

from builtins import range
from builtins import object
import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange

class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    D, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function on the weight matrices. The
    network uses a ReLU nonlinearity after the first fully connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def cost(self, X, y=None):
        """
        Compute the cost and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample of dimension
          D, and there are N samples.
        - y: Vector of training labels, of length N. y[i] is the label for X[i], and
          each y[i] is an integer in the range 0 <= y[i] <= C - 1. This parameter is 
          optional; if it is not passed then we only return scores, and if it is 
          passed then we instead return the cost and gradients.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - cost: cost for this batch of training samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the cost function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # Compute the forward pass
        scores = None
        #############################################################################
        # TODO: Perform the first step of the forward pass by computing the class   #
        # scores for the input. Store the result in the scores variable, which      #
        # should be an array of shape (N, C). Our solution uses 3 lines of code.    #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        K1 = np.dot(W1.T, X.T) + b1.reshape(-1, 1) # (D, H).T dot (N, D).T = (H, N)+(H, 0)
        A1 = np.maximum(0, K1) # (H, N)
        scores = np.dot(W2.T, A1) + b2.reshape(-1, 1) # (H, C).T dot (H, N) = (C, N)
        scores = scores.T # (C, N).T
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the cost
        cost = None
        #############################################################################
        # TODO: Finish the forward pass by computing the cost. Store the result in  #
        # the variable cost, which should be a scalar. Use the softmax classifer    #
        # cost. Our solution uses 5 lines of code.                                  #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        scores -= np.amax(scores, axis=1, keepdims = True) # Maxima along the second axis (N, C)
        H = np.exp(scores)/np.sum(np.exp(scores), axis=1, keepdims=True) # (N, C)
        L = -np.log(H[range(N), y]).reshape(-1, 1) # (N, 1)*************
        cost = np.sum(L, axis=0)/N
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Backward pass: compute gradients
        grads = {}
        #############################################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same      #
        # size. Our solution uses 9 lines of code.                                  #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        sigma = np.exp(scores) / (np.sum(np.exp(scores), axis = 1, keepdims=True))
        sigma[range(N), y] -= 1 # super important (N, C)
        grads_classScores = sigma / N # note: cost function has factor 1/N
        grads['b2'] = np.sum(grads_classScores, axis = 0) # (N, C) to (1, C)
        grads['W2'] = np.dot(A1, grads_classScores) # (H, N) dot (N, C)
        dA = np.dot(W2, grads_classScores.T) # (H, C) dot (N, C).T
        dK = dA * np.where(K1>0, 1, 0) #(H, N)
        grads['b1'] = np.sum(dK, axis=1) # (H, N) to (H, 1)
        grads['W1'] = np.dot(dK, X).T # (H, N) dot (N, D) .T to (D, H)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return cost, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95, 
              num_iters=100, batch_size=200,
              verbose=False):
        """
        Train this neural network using batch gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        print('num_train', num_train)
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use batch GD to optimize the parameters in self.model
        cost_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively. Our solution uses 3 lines   #
            # of code.                                                              #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            if batch_size > num_train:
              status = True
            else:
              status = False
            index = np.random.choice(num_train, batch_size, replace=status) # e.g. [44783, 1920, 24571]
            # index = np.random.permutation(num_train)[:batch_size]
            X_batch, y_batch = X[index], y[index]
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # Compute cost and gradients using the current minibatch
            cost, grads = self.cost(X_batch, y=y_batch)
            cost_history.append(cost)

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using batch gradient descent. You'll need to use the gradients        #
            # stored in the grads dictionary defined above. Our solution uses 4     #
            # lines of code.                                                        #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            self.params['W1'] = self.params['W1'] - learning_rate*grads['W1']
            self.params['W2'] = self.params['W2'] - learning_rate*grads['W2']
            self.params['b1'] = self.params['b1'] - learning_rate*grads['b1']
            self.params['b2'] = self.params['b2'] - learning_rate*grads['b2']
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            if verbose and it % 100 == 0:
                print('iteration %d / %d: cost %f' % (it, num_iters, cost))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
          'cost_history': cost_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        y_pred = None

        ###########################################################################
        # TODO: Implement this function. Our solution uses 4 lines of code.       #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        K = np.dot(W1.T, X.T) + b1.reshape(-1, 1) # (D, H).T dot (N, D).T = (H, N)+(H, 0)
        A = np.maximum(0, K) # (H, N)
        scores = np.dot(W2.T, A) + b2.reshape(-1, 1) # (H, C).T dot (H, N) = (C, N)
        scores = scores.T # (N, C)
        H = np.exp(scores)/np.sum(np.exp(scores), axis=1, keepdims=True) # (N, C)
        y_pred = np.argmax(H, axis=1) # (N, C) to (N,)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred
