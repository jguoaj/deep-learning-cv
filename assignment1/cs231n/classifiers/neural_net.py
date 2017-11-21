from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  # this class contains __init__, loss(), train(), predict()
  # example: TwoLayerNet(4, 10, 3, std=1e-1)
  # input_size = 4, hidden_size = 10, num_classes = 3, num_inputs = 5
  # ReLU: max(0, x)
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
    self.params = {}													# for example
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)	# (4, 10)
    self.params['b1'] = np.zeros(hidden_size)							# (10,)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)	# (10, 3)
    self.params['b2'] = np.zeros(output_size)							# (3,)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']	# W1: (4, 10), b1: (10,)
    W2, b2 = self.params['W2'], self.params['b2']	# W2: (10, 3), b2: (3,)
    N, D = X.shape									# X.shape = (5, 4), y.shape = (5,)

    # Compute the forward pass
    scores = None 									
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    h1 = np.maximum(0, X.dot(W1) + b1)              # (5, 10)
    scores = h1.dot(W2) + b2						# (5, 3)

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = 0.0
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss.                                                          #
    #############################################################################
    # from cs231n.classifiers.softmax import softmax_loss_vectorized
    # regulization trick
    # scores_row_max = np.max(scores, axis=1)                       # (5,)
    # scores = scores - scores_row_max.reshape(scores.shape[0], 1)  # (5, 3)

    ### forward pass
    # correct_score = scores[np.arange(scores.shape[0]), y]             # (5,)
    # exp_correct_score = np.exp(correct_score)                         # (5,)
    # exp_scores = np.exp(scores)                                       # (5, 3)
    # exp_row_sum = np.sum(exp_scores, axis=1)                          # (5,)
    # inv_exp_row_sum = 1 / exp_row_sum                                 # (5,)
    # product_num_den = exp_correct_score * inv_exp_row_sum             # (5,)
    # loss = loss - np.sum(np.log(product_num_den))
    
    ### new method
    num_examples = X.shape[0]
    # get unnormalized probabilities, (5, 3)
    exp_scores = np.exp(scores)
    # normalize them for each example, (5, 3)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    corect_logprobs = -np.log(probs[range(num_examples),y])

    data_loss = np.sum(corect_logprobs)/num_examples
    reg_loss = reg * np.sum(W1 * W1) + reg * np.sum(W2 * W2)
    loss = data_loss + reg_loss

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################

    ### back propagation
    # d_product_num_den = (-1/N) * (-1/product_num_den)                  # (5,)
    # d_exp_correct_score = inv_exp_row_sum                              # (5,)
    # d_inv_exp_row_sum = exp_correct_score                              # (5,)
    # d_exp_row_sum = (-1.0 / (exp_row_sum**2)) * d_inv_exp_row_sum      # (5,)
    # d_exp_scores = np.ones(exp_scores.shape) * d_exp_row_sum           # (5, 3)
    # d_scores = np.exp(scores) * d_exp_scores                           # (5, 3)
    
    ### using new method
    d_scores = probs                           # (5, 3)
    d_scores[range(num_examples), y] -= 1
    d_scores /= num_examples

    # W1: (4, 10), b1: (10,)
    # W2: (10, 3), b2: (3,)
    # X.shape = (5, 4), y.shape = (5,)
     
    # scores = h1 * W2 + b2
    d_W2 = np.dot(h1.T, d_scores)                           # (10, 3)
    d_W2 += reg * 2 * W2                                    
    d_b2 = np.sum(d_scores, axis=0, keepdims=False)

    # h1 = max(0, X * W1 + b1)
    d_h1 = d_scores.dot(W2.T)                               # (5, 10)
    ReLU_d_h1 = d_h1 * (h1 > 0)                             # max(0, ) operation
    d_W1 = np.dot(X.T, ReLU_d_h1)
    d_W1 += reg * 2 * W1                                    # (4, 10)
    d_b1 = np.sum(ReLU_d_h1, axis=0, keepdims=False)
    
    grads['W2'] = d_W2
    grads['W1'] = d_W1
    grads['b2'] = d_b2
    grads['b1'] = d_b1

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

    # stats = net.train(X, y, X, y,
    #         learning_rate=1e-1, reg=5e-6,
    #         num_iters=100, verbose=False)
  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      mask = np.random.choice(num_train, batch_size, replace=True)
      X_batch = X[mask]
      y_batch = y[mask]
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      self.params['W1'] -= learning_rate * grads['W1']
      self.params['b1'] -= learning_rate * grads['b1']
      self.params['W2'] -= learning_rate * grads['W2']
      self.params['b2'] -= learning_rate * grads['b2']
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

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
      'loss_history': loss_history,
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
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    y_pred = np.zeros(X.shape[0])
    h1 = np.maximum(0, X.dot(self.params['W1']) + self.params['b1'])
    scores = h1.dot(self.params['W2']) + self.params['b2']
    y_pred = np.argmax(scores, axis=1)

    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


