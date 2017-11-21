import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.                         # W.shape = (3073, 10)
  - X: A numpy array of shape (N, D) containing a minibatch of data.             # X_dev.shape = (500, 3073)
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means    # y_dev.shape = (500,)
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # compute the loss and the gradient
  num_classes = W.shape[1]  # 10, as example
  num_train = X.shape[0]    # 500, as example

  for i in xrange(num_train):
    scores = X[i].dot(W)        # (10,)
    scores -= np.max(scores)    # normalization trick
    correct_class_score = scores[y[i]]
    denominator = np.sum(np.exp(scores))
    correct_p = (np.exp(correct_class_score) / denominator)  # correct class probability
    loss = loss - np.log(correct_p)

    for j in xrange(num_classes):
      class_p = (np.exp(scores[j]) / denominator)
      if j == y[i]:         # if the index is the correct class label's index
        dW[:,j] += (-1 + class_p) * X[i,:].T
      else:
        dW[:,j] += class_p * X[i,:].T

  loss /= num_train
  loss += reg * np.sum(W * W)  # element-wise multiplication

  dW /= num_train              # that's important
  dW += reg * 2 * W            # that's tricky
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  Inputs:
  - W: A numpy array of shape (D, C) containing weights.                         # W.shape = (3073, 10)
  - X: A numpy array of shape (N, D) containing a minibatch of data.             # X_dev.shape = (500, 3073)
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means    # y_dev.shape = (500,)
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]  # 10, as example
  num_train = X.shape[0]    # 500, as example

  # normalization trick
  scores = X.dot(W)            # scores.shape = (500, 10), 500 sets of scores for 10 classes
  scores_row_max = np.max(scores, axis=1)                       # (500,)
  scores = scores - scores_row_max.reshape(scores.shape[0], 1)  # (500, 10)

  correct_score = np.exp(scores[np.arange(scores.shape[0]), y])     # (500,)
  exp_scores = np.exp(scores)                                       # (500, 10)
  exp_row_sum = np.sum(exp_scores, axis=1)                          # (500,)
  loss = loss - np.sum(np.log( correct_score / exp_row_sum ))

  ### compute gradient
  exp_scores = exp_scores / exp_row_sum.reshape(exp_row_sum.shape[0], 1)
  exp_scores[np.arange(exp_scores.shape[0]), y] -= 1
  dW = (X.T).dot(exp_scores)

  loss /= num_train
  loss += reg * np.sum(W * W)  # element-wise multiplication

  dW /= num_train              # that's important
  dW += reg * 2 * W            # that's tricky
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

