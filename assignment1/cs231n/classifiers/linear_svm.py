import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:                                                                          eg. for X_dev, y_dev
  - W: A numpy array of shape (D, C) containing weights.                           eg. W.shape = (3073, 10), init randomly
  - X: A numpy array of shape (N, D) containing a minibatch of data.               eg. X.shape = (500, 3073)
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means      eg. y.shape = (500,), where 0< = y[i] < 10
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape)    # initialize the gradient as zero, (3073, 10)

  # compute the loss and the gradient
  num_classes = W.shape[1]  # 10, as example
  num_train = X.shape[0]    # 500, as example
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:         # if the index is the correct class label's index
        continue
      margin = scores[j] - correct_class_score + 1
      if margin > 0:
        loss += margin
        dW[:,j] += X[i,:].T
        dW[:,y[i]] -= X[i,:].T

#       if j == y[i]:
#         continue
#       margin = scores[j] - correct_class_score + 1
#       if margin > 0:
#         loss += margin

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
    
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)  # element-wise multiplication
    
  # average the gradient dW
  dW /= num_train              # that's important

  # Add regularization to the gradient dW.
  dW += reg * 2 * W            # that's tricky

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.
  Inputs and outputs are the same as svm_loss_naive.
  
  for X_dev and y_dev
  eg. W.shape = (3073, 10), init randomly
  eg. X.shape = (500, 3073)
  eg. y.shape = (500,), where 0< = y[i] < 10
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  ### half-vectorized form
  # delta = 1.0
  # scores = X.dot(W)
  # margins = np.maximum(0, scores - scores[y] + delta)
  # margins[y] = 0
  # loss_i = np.sum(margins)
  # return loss_i

  delta = 1.0
  scores = X.dot(W)                                            # scores.shape = (500, 10), 500 sets of scores for 10 classes
  correct_class_scores = scores[np.arange(scores.shape[0]), y] # integer array indexing as in tutorial, (500,)
  
  # margins.shape = (500, 10)
  margins = np.maximum( np.zeros(scores.shape), scores - correct_class_scores.reshape((scores.shape[0],-1)) + delta )

  # set the correct class value to be zero
  margins[np.arange(scores.shape[0]), y] = 0
  loss += np.sum(margins)
  
  num_train = X.shape[0]    # 500, as example
  loss /= num_train
  loss += reg * np.sum(W * W)  # element-wise multiplication

  ### for gradient
  margins[margins > 0] = 1
  margins[np.arange(scores.shape[0]), y] = 0
  margins[np.arange(scores.shape[0]), y] = -1 * np.sum(margins, axis=1)
  
  dW = (X.T).dot(margins)

  dW /= num_train              # that's important
  dW += reg * 2 * W            # that's tricky

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
    
  return loss, dW
