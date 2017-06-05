import numpy as np
from random import shuffle
#from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
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
  num_classes = W.shape[1]
  num_train = X.shape[0]
  for i in range(num_train):
    si = np.exp(X[i,:].dot(W[:,y[i]]))
    den = 0
    dW[:,y[i]] += -X[i,:]
    for j in range(num_classes):
        den += np.exp(X[i,:].dot(W[:,j]))
    for j in range(num_classes):
        dW[:,j] += (X[i,:]*np.exp(X[i,:].dot(W[:,j]))/den)
    loss += -np.log(si/den)

  loss /=num_train
  loss += reg * np.sum(W * W)

  dW /= num_train
  dW += 2*+reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
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
  num_classes = W.shape[1]
  num_train = X.shape[0]
  S = np.exp(X.dot(W))
  sigS = np.sum(S, axis=1)
  S_i = S[np.arange(num_train),y]
  
  # loss
  loss +=  -np.sum(np.log(S_i/sigS))/num_train
  
  loss += reg * np.sum(W * W)
  
  # 
  div = S.T/sigS
  dW1 = np.dot(X.T,  div.T)
    
  idx = np.zeros((num_train,num_classes))
  idx[np.arange(num_train), y] = 1
  
  dW = -np.dot(X.T,idx)
  dW += dW1
  
  dW /= num_train
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

