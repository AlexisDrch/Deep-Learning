import numpy as np
from random import shuffle

def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  N = X.shape[1] 
  # Compute cross-entropy loss
  z = np.dot(W, X)
  exp_z = np.exp(z - np.max(z, axis = 0) + np.min(z, axis =0)) # for numerical stability
  p = exp_z / np.sum(exp_z, axis = 0) # softmax 
  R = np.sum(W*W) # regularization

  dR = 2*W
  loss = - np.sum(np.log(p[y, np.arange(0, N)]))/ N + reg * R 
  
  p[y, np.arange(0, N)] -= 1
  dW = np.dot(p, X.T)

  dW = dW/N + reg * dR


  return loss, dW
