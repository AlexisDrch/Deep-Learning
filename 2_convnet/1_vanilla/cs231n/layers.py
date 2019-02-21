import numpy as np

def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
  We multiply this against a weight matrix of shape (D, M) where
  D = \prod_i d_i

  Inputs:
  x - Input data, of shape (N, d_1, ..., d_k)
  w - Weights, of shape (D, M)
  b - Biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  N, D = x.shape[0], np.prod(x.shape[1:])
  x_ = x.reshape(N, D)

  out = np.dot(x_, w) + b
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  x_ = x.reshape(x.shape[0], w.shape[0]) # (N, M)
  dx = np.dot(dout, w.T).reshape(x.shape) # (N, d_1, .., d_k)
  dw = np.dot(x_.T, dout)
  db = np.sum(dout, axis = 0)
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = np.maximum(x, 0)
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None , cache
  dout[x<=0] = 0
  dx = dout
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  stride, pad = conv_param['stride'], conv_param['pad']
  x_padded = np.pad(x, [(0,0), (0,0), (pad,pad),(pad,pad)], mode='constant', constant_values=0)
  (N, C, H, W) = x.shape
  (F, C, HH, WW) = w.shape
  H_, W_ = int(1+(H + 2*pad - HH)/stride), int(1+(W + 2*pad - WW)/stride)
  out = np.zeros((N, F, H_, W_))

  for i in range(N): # per samples
    for j in range(F): # per filter
        for r in np.arange(0, H_):
            rs = r*stride
            for c in np.arange(0, W_):
                cs = c*stride
                # input patch extraction
                patch = x_padded[i, :, rs:rs+HH, cs:cs+WW]
                # convolution
                out[i,j,r,c] = np.sum(patch*w[j]) + b[j] 
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  x, w, b, conv_param = cache
  stride, pad = conv_param['stride'], conv_param['pad']
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape
  H_, W_ = int(1+ (H + 2*pad - HH)/stride), int(1+ (W + 2*pad - WW)/stride)

  dw, dx, db = np.zeros(w.shape), np.zeros(x.shape), np.zeros(b.shape)
  x_padded = np.pad(x, [(0,0),(0,0),(pad,pad),(pad,pad)], mode='constant', constant_values= 0)
  dx_padded = np.pad(dx, [(0,0),(0,0),(pad,pad),(pad,pad)], mode = 'constant', constant_values=0)

  for i in range(N):
    for j in range(F):
      for r in range(H_):
        rs = r*stride
        for c in range(W_):
          cs = c*stride
          # extract input patch 
          patch = x_padded[i, :, rs:rs+HH, cs:cs+WW]

          # gradient of out[i,j,r,c] = patch * w[j] + b[j]
          dw[j] += dout[i,j,r,c] * patch
          db[j] += dout[i,j,r,c]
          dx_padded[i,:,rs:rs+HH,cs:cs+WW] += dout[i,j,r,c] * w[j]
  
  # crop padded 
  dx = dx_padded[:,:,pad:-pad, pad:-pad]
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  N, C, H, W = x.shape
  HH, WW, stride = pool_param["pool_height"], pool_param["pool_width"], pool_param["stride"]
  H_, W_ = int(1 + (H-HH)/stride), int(1 + (W-WW)/stride)
  out = np.zeros((N, C, H_, W_))
  
  for i in range(N): # per samples
    for r in range(H_):
      rs = r*stride
      for c in range(W_):
        cs = c*stride
        # extract pool region
        patch = x[i,:,rs:rs+HH,cs:cs+WW]
        # max on h and w axis
        out[i,:,r,c] = np.max(patch, axis=(1,2))

  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  x, pool_param = cache
  N, C, H, W = x.shape
  HH, WW, stride = pool_param["pool_height"], pool_param["pool_width"], pool_param["stride"]
  H_, W_ = int(1 + (H-HH)/stride), int(1 + (W-WW)/stride)
  
  dx = np.zeros(x.shape)
  for i in range(N): # per samples
    for j in range(C): # per channels to get max
      for r in range(H_):
        rs = r*stride
        for c in range(W_):
          cs = c*stride
          # extract pool region
          patch = x[i,j,rs:rs+HH,cs:cs+WW]
          max_ = np.max(patch)
          # gradient of max(x) is null if x!= max
          dx[i,j,rs:rs+HH, cs:cs+WW] += dout[i,j,r,c] * (patch == max_)
    
  return dx


def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx

