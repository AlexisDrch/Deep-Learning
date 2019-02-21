import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
"""
python -u train.py --model twolayernn --hidden-dim 300 --epochs 15 \
--weight-decay 0.001 --momentum 0.9 --batch-size 200 --lr 0.0001 | tee twolayernn.log"""

class TwoLayerNN(nn.Module):
    def __init__(self, im_size, hidden_dim, n_classes):
        '''
        Create components of a two layer neural net classifier (often
        referred to as an MLP) and initialize their weights.

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            n_classes (int): Number of classes to score
        '''
        super(TwoLayerNN, self).__init__()
        C, H, N = im_size
        # initialized with default uniform U(-sqr(k), sqrt(k))
        self.linear_1 = nn.Linear(C*H*N, hidden_dim, bias=True)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(hidden_dim, n_classes, bias=True)
        

    def forward(self, images):
        '''
        Take a batch of images and run them through the NN to
        produce a score for each class.

        Arguments:
            images (Variable): A tensor of size (N, C, H, W) where
                N is the batch size
                C is the number of channels
                H is the image height
                W is the image width

        Returns:
            A torch Variable of size (N, n_classes) specifying the score
            for each example and category.
        '''
        N, C, H, W = images.shape
        X = images.view(N, -1)
        X = self.linear_1(X)
        X = self.relu(X)
        scores = self.linear_2(X)
        return scores

