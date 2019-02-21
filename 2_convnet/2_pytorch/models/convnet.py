import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
"""
python -u train.py     --model convnet     --kernel-size 3     --hidden-dim 32     --epochs 10    \
--weight-decay 0.0001     --momentum 0.9     --batch-size 50     --lr 0.0001 | tee convnet.log
"""

class CNN(nn.Module):
    def __init__(self, im_size, hidden_dim, kernel_size, n_classes):
        '''
        Create components of a CNN classifier and initialize their weights.

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            n_classes (int): Number of classes to score
        '''
        super(CNN, self).__init__()
        C, H, W = im_size
        # required to keep image size through convolution
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(C, hidden_dim, kernel_size, stride=1, padding=padding, bias=True)
        self.relu = nn.ReLU()
        # parameters from Vanilla convnet
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.linear = nn.Linear((1+(H+2*0-2)//2)**2*hidden_dim, n_classes)

    def forward(self, images):
        '''
        Take a batch of images and run them through the CNN to
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
        N, _, _, _ = images.shape
        X = self.conv(images)
        X = self.relu(X)
        X = self.pool(X)
        X = X.view(N, -1)
        X = self.linear(X)
        return X

