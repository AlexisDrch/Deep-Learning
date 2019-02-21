import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
"""
python -u train.py --model softmax --hidden-dim 300 --epochs 2 \
--weight-decay 0.001 --momentum 0.9 --batch-size 200 --lr 0.0001 | tee softmax.log"""
class Softmax(nn.Module):
    def __init__(self, im_size, n_classes):
        '''
        Create components of a softmax classifier and initialize their weights.

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            n_classes (int): Number of classes to score
        '''
        super(Softmax, self).__init__()
        C, H, W = im_size
        self.linear = nn.Linear(C*H*W, n_classes, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, images):
        '''
        Take a batch of images and run them through the classifier to
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
        scores = self.linear(X)
        scores = self.softmax(scores)
        return scores

