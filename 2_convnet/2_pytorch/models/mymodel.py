import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
python -u train.py     --model mymodel     --kernel-size 3     --hidden-dim 32     --epochs 10    \
--weight-decay 0.0001     --momentum 0.9     --batch-size 50     --lr 0.0001 | tee mymodel.log
"""


class MyModel(nn.Module):
    def __init__(self, im_size, params_cnn, kernel_sizes, n_classes, p=0.2):
        '''
        Create components of a CNN classifier and initialize their weights.

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            n_classes (int): Number of classes to score
        '''
        

        super(MyModel, self).__init__()
        
        C, H, W = im_size
        pool_stride, pool_pad, pool_size = 2,0,kernel_sizes[2]
        for i in range(len(params_cnn)-1):
            setattr(self, 'conv_{}'.format(str(i+1)), self.conv_layer(params_cnn[i], params_cnn[i+1],
                                                             kernel_sizes[i], 1, True))
            if i < len(params_cnn)-3:
                H = (1+(H+2*pool_pad-pool_size)//pool_stride)
            print(H)
        in_shape = int(params_cnn[-1]*H**2)
        self.pool = nn.MaxPool2d(pool_size, stride=pool_stride, padding=pool_pad)
        self.linear_1 = nn.Linear(in_shape, in_shape, bias=True)
        self.linear_2 = nn.Linear(in_shape, in_shape//2, bias=True)
        self.linear_3 = nn.Linear(in_shape//2, n_classes, bias=True)
        self.relu = nn.ReLU()
        self.dropout_c = nn.Dropout(p=p)
        self.dropout_l = nn.Dropout(p=2*p)
    
    def conv_layer(self, inp, out, kernel_size, stride, bn=False):
        layers = []
        layers.append(nn.Conv2d(inp, out, kernel_size, stride, padding=(kernel_size-1)//2))

        if bn:
            layers.append(nn.BatchNorm2d(out))
        layers.append(nn.ReLU())
        return nn.Sequential(*layers)

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
        X = self.conv_1(images)
        X = self.dropout_c(X)
        X = self.pool(X)
        X = self.conv_2(X)
        X = self.dropout_c(X)
        X = self.pool(X)
        X = self.conv_3(X)
        X = self.dropout_c(X)
        X = self.pool(X)
        X = self.conv_4(X)
        X = self.dropout_c(X)
        X = self.conv_5(X)
        X = X.view(N, -1)
        X = self.linear_1(X)
        X = self.relu(X)
        X = self.dropout_l(X)
        X = self.linear_2(X)
        X = self.relu(X)
        X = self.dropout_l(X)
        X = self.linear_3(X)
        return X



