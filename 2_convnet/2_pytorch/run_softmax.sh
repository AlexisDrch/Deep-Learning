#!/bin/sh
#############################################################################
# TODO: Modify the hyperparameters such as hidden layer dimensionality, 
#       number of epochs, weigh decay factor, momentum, batch size, learning 
#       rate mentioned here to achieve good performance
#############################################################################
python -u train.py \
    --model softmax \
    --hidden-dim 300 \
    --epochs 2 \
    --weight-decay 0.001 \
    --momentum 0.9 \
    --batch-size 200 \
    --lr 0.0001 | tee softmax.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
