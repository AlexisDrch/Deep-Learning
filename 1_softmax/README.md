http://ripl.cc.gatech.edu/classes/AY2019/cs7643_spring/hw0-q8/
 
### Softmax Classifier
I implemented a Softmax classifier using no machine learning libraries. This classifier will be trained on the CIFAR10 dataset.
This project included the implementations of a vectorized loss function, gradient computation and the Stochastic Gradient Descent optimization algorithm.

#### 1 Stochastic Gradient Descent

Softmax regression uses the cross-entropy loss L and a regularization term R to penalize high-
valued weights. The final loss Ltot is the addition of: $L=−1 Nlog(pyi)andR(W)=   W2
 Ni i klk,l$

Note that the regularization term is multiplied by a hyper-parameter constant $θ$ (equal to 0 for the gradient check as computed in Figure 1).
The gradient $∇W$ Ltot = ∇W L + θ∇W R.

#### 2  Cross-entropy loss minimization

![alt text](https://github.com/AlexisDrch/Deep-Learning/blob/master/hw0/output/loss_plot.png)

The gradient $∇W Ltot = ∇W L + ∇W R.$ 

The gradient is used in the Stochastic Gradient De- scent (SGD) to update the matrix of parameters W in order to minimize iteratively the loss function (see Figure 2). Since the softmax loss is a convex function, it converges toward a global minimum, given appropriate hyper parameters.

Softmax computations may lead to numerical instabilities because of the normalization by exponential numbers. To cope with this issue, I restrained the zi values by adding a constant term αi = −max(zi) + min(zi) to the zi values. Note that adding constant terms to the zi values don’t impact the softmax outputs pi (ref [1] for the proof of mathematical consistency).
To fine-tune the model, I used the validation set to find the best regularization parameter θ and learning rate lr via a grid search over those two hyper parameters. I noticed that the distribution of the validation and test datasets were not very correlated since better accuracy (fine-tuned parameters) on the Training dataset didn’t necessarily yield better test accuracy nor visuliaztion. However, the grid search still helped reducing the possibilities for those pa- rameters and find a correct trade-off. Also, I reduced the batch size from 200 to 64 in order to proportionally increase the number of SGD iterations per epochs (N/batchsize).

The final accuracy was around 30 %. This result is poor but we could expect the linear classifier to perform poorly regarding the very likely non-linearity of this classification prob- lem, and the high-dimension of our data.

#### 3 Weight visualization

![alt text](https://github.com/AlexisDrch/Deep-Learning/blob/master/hw0/output/weights_viz.png)

Once the classifier is trained, the weights are not updated anymore. We can visualize each weight "layers" (i.e each row Wj) per class j.

It is interesting to notice shape and color patterns in the weights layer corresponding to the class their are related to. For the ship class for instance, we can observe an average blue and grey/white color corresponding to an average ship shape surrounded by water. For the frog class, weights have an average green/brown color and a pattern similar to an average frog’s shape can be detected. Similar observations can be drawn for each other weight layers and their respective class.
This observation is a proof that the classifier learned the appropriate patterns to recognize the different class w.r.t to the training dataset. However, the low accuracy of the model indicates that this model clearly over-fitted on the training samples and thus, didn’t generalize to edge cases (i.e a bird surronded by water could be predicted as a plane or ship using this classifier).

#### Interpretation
Single layer, linear Neural Net has low accuracy on apparently non-linear multi-class problem.
Next work: add layers, non-linearity (reLu), convolutional filters to extract state-of-the-art image data features.
