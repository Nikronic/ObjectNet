# ObjectNet

## ResNet101
Deep Residual Learning for Image Recognition [link to paper](https://arxiv.org/pdf/1512.03385.pdf)

![resblock-fig2](wiki/img/resblock.jpg "res block")

we address the degradation problem by
introducing a deep residual learning framework. Instead
of hoping each few stacked layers directly fit a
desired underlying mapping, we explicitly let these layers
fit a residual mapping. Formally, denoting the desired
underlying mapping as **H(x)**, we let the stacked nonlinear
layers fit another mapping of **F(x) := H(x)-x**. The original
mapping is recast into **F(x)+x**. We hypothesize that it
is easier to optimize the residual mapping than to optimize
the original, unreferenced mapping. To the extreme, if an
identity mapping were optimal, it would be easier to push
the residual to zero than to fit an identity mapping by a stack
of nonlinear layers.<br>
In our case, the shortcut connections simply
perform identity mapping, and their outputs are added to
the outputs of the stacked layers.<br><br>
In the paper they show that:
1) Our extremely deep residual nets
are easy to optimize, but the counterpart “plain” nets (that
simply stack layers) exhibit higher training error when the
depth increases. 
2) Our deep residual nets can easily enjoy
accuracy gains from greatly increased depth, producing results
substantially better than previous networks.

Here is implementation structure they used in theirs paper
![compare-res-plain](wiki/img/compare-res.jpg)

And they got this top-1 error rates:
![acc-res-plain](wiki/img/resplainacc.jpg)

Because we are using resnet101, residual blocks constructed over Bottleneck block. Here is the structure:
![bottleneck](wiki/img/bottleneck.jpg)


## Pyramid Pooling
Pyramid Scene Parsing Network [link to paper](https://arxiv.org/pdf/1612.01105.pdf)


## Supervision
Training Deeper Convolutional Networks with Deep Supervision [link to paper](https://arxiv.org/pdf/1505.02496.pdf)


## Dilation
Dilated Residual Networks [link to paper](https://arxiv.org/pdf/1705.09914.pdf)



# Reference
Repository of models [link](https://github.com/CSAILVision/sceneparsing)

Repository of pytorch implementation [link](https://github.com/hangzhaomit/semantic-segmentation-pytorch)
