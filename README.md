# ObjectNet
Note I have cloned the [official PyTorch](https://github.com/hangzhaomit/semantic-segmentation-pytorch) implementation and just added a function to merge most common classes to reduce number of classes from 150 to 25.
I could not address this modification in the title of the repo.

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


## Dilation
Dilated Residual Networks [link to paper](https://arxiv.org/pdf/1705.09914.pdf)

We show that dilated residual networks
(DRNs) outperform their non-dilated counterparts in image
classification without increasing the model’s depth or
complexity. We then study gridding artifacts introduced by
dilation, develop an approach to removing these artifacts
(‘degridding’), and show that this further increases the performance
of DRNs. In addition, we show that the accuracy
advantage of DRNs is further magnified in downstream applications
such as object localization and semantic segmentation.

While convolutional networks have done well, the almost
complete elimination of spatial acuity may be preventing
these models from achieving even higher accuracy, for
example by preserving the contribution of small and thin
objects that may be important for correctly understanding
the image.

![converting resnet to dilated resnet](wiki/img/dilation.jpg)

The use of dilated convolutions can cause gridding artifacts.

![artifacts of dilation](wiki/img/artifacts.jpg)

So they introduced three methods to remove this artifacts. Here is the structure of them:

![DRN-A, DRN-B and DRN-C](wiki/img/drna-drnb-drnc.jpg)

Here is comparison of differenet dilated resnets based on error rates

![error rate of dilated resnet and original resnet](wiki/img/error-rate-dilation.jpg)

And you can see the accuracy of this model in semantic segmentation on cityscapes dataset:

![accuracy in semantic segmentation on cityscapes dataset](wiki/img/semantic-acc-dilated.jpg)

A sample comparison:

![output on real world dilated resnet semantic segmentation](wiki/img/semantic-sample.jpg)


## Supervision
Training Deeper Convolutional Networks with Deep Supervision [link to paper](https://arxiv.org/pdf/1505.02496.pdf)

In order to train deeper networks, we propose to
add auxiliary supervision branches after certain intermediate
layers during training. We formulate a simple rule of
thumb to determine where these branches should be added.
The resulting deeply supervised structure makes the training
much easier and also produces better classification results
on ImageNet and the recently released, larger MIT
Places dataset.

Illustration of our deep models with 8 and 13 convolutional layers. The additional supervision loss branches
are indicated by dashed red boxes. Xl denote the intermediate layer outputs and Wl are the weight matrices for each
computational block. Blocks of the same type are shown in the same color. A legend below the network diagrams shows the
internal structure of the different block types.

![supervision structure](wiki/img/supervision.jpg)

Loss functions:

auxiliary loss
 
![loss of supervision](wiki/img/loss-supervision.jpg)

Note that this loss depends on W, not just Ws, because the
computation of the feature map S8 involves the weights of
the early convolutional layers W1; : : :W4.
The combined loss function for the whole network is
given by a weighted sum of the main loss L0(W) and the
auxiliary supervision loss Ls(Ws):

![overall loss of model with supervision](wiki/img/overall-loss-supervision.jpg)

where alpha-t controls the trade-off between the two terms. In
the course of training, in order to use the second term
mainly as regularization, we adopt the same strategy as
in [6], where alpha decays as a function of epoch t (with N
being the total number of epochs):

![regularization of supervision](wiki/img/regularization-supervision.jpg)

We train our deeply supervised model using stochastic
gradient descent.

Here is top-1 and top-5 accuracies on places dataset:

![top-1 and top-5 accuracies on places dataset](wiki/img/top-accuracies-on-places-supervision.jpg)


## Pyramid Pooling
Pyramid Scene Parsing Network [link to paper](https://arxiv.org/pdf/1612.01105.pdf)<br>
[Link](https://github.com/hszhao/PSPNet) to implementation repository.

In this paper, we exploit the
capability of global context information by different-regionbased
context aggregation through our pyramid pooling
module together with the proposed pyramid scene parsing
network (PSPNet).

Our main contributions are threefold.

- We propose a pyramid scene parsing network to embed
difficult scenery context features in an FCN based
pixel prediction framework.
- We develop an effective optimization strategy for deep
ResNet based on deeply supervised loss.
- We build a practical system for state-of-the-art scene
parsing and semantic segmentation where all crucial
implementation details are included.

Scene parsing issues:

![scene parsing issues-pyramid](wiki/img/scene-parsing-issues-pyramid.jpg)


feature maps in different levels generated by
pyramid pooling were finally flattened and concatenated to
be fed into a fully connected layer for classification. This
global prior is designed to remove the fixed-size constraint
of CNN for image classification. To further reduce context
information loss between different sub-regions, we propose
a hierarchical global prior, containing information with different
scales and varying among different sub-regions.
we call it pyramid pooling module for global scene prior construction
upon the final-layer-feature-map of the deep neural
network.


Here is the PSPNet structure:

![pspnet structure](wiki/img/pspnet-structure.jpg)

The pyramid pooling module fuses features under four
different pyramid scales. The coarsest level highlighted in
red is global pooling to generate a single bin output. The
following pyramid level separates the feature map into different
sub-regions and forms pooled representation for different
locations. The output of different levels in the pyramid
pooling module contains the feature map with varied
sizes. To maintain the weight of global feature, we use 1x1
convolution layer after each pyramid level to reduce the dimension
of context representation to 1/N of the original
one if the level size of pyramid is N. Then we directly upsample
the low-dimension feature maps to get the same size
feature as the original feature map via bilinear interpolation.
Finally, different levels of features are concatenated as the
final pyramid pooling global feature.

You can see the comparison of PSPNet with other models:

![pspnet comparison](wiki/img/pspnet-comparison.jpg)

Also there is a result on ImageNet dataset:

![psp on imgenet](wiki/img/psp-on-imgenet.jpg)

Example of PSPNet"

![example of PSPNet](wiki/img/example-psp.jpg)

The result of PSPNet and other methods on CityScapes dataset in semantic segmentation:

![psp-semantic-cityscapes](wiki/img/psp-semantic-cityscapes.jpg)


# Reference
Repository of models [link](https://github.com/CSAILVision/sceneparsing)

Repository of pytorch implementation [link](https://github.com/hangzhaomit/semantic-segmentation-pytorch)

# Citation
Please cite this porject as:
Nikan Doosti. (2020). Nikronic/ObjectNet: DOI Release (v0.1-alpha). Zenodo. https://doi.org/10.5281/zenodo.3838620

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3838620.svg)](https://doi.org/10.5281/zenodo.3838620)
