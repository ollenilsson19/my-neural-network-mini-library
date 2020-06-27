import numpy as np

from base import Layer


import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True):

        super(Conv2d, self).__init__()
        """
        An implementation of a convolutional layer.

        The input consists of N data points, each with C channels, height H and
        width W. We convolve each input with F different filters, where each filter
        spans all C channels and has height HH and width WW.

        Parameters:
        - w: Filter weights of shape (F, C, HH, WW)
        - b: Biases, of shape (F,)
        - kernel_size: Size of the convolving kernel
        - stride: The number of pixels between adjacent receptive fields in the
            horizontal and vertical directions.
        - padding: The number of pixels that will be used to zero-pad the input.
        """

        ########################################################################
        # TODO: Define the parameters used in the forward pass                 #
        ########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self.out_channels = out_channels
        self.in_channels = in_channels
        if isinstance(kernel_size, int) == 1:
          self.HH = kernel_size
          self.WW = kernel_size
        else:
          self.HH = kernel_size[0]
          self.WW = kernel_size[1]

        if isinstance(stride, int) == 1:
          self.H_stride = stride
          self.W_stride = stride
        else:
          self.H_stride = stride[0]
          self.W_stride = stride[1]

        if isinstance(padding, int) == 1:
          self.H_padding = padding
          self.W_padding = padding
        else:
          self.H_padding = padding[0]
          self.W_padding = padding[1]

        self.use_bias = bias

        self.w = torch.empty(out_channels, in_channels, self.HH, self.WW, requires_grad=True)
        nn.init.kaiming_uniform_(self.w)

        if self.use_bias:
          self.b = torch.zeros(out_channels, requires_grad=True)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    def forward(self, x):
        """
        Input:
        - x: Input data of shape (N, C, H, W)
        Output:
        - out: Output data, of shape (N, F, H', W').
        """

        ########################################################################
        # TODO: Implement the forward pass                                     #
        ########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        N, C, H, W = x.shape
        if self.H_padding or self.W_padding:
          start_index_H = self.H_padding
          start_index_W = self.W_padding
          end_index_H = self.H_padding + H
          end_index_W = self.W_padding + W
          padded = torch.zeros(N, C, H+2*self.H_padding, W+2*self.W_padding)
          padded[:,:,start_index_H:end_index_H, start_index_W:end_index_W] = x
          x = padded
          
        H_out = int((H - self.HH + 2*self.H_padding)/self.H_stride + 1)
        W_out = int((W - self.WW + 2*self.W_padding)/self.W_stride + 1)

        unfold_row = x.unfold(2, self.HH, self.H_stride)
        unfold_col = unfold_row.unfold(3, self.WW, self.W_stride)
        x_folds_as_rows =\
         torch.cat(unfold_col.reshape(N,-1, self.HH*self.WW).split(H_out*W_out, dim=1), dim=2)
        x_folds_as_cols = x_folds_as_rows.permute(0, 2, 1)#batch_transpose

        w_as_rows = self.w.view(self.out_channels,-1)
        out = (w_as_rows@x_folds_as_cols).view(N, self.out_channels, -1)
        if self.use_bias:
          out += self.b.view(self.out_channels, 1)
        out = out.view(N, self.out_channels, H_out, W_out)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return out
#TEST
N = 100
in_channels = 64
out_channels = 128
kernel_size = (2, 4)
stride = (1,2)
padding = (2,1)
input_size = 8

test = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
x = torch.rand(N, in_channels, input_size, input_size)

conv1 =nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
conv1.weight.data = test.w
conv1.bias.data = test.b

print("shapes match?: ", conv1(x).shape, test(x).shape)
print("outputs are the same?: ", torch.allclose(test(x), conv1(x), rtol=1e-05, atol=1e-05))


class MaxPool2d(nn.Module):
    def __init__(self, kernel_size):
        super(MaxPool2d, self).__init__()
        """
        An implementation of a max-pooling layer.

        Parameters:
        - kernel_size: the size of the window to take a max over
        """
        ########################################################################
        # TODO: Define the parameters used in the forward pass                 #
        ########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        if isinstance(kernel_size, int):
          self.HH = kernel_size
          self.WW = kernel_size
        else: 
          self.HH = kernel_size[0]
          self.WW = kernel_size[1]

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    def forward(self, x):
        """
        Input:
        - x: Input data of shape (N, C, H, W)
        Output:
        - out: Output data, of shape (N, C, H', W').
        """
        ########################################################################
        # TODO: Implement the forward pass                                     #
        ########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        N, C, H, W = x.shape
        H_out = int(H/self.HH)
        W_out = int(W/self.WW)
        unfold_row = x.unfold(2, self.HH, self.HH).unfold(3, self.WW, self.WW)
        x_ = unfold_row.reshape(N, C, -1, self.HH*self.WW)
        out = x_.max(dim=3)[0].view(N, C, H_out, W_out)  
    
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return out

#TEST
N = 128
in_channels = 64
kernel_size = (4, 4)
input_size = 32

test = MaxPool2d(kernel_size=kernel_size)
x = torch.rand(N, in_channels, input_size, input_size)

pool=nn.MaxPool2d(kernel_size=kernel_size)

print("shapes match?: ", pool(x).shape, test(x).shape)
print("outputs are the same?: ", torch.allclose(test(x), pool(x), rtol=1e-05, atol=1e-05))


class BatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1):
        super(BatchNorm2d, self).__init__()
        """
        An implementation of a Batch Normalization over a mini-batch of 2D inputs.

        The mean and standard-deviation are calculated per-dimension over the
        mini-batches and gamma and beta are learnable parameter vectors of
        size num_features.

        Parameters:
        - num_features: C from an expected input of size (N, C, H, W).
        - eps: a value added to the denominator for numerical stability. Default: 1e-5
        - momentum: momentum – the value used for the running_mean and running_var
        computation. Default: 0.1
        - gamma: the learnable weights of shape (num_features).
        - beta: the learnable bias of the module of shape (num_features).
        """
        ########################################################################
        # TODO: Define the parameters used in the forward pass                 #
        ########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.gamma = torch.ones(num_features, requires_grad=True)
        self.beta = torch.zeros(num_features, requires_grad=True)
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    def forward(self, x):
        """
        During training this layer keeps running estimates of its computed mean and
        variance, which are then used for normalization during evaluation.
        Input:
        - x: Input data of shape (N, C, H, W)
        Output:
        - out: Output data of shape (N, C, H, W) (same shape as input)
        """
        ########################################################################
        # TODO: Implement the forward pass                                     #
        #       (be aware of the difference for training and testing)          #
        ########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        N, C, H, W = x.shape
        if self.training:#traning mode
          mean = torch.mean(x, dim=(0, 2, 3), keepdim=True)
          variance = torch.var(x, dim=(0, 2, 3), unbiased=False, keepdim=True)

          self.running_mean = (1-self.momentum)*self.running_mean.view_as(mean)\
                                                         + (self.momentum*mean) 
          self.running_var = (1-self.momentum)*self.running_var.view_as(variance)\
                                      + self.momentum*variance*N*H*W/(N*H*W - 1)
        else:#eval mode
          mean = self.running_mean
          variance = self.running_var
        x_norm = ((x - mean)/(variance + self.eps)**0.5).view(N, C, H*W)
        x_scaled_and_shifted =  x_norm*self.gamma.view(self.num_features, 1)\
                                       + self.beta.view(self.num_features, 1)
        x = x_scaled_and_shifted.view(N, C, H, W)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return x

#TEST
N = 128
in_channels = 64
input_size = 4


test = BatchNorm2d(num_features=in_channels, eps=1e-05, momentum=0.1)
x = torch.rand(N, in_channels, input_size, input_size)
batch =nn.BatchNorm2d(num_features=in_channels, eps=1e-05, momentum=0.1)
res_test = test(x)
res_test = test(x)
res_torch = batch(x)
res_torch = batch(x)
print("shapes match?: ", res_test.shape, res_torch.shape)
print("outputs are the same?: ", torch.allclose(res_test, res_torch, rtol=1e-05, atol=1e-05))

test.eval()
batch.eval()
print("shapes match?: ", batch(x).shape, test(x).shape)
print("outputs are the same?: ", torch.allclose(test(x), batch(x), rtol=1e-05, atol=1e-05))