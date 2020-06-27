import numpy as np

from .base_layer import Layer


class SigmoidLayer(Layer):
    """
    SigmoidLayer: Applies sigmoid function elementwise.
    """

    def __init__(self):
        self._cache_current = None


    def forward(self, x):
        sigmoid_x = 1/(1 + np.exp(-x))
        self._cache_current = sigmoid_x
        return sigmoid_x


    def backward(self, grad_z):
        sigmoid_x_prime = self._cache_current*(1 - self._cache_current)
        return grad_z*sigmoid_x_prime


class TanhLayer(Layer):
    pass


class ReluLayer(Layer):
    """
    ReluLayer: Applies Relu function elementwise.
    """

    def __init__(self):
        self._cache_current = None


    def forward(self, x):
        self._cache_current = x
        return np.maximum(0, x)


    def backward(self, grad_z):
        return grad_z*(self._cache_current > 0)*1.0


class LeakyReluLayer(Layer):
    """
    LeakyReluLayer: Applies LeakyRelu function elementwise.
    
    Arguments:
            alpha {float} -- slope of activation when x < 0 (x >= 0 slope = 1).
    """

    def __init__(self, alpha=0.02):
        self.alpha = alpha
        self._cache_current = None


    def forward(self, x):
        self._cache_current = x
        return np.maximum(self.alpha*x, x)


    def backward(self, grad_z):
        grad_x = np.ones_like(self._cache_current)
        grad_x[self._cache_current < 0] = self.alpha
        return grad_z*grad_x
