import numpy as np

from .base_layer import Layer
from .parameter_init import xavier_init


class LinearLayer(Layer):
    """
    LinearLayer: Performs affine transformation of input.
    """

    def __init__(self, n_in, n_out):
        """Constructor.

        Arguments:
            n_in {int} -- Number (or dimension) of inputs.
            n_out {int} -- Number (or dimension) of outputs.
        """
        self.n_in = n_in
        self.n_out = n_out

        self.optimizer = None
        self.init_func = xavier_init

        self._cache_current = None
        self._grad_W_current = None
        self._grad_b_current = None


    def init_paramaters(self):
        self._W = self.init_func((self.n_in, self.n_out))
        self._b = self.init_func((1, self.n_out))


    def forward(self, x):#OK
        """
        Performs forward pass through the layer (i.e. returns Wx + b).

        Logs information needed to compute gradient at a later stage in
        `_cache_current`.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out)
        """
        self.batch_size = len(x)
        self._cache_current = x
        return x@self._W + self._b


    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out).

        Returns:
            {np.ndarray} -- Array containing gradient with repect to layer
                input, of shape (batch_size, n_in).
        """

        self._grad_W_current = self._cache_current.T@grad_z 
        self._grad_b_current = np.ones((1, self.batch_size))@grad_z
        return grad_z@self._W.T


    def step(self):
        """
        Performs one step of gradient descent with given learning rate on the
        layer's parameters using currently stored gradients.

        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """
        self._W += self.optimizer(self._grad_W_current)
        self._b += self.optimizer(self._grad_b_current)


if __name__ == "__main__":
    LinearLayer(5,5)
