import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from nn_library import Linear, ReLU, CELoss, SGD, xavier_init


class MultiLayerPerceptron(object):
    """
    MultiLayerPerceptron: A network consisting of stacked linear layers and
    activation functions.
    """

    def __init__(self):
        self.fc1 = Linear(4, 64)
        self.act1 = ReLU()
        self.fc2 = Linear(64, 3)
        self.layers = [self.fc1, self.act1, self.fc2]

        self.init_func = xavier_init
        self.loss_function = CELoss()
        self.optimizer = SGD(lr=0.01)

        for layer in self.layers:
            if callable(getattr(layer, "step", None)):
                layer.optimizer = self.optimizer
                layer.init_func = self.init_func
                layer.init_paramaters()


    def forward(self, x):#forwardpass trough entire network 
        """
        Performs forward pass through the network.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, input_dim).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size,
                #_neurons_in_final_layer)
        """

        return self.fc2(self.act1(self.fc1(x)))

    
    def __call__(self, x):
        return self.forward(x)


    def backward(self, grad_z):
        """
        Performs backward pass through the network.

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (1,
                #_neurons_in_final_layer).

        Returns:
            {np.ndarray} -- Array containing gradient with repect to layer
                input, of shape (batch_size, input_dim).
        """
        for layer in self.layers[::-1]:
            grad_z = layer.backward(grad_z)
        return grad_z


    def step(self):
        """
        Performs one step of gradient descent with given learning rate on the
        parameters of all layers using currently stored gradients.

        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """
        for layer in self.layers:
            if callable(getattr(layer, "step", None)):
                layer.step()
            else:
                continue


if __name__ == "__main__":
    MultiLayerPerceptron()