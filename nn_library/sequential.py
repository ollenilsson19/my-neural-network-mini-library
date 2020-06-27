
class Sequential(object):
    """
    MultiLayerPerceptron: A network consisting of stacked linear layers and
    activation functions.
    """

    def __init__(self, *layers):
        """
        Arguments:
            input_dim {int} -- Dimension of input (excluding batch dimension).
            neurons {list} -- Number of neurons in each layer represented as a
                list (the length of the list determines the number of layers).
            activations {list} -- List of the activation function to use for
                each layer.
        """
        self._layers = layers


    def forward(self, x):
        """
        Performs forward pass through the network.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, input_dim).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size,
                #_neurons_in_final_layer)
        """

        for layer in self._layers:
            x = layer.forward(x)
        return(x)

    
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
        for layer in self._layers[::-1]:
            grad_z = layer.backward(grad_z)
        return grad_z


    def update_params(self, learning_rate):
        """
        Performs one step of gradient descent with given learning rate on the
        parameters of all layers using currently stored gradients.

        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """
        for layer in self._layers:
            if isinstance(layer, LinearLayer):
                layer.update_params(learning_rate)
            else:
                continue
