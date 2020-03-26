import pickle
import numpy as np

from neural_network_library import MSELossLayer, CrossEntropyLossLayer



class Trainer(object):
    """
    Trainer: Object that manages the training of a neural network.
    """

    def __init__(
        self,
        network,
        batch_size,
        nb_epoch,
        learning_rate,
        loss_fun,
        shuffle_flag,
    ):
        """Constructor.

        Arguments:
            network {MultiLayerNetwork} -- MultiLayerNetwork to be trained.
            batch_size {int} -- Training batch size.
            nb_epoch {int} -- Number of training epochs.
            learning_rate {float} -- SGD learning rate to be used in training.
            loss_fun {str} -- Loss function to be used. Possible values: mse,
                bce.
            shuffle_flag {bool} -- If True, training data is shuffled before
                training.
        """
        self.network = network
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.loss_fun = loss_fun
        self.shuffle_flag = shuffle_flag

        if self.loss_fun == 'mse':
            self._loss_layer = MSELossLayer()
        elif self.loss_fun == 'cross_entropy':
            self._loss_layer = CrossEntropyLossLayer()


    @staticmethod
    def shuffle(input_dataset, target_dataset):
        """
        Returns shuffled versions of the inputs.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_data_points, ).

        Returns: 2-tuple of np.ndarray: (shuffled inputs, shuffled_targets).
        """

        permutation = np.random.permutation(len(input_dataset))
        return(input_dataset[permutation], target_dataset[permutation])
 

    def train(self, input_dataset, target_dataset):
        """
        Main training loop. Performs the following steps `nb_epoch` times:
            - Shuffles the input data (if `shuffle` is True)
            - Splits the dataset into batches of size `batch_size`.
            - For each batch:
                - Performs forward pass through the network given the current
                batch of inputs.
                - Computes loss.
                - Performs backward pass to compute gradients of loss with
                respect to parameters of network.
                - Performs one step of gradient descent on the network
                parameters.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_training_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_training_data_points, ).
        """

        for epoch in range(self.nb_epoch):
            if self.shuffle_flag:
                input_dataset, target_dataset =\
                self.shuffle(input_dataset, target_dataset)
            # split into batches
            n_batches = len(input_dataset)//self.batch_size
            end = n_batches*self.batch_size
            input_batches = np.split(input_dataset[:end], n_batches)
            target_batches = np.split(target_dataset[:end], n_batches)
            if len(input_dataset)%self.batch_size != 0:
                input_batches = input_batches + [input_dataset[end:]]
                target_batches = target_batches + [target_dataset[end:]]

            for batch in range(len(input_batches)):
                network_output = self.network(input_batches[batch])
                loss = self._loss_layer.forward(network_output, target_batches[batch])
                grad_z = self._loss_layer.backward()
                self.network.backward(grad_z)
                self.network.update_params(self.learning_rate)
            print(f"Train loss epoch {epoch} = ", loss)

    def eval_loss(self, input_dataset, target_dataset):
        """
        Function that evaluate the loss function for given data.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_evaluation_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_evaluation_data_points, ).
        """

        network_output = self.network(input_dataset)
        loss = self._loss_layer.forward(network_output, target_dataset)
        return loss


class Preprocessor(object):
    """
    Preprocessor: Object used to apply "preprocessing" operation to datasets.
    The object can also be used to revert the changes.
    """

    def __init__(self, data):
        """
        Initializes the Preprocessor according to the provided dataset.
        (Does not modify the dataset.)

        Arguments:
            - data {np.ndarray} dataset used to determined the parameters for
            the normalization.
        """
        self.minumim_values = np.amin(data, axis=0)
        self.maximum_values = np.amax(data, axis=0)


    def apply(self, data):
        """
        Apply the pre-processing operations to the provided dataset.

        Arguments:
            - data {np.ndarray} dataset to be normalized.

        Returns:
            {np.ndarray} normalized dataset.
        """
        normalized_data = \
        (data - self.minumim_values)/(self.maximum_values-self.minumim_values)
        return normalized_data


    def revert(self, data):
        """
        Revert the pre-processing operatibhons to retreive the original dataset.

        Arguments:
            - data {np.ndarray} dataset for which to revert normalization.

        Returns:
            {np.ndarray} reverted dataset.
        """
        revered_data =\
        data*(self.maximum_values-self.minumim_values) + self.minumim_values
        return revered_data


def save_network(network, fpath):
    """
    Utility function to pickle `network` at file path `fpath`.
    """
    with open(fpath, "wb") as f:
        pickle.dump(network, f)


def load_network(fpath):
    """
    Utility function to load network found at file path `fpath`.
    """
    with open(fpath, "rb") as f:
        network = pickle.load(f)
    return network
