import numpy as np


class DataLoader(object):

    def __init__(self, dataset, batch_size, shuffle=True):
        if shuffle:
            dataset = self.shuffle(dataset[0], dataset[1])
        self.dataset = dataset
        self.batch_size = batch_size
        self.data_split(self.dataset, self.batch_size)
        self.i = 0

     
    def data_split(self, dataset, batch_size):
        n_batches = len(dataset[0]) // batch_size
        end = n_batches * batch_size
        self.input_batches = np.split(dataset[0][:end], n_batches)
        self.target_batches = np.split(dataset[1][:end], n_batches)
        self.nr_of_batches = n_batches
        if len(dataset[0]) % batch_size != 0:
            self.input_batches = self.input_batches + [dataset[0][end:]]
            self.target_batches = self.target_batches + [dataset[1][end:]]
            self.nr_of_batches += 1


    def __iter__(self):
        return self

    
    def __next__(self):
        if self.i < self.nr_of_batches:
            inputs = self.input_batches[self.i]
            targets = self.target_batches[self.i]
            self.i += 1
            return inputs, targets
        
        if self.shuffle:
            self.dataset = self.shuffle(self.dataset[0], self.dataset[1])
            self.data_split(self.dataset, self.batch_size)
        self.i = 0
        raise StopIteration


    @staticmethod
    def shuffle(data, targets):
        """
        Returns shuffled versions of the inputs.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_data_points, ).

        Returns: 2-tuple of np.ndarray: (shuffled inputs, shuffled_targets).
        """
        permutation = np.random.permutation(len(data))
        return data[permutation], targets[permutation]
