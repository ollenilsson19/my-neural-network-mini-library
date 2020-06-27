import numpy as np



class Preprocessor(object):
    """
    Preprocessor: Object used to apply "preprocessing" operation to datasets.
    The object can also be used to revert the changes.
    """

    def __init__(self, *transforms):
        """
        Initializes the Preprocessor according to the provided dataset.
        (Does not modify the dataset.)

        Arguments:
            - data {np.ndarray} dataset used to determined the parameters for
            the normalization.
        """
        self.transforms = transforms

    def fit(self, data):
        for transform in self.transforms:
            data = transform.fit(data)
        return data


    def apply(self, data):
        """
        Apply the pre-processing operations to the provided dataset.

        Arguments:
            - data {np.ndarray} dataset to be normalized.

        Returns:
            {np.ndarray} normalized dataset.
        """
        for transform in self.transforms:
            data = transform(data)
        return data


    def __call__(self, data):
        return self.apply(data)


    def revert(self, data):
        """
        Revert the pre-processing operatibhons to retreive the original dataset.

        Arguments:
            - data {np.ndarray} dataset for which to revert normalization.

        Returns:
            {np.ndarray} reverted dataset.
        """
        for transform in self.transforms[::-1]:
            data = transform.revert(data)
        return data
