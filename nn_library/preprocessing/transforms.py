import numpy as np

__all__ = ["Normalize"]



class Transform(object):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError()


    def fit(self, *args, **kwargs):
        raise NotImplementedError()


    def apply(self, data):
        raise NotImplementedError()


    def __call__(self, data):
        return self.apply(data)
        
    
    def revert(self, data):
        raise NotImplementedError()



class Normalize(Transform):
    def __init__(self):
        pass


    def fit(self, data):
        self.minimum_values = np.amin(data, axis=0)
        self.maximum_values = np.amax(data, axis=0)


    def apply(self, data):
        normalized_data = \
            (data - self.minimum_values)/(self.maximum_values-self.minimum_values)
        return normalized_data

    
    def revert(self, data):
        revered_data =\
            data*(self.maximum_values-self.minimum_values) + self.minimum_values
        return revered_data
