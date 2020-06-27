import numpy as np
import os

dir_path = os.path.dirname(os.path.realpath(__file__))


class Datasets(object):
    def __init__(self, test_split=0.2, val_split=None, Preprocessor=None):
        self.test_split = test_split
        self.val_split = val_split
        if self.val_split and not self.test_split:
            raise ValueError(f"Specified validation split: {self.val_split} while test split is: {self.test_split}. ",
                             f"When using a validation set you must also use a test set")
        self.preprocessor = Preprocessor


    def split(self, inputs, targets):
        if self.val_split and self.test_split:
            test_split_idx = int((1-self.test_split) * len(inputs))
            val_split_idx = int((1-self.val_split) * len(inputs)) - test_split_idx
            self.train_data = inputs[:val_split_idx], targets[:val_split_idx]
            self.val_data = inputs[val_split_idx:test_split_idx], targets[val_split_idx:test_split_idx]
            self.test_data = inputs[test_split_idx:], targets[test_split_idx:]
            return self.train_data, self.val_data, self.test_data
        elif self.test_split:
            test_split_idx = int((1-self.test_split) * len(inputs))
            self.train_data = inputs[:test_split_idx], targets[:test_split_idx]
            self.test_data = inputs[test_split_idx:], targets[test_split_idx:]
            return self.train_data, self.test_data
        else:
            self.train_data = inputs, targets
            return self.train_data


    def apply_preprocessor(self, data):
        if self.preprocessor:
            preprocessed_data = [[0, 0] for _ in range(len(data))]
            self.preprocessor.fit(data[0][0])
            for idx, dataset in enumerate(data):
                preprocessed_data[idx][0] = self.preprocessor(dataset[0])
                preprocessed_data[idx][1] = dataset[1]
            return tuple(preprocessed_data)
        return data


    def iris(self):
        print (os.getcwd())
        raw_data = np.loadtxt(f"{dir_path}/raw/iris.dat")
        np.random.shuffle(raw_data)
        inputs = raw_data[:, :4]
        targets = raw_data[:, 4:]
        data = self.split(inputs, targets)
        return self.apply_preprocessor(data)


if __name__ == "__main__":
    print(dir_path)
    train_data, test_data = Datasets(test_split=0.2, val_split=None, Preprocessor=None).iris()
    print(train_data)
    print(test_data)
