
class Optimizer(object):
    def __init__(self, lr):
        pass


    def step(self, *args):
        raise NotImplementedError()

    
    def __call__(self, gradient):
        return self.step(gradient)

class SGD(Optimizer):
    def __init__(self, lr):
        super().__init__(lr)
        self.lr = lr
        

    def step(self, gradient):
        return -self.lr*gradient


class Adam(Optimizer):
    pass