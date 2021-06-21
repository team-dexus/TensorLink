from tensor_link.optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self,lr=0.01):
        super().__init__()
        self.lr = lr
    def update(self):
        for param in self.params:
            param.data -= param.grad * self.lr