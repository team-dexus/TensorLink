from tensor_link.learnable import Learnable

#OptimizerはOptimizerを継承して作る
class Optimizer:
    def __init__(self):
        self.params = []
    def assign(self,network):
        for obj in network.__dict__.values():
            if issubclass(obj.__class__, Learnable):
                self.params = self.params + obj.get_all_params()

#作ったFunctionは全てここでimport
from tensor_link.optimizer.sgd import SGD
