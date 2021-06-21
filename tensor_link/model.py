from tensor_link.learnable import Learnable

class Model:
    def __init__(self):
        pass
    def zero_grads(self):
        for obj in self.__dict__.values():
            if issubclass(obj.__class__, Learnable):
                obj.zero_grads()
