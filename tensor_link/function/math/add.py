from tensor_link import Tensor
from tensor_link.function import Function

class _add(Function):        
    def __init__(self):
        super().__init__()

    def __call__(self,a,b):
        new_tensor = Tensor(a.data + b.data)
        self.graph = [a,b]
        new_tensor.graph=self
        return new_tensor

    def backward(self,grad):
        self.graph[0].grad += grad.reshape((-1,)+self.graph[0].grad.shape).sum(axis=0)
        self.graph[1].grad += grad.reshape((-1,)+self.graph[1].grad.shape).sum(axis=0)
        self.graph[0].backward(self.graph[0].grad)
        self.graph[1].backward(self.graph[1].grad)

def add(a,b):
    return _add()(a,b)