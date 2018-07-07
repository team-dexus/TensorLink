class Tensor:
    def __init__(self,data):
        self.data = data
        self.graph = []
        self.grad = xp.zeros_like(data)
    def backward(self):
        pass
