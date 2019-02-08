from tensor_link import Tensor

#Functionに用いるクラスはFunctionクラスを継承して作る
class Function:
    def __init__(self):
        self.graph = None

    def __call__(self):
        pass

    def backward(self,grad):
        pass

#作ったFunctionは全てここでimport
from tensor_link.function.math.add import add
from tensor_link.function.math.product import product
