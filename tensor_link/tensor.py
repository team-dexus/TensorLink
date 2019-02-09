import tensor_link
from tensor_link import xp

class Tensor:
    def __init__(self,data = None,train = True):
        self.data = data.astype(tensor_link.config.DEFAULT_DTYPE)
        self.graph = None
        self.grad = xp.zeros_like(data,dtype=tensor_link.config.DEFAULT_DTYPE)
        
    def backward(self,grad=None):
        if self.graph is None:#計算グラフがNoneなら誤差逆伝播をやめる
            return
        if grad is None:#誤差逆伝播のスタートなら全部1を流す
            grad = xp.ones_like(self.data,dtype=tensor_link.config.DEFAULT_DTYPE)

        self.graph.backward(grad)