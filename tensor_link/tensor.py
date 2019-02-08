from tensor_link import xp

class Tensor:
    def __init__(self,data = 0,train = True):
        self.data = data
        self.graph = None
        self.grad = xp.zeros_like(data)
        
    def backward(self,grad=None):
        if self.graph is None:#計算グラフがNoneなら誤差逆伝播をやめる
            return
        if grad is None:#誤差逆伝播のスタートなら全部1を流す
            grad = xp.ones_like(self.data)

        self.graph.backward(grad)