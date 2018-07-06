import random
eps = 1e-8
print('w')

class Tensor:
    def __init__(self,value = 0,train = True):
        self.value = value
        self.graph = 0
        self.grad = 0
        self.train = train
    
    def backward(self,grad = 1):
        self.graph.backward(grad)
       
class add:        
    def __init__(self):
        self.graph = []
    def __call__(self,a,b):
        new_tensor = Tensor()
        new_tensor.value = a.value + b.value
        self.graph = [a,b]
        new_tensor.graph=self
        return new_tensor
    def backward(self,grad):
        for node in self.graph:
            this_grad = grad
            node.grad += this_grad
            if not node.graph == 0:
                node.graph.backward(this_grad)

class minus:        
    def __init__(self):
        self.graph = []
    def __call__(self,a,b):
        new_tensor = Tensor()
        new_tensor.value = a.value - b.value
        self.graph = [a,b]
        new_tensor.graph=self
        return new_tensor
    def backward(self,grad):
        for node in self.graph:
            this_grad = grad
            node.grad += this_grad
            if not node.graph == 0:
                node.graph.backward(this_grad)

class product:        
    def __init__(self):
        self.graph = []
    def __call__(self,a,b):
        new_tensor = Tensor()
        new_tensor.value = a.value * b.value
        self.graph = [a,b]
        self.graph_sub = [b,a]
        new_tensor.graph = self
        
        return new_tensor

    def backward(self,grad):
        for i in range(len(self.graph)):
            node = self.graph[i]
            this_grad = grad * self.graph_sub[i].value
            node.grad += this_grad
            if not node.graph == 0:
                node.graph.backward(this_grad)
 

class sgd:
    def __init__(self,lr):
        self.lr = lr
    def update(self,tensors):
        for tensor in tensors:
            if tensor.train:
                tensor.value -= tensor.grad * self.lr
            
def clear_grad(tensors):
    for tensor in tensors:
        tensor.grad = 0
        

a1 = add()
a2 = add()
a3 = add()
a4 = add()
a5 = add()

p1 = product()
p2 = product()
p3 = product()
p4 = product()
p5 = product()

m1 = minus()
pe = product()


weight_rate = 0.001
t1 = Tensor(random.random() * weight_rate)
t2 = Tensor(random.random() * weight_rate)
t3 = Tensor(random.random() * weight_rate)
t4 = Tensor(random.random() * weight_rate)
t5 = Tensor(random.random() * weight_rate)
t6 = Tensor(random.random() * weight_rate)

train_data = [[-3,9],[-2,4],[-1,1],[0,0],[1,1],[2,4],[3,9]]

opt = sgd(0.00001)
for epoch in range(10000):
    total_error = 0
    for batch in range(len(train_data)):
        x = Tensor(train_data[batch][0],False)
        h = p1(t1,x)
        h = a1(h,t2)
        h = p2(h,x)
        h = a2(h,t3)
        h = p3(h,x)
        h = a3(h,t4)
        h = p4(h,x)
        h = a4(h,t5)
        h = p5(h,x)
        h = a5(h,t6)

        error = m1(h,Tensor(train_data[batch][1]))
        error = pe(error,error)
        clear_grad([t1,t2,t3,t4,t5,t6])
        error.backward()
        opt.update([t1,t2,t3,t4,t5,t6])
        total_error += error.value
    print('error: %f' % (total_error))
print('\n')
print(t1.value)
print(t2.value)
print(t3.value)
print(t4.value)
print(t5.value)
print(t6.value)
