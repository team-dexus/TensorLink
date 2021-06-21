# Tensor Linkの仕様
## 1.記法
	import tensor_link.learnable as L
	import tensor_link.function as F
	from tensor_link import Model
	import train_data
	import teacher_data

	class CNN(Model):
		def __init__(self):
			super().__init__()
			self.c1 = L.conv2D(3,128,3,1,1)
			self.c2 = L.conv2D(128,1,3,1,1)
			self.l1 = L.Dense(None,2)
		def __call__(self, x):
			h = self.c1(x)
			h = self.c2(h)
			h = self.l1(h)

			return h
	opt = Adam(0.001,0.99,0.99)
	network = CNN()
	opt.assign(network)
	while True:
		result = CNN(train_data)
		loss = F.mse(result,teacher_data)
		opt.update(network,loss)#networkを、lossの誤差逆伝播をもとに更新、の意。
		print(loss.data)

## 2.クラス概要
Tensor → 計算グラフを保持した変数（ベクトル、二次元配列なども値として入る。）  
Function → 学習不可な関数  
Learnable → 学習可能な関数。TensorクラスとFunctionクラスの組み合わせで作られる。  

## 3.Tensorクラス
#### 変数
graph → 計算グラフを何らかの形で保持  
data → 生の値  
grad → 勾配を格納する  
#### 関数
backward() → 誤差逆伝播を行う。  
## 4.Function
Tensorを返す関数。クラスを生成して、それで計算してTensorを返すことで、Tensorが計算グラフを保持できるようにする
## 5.Learnableクラス
（考え中）
