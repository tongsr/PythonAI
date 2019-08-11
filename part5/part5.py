import numpy as np


class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self,x,y):
        self.x = x
        self.y = y
        out = x*y
        return out

    def backword(self,dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx,dy

class AddLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self,x,y):
        self.x = x
        self.y = y
        out = x+ y
        return out

    def backward(self,dout):
        dx = dout
        dy = dout
        return dx,dy


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self,x):
        self.mask = (x<=0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self,dout):
        dout[self.mask]=0
        dx = dout
        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self,x):
        out = 1/(1+np.exp(-x))
        self.out = out
        return out

    def backward(self,dout):
        dx = dout*(1.0 - self.out)*self.out
        return dx




apple = 100
orange = 150
apple_num = 2
orange_num = 3
tax = 1.1

mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_price_layer = AddLayer()
mul_tax_layer = MulLayer()


apple_price = mul_apple_layer.forward(apple,apple_num)
orange_price = mul_orange_layer.forward(orange,orange_num)
all_price = add_price_layer.forward(apple_price,orange_price)
price = mul_tax_layer.forward(all_price,tax)

dprice = 1
dall_price,dtax = mul_tax_layer.backword(dprice)
dapple_price,dorange_price = add_price_layer.backward(dall_price)
dapple ,dapple_num= mul_apple_layer.backword(dapple_price)
dorange,dorange_num = mul_orange_layer.backword(dorange_price)


print(price)
print(dapple)
print(dorange)
