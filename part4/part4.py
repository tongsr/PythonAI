import numpy as np

import sys,os
sys.path.append(os.pardir)

from common.functions import softmax,cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)

    def predict(self,x):
        return np.dot(x,self.W)

    def loss(self,x,t):
        z=self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y,t)
        return loss




def numerical_gradient2(f,x):
    h=1e-4
    grad = np.zeros_like(x)
    for idx in range(x.size):
        tmp_val=x[idx]
        x[idx]=tmp_val+h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1-fxh2)/(2*h)
        x[idx]=tmp_val
    return grad


def gradient_descent(f,init_x,lr=0.01,step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f,x)
        x -= lr*grad
    return x



net = simpleNet()


x = np.array([0.6,0.9])


t = np.array([0,0,1])



def func1(W):
    return net.loss(x,t)




result = gradient_descent(func1,net.W,lr=0.02,step_num=1000)
print(result)


print(np.argmax(np.dot(x,result)))
print(np.argmax(t))

'''
def function(x):
    return x[0]**2+x[1]**2

init_x=np.array([-3.0,4.0])

print(gradient_descent(function,init_x,0.02,5000))
'''



