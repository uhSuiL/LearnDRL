# 深度强化学习——原理、算法与PyTorch实战，代码名称：代33-PyTorch与神经网络.py

# 通过torch.Tensor函数定义一个Tensor
import torch
a = torch.Tensor(2,4)
print(a)
b = torch.Tensor([[1,3,5,7],[2,4,6,8]])
print(b)


c = torch.IntTensor([[1,3,5,7],[2,4,6,8]])
print(c)


c = torch.ByteTensor([[1,3,5,7],[2,4,6,8]])
print(c)


a = torch.zeros(2,4)
print(a)


a = torch.rand(2,4)
print(a)


a = torch.randn(2,4)
print(a)


c[1,1] = 40
print(c)


import torch
import numpy as np
a = np.ones((2,3),dtype = np.float32)
print(a)
b = torch.from_numpy(a) 	#这里a,b两个对象是共享内存
a[1,1]=10
print(a)
print(b)


print(b.dtype)
b = b.int()
print(b.dtype)


if torch.cuda.is_available():
    a_cuda() = a.cuda()
    print(a_cuda)


a = torch.ones(3,4)
print(a.view(6,-1).shape)
print(a.reshape(-1,3).shape)


a = torch.randn(1,2,3,1)
print(a.unsqueeze(2).shape)
print(a.squeeze().shape)


a = torch.randn(2,3)
print(a.shape)
b = a.t
a = torch.randn(5,4,3,2)
print(a.shape)
print(a.transpose(0,1).contiguous().shape)
print(a.permute(3,1,0,2).shape)


import torch
from torch.autograd import Variable
x = Variable(torch.Tensor([8]), requires_grad = True)
w = Variable(torch.Tensor([2]))
y = w*x
y.backward()
print(x.grad)
print(w.grad)
print(x.data)
print(x.grad_fn)


import torch
from torch.autograd import Variable
x = Variable(torch.Tensor([3,5,7,9]), requires_grad = True)
w = Variable(torch.Tensor([2,4,6,8]))
y = w*x*x
#weight = torch.ones(4)
#y.backward(weight,create_graph = True)
y.backward(torch.ones_like(y),retain_graph = True)
print(x.grad)
x.grad.data.zero_() 		#对x的梯度清0
weight = torch.Tensor([0.2,0.4,0.6,0.8])
y.backward(weight)
print(x.grad)


x = Variable(torch.Tensor([[2,3,4],[1,2,3]]), requires_grad=True)
y = Variable(torch.ones(2,4))
w = Variable(torch.Tensor([[1,3,5,7],[2,4,6,8],[7,8,9,10]]))
out = torch.mean(y - torch.matmul(x, w)) 		# torch.matmul 是做矩阵乘法
out.backward()
print(x.grad)


print(torch.mean(torch.Tensor([2,4,5,6])))


import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


seq_net = nn.Sequential(
    nn.Linear(2, 4), 
    nn.ReLU(),
    nn.Linear(4, 1)
)
print(seq_net)
print(seq_net[0])
print(seq_net[0].weight)


from collections import OrderedDict
seq_orderdict_net = nn.Sequential(OrderedDict([
    ("Line1",nn.Linear(2,4)),
    ("Relu1",nn.ReLU()),
    ("Line2",nn.Linear(4,1))]))
print(seq_orderdict_net)


class module_net(nn.Module):
    def __init__(self, num_input, num_hidden, num_output):
        super(module_net, self).__init__()
        self.layer1 = nn.Linear(num_input, num_hidden)        
        self.layer2 = nn.ReLU()
        self.layer3 = nn.Linear(num_hidden, num_output)        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
m_net = module_net(2, 4, 1)
print(m_net)


class module_net(nn.Module):
    def __init__(self, num_input, num_hidden, num_output):
        super(module_net, self).__init__()
        self.layer1 = nn.Linear(num_input, num_hidden)       
        self.layer2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(num_hidden, num_output))       
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
m_net = module_net(2, 4, 1)
print(m_net)


input = torch.FloatTensor([[-0.4089,-1.2471,0.5907],[-0.4897,-0.8267,-0.7349],[0.5241,-0.1246,-0.4751]])
input

m = nn.Sigmoid()
n = m(input)
n


target = torch.FloatTensor([[0,1,1],[0,0,1],[1,0,1]])
target


loss_fn = nn.BCELoss()
loss = loss_fn(n, target)
loss


loss_fn = nn.BCEWithLogitsLoss()
loss = loss_fn(input, target)
loss


loss_fn = nn.MSELoss()
loss = loss_fn(n, target)
loss


loss_fn = nn.L1Loss()
loss = loss_fn(n, target)
loss.data


loss_fn = nn.CrossEntropyLoss()
x = Variable(torch.randn(3,5))
y = Variable(torch.LongTensor(3).random_(5))
loss = loss_fn(x, y)
loss.data


