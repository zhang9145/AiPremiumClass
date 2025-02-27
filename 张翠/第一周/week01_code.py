import numpy as np
a = [1,2,3]
b = np.array(a)
print(b)

a = np.array([1,2,3],dtype=float)
print(a)

a1 = np.array([(1,3,5),(4,6,8),(2,7,9)])
print(a1)

a2 = np.zeros((3,3,3),int)
print(a2)

c = np.ones((3,3),dtype = np.float32) * 3
print(c)

d = np.arange(1,3,0.2)
print(d)

e = np.eye(5)
print(e)

b = np.random.random(6)
print(b)

mu,sigma = 0.5,0.1    #均值0.5，标准差为0.1
a2 = np.random.normal(mu,sigma,6)
s2 = np.random.normal(mu,sigma,(5,6))
print(a2)
print(s2)

a3 = np.array([(1,2,3),(4,5,6),(7,8,9)])
for i,j,k in a3[1:]:
    print(i,j,k)

a4 = np.array([(3,3,3),(3,3,3),(3,3,3)])
print("ndim:",a4.ndim)
print("shape:",a4.shape)
print("size:",a4.size)
print("dtype:",a4.dtype)
print(3 in a4)
print((3,) in a4)
print((1,) in a4)

a5 = np.ones((3,3,3))
print(a5)
a5.reshape(3,9)

a6 = np.array([(1,2,3),(4,5,6),(7,8,9)])
print(a6)
a7 = a6.T
print(a7)
a8 = a6.flatten()
print(a8)
a9 = a6[:,:,np.newaxis]
a10 = a6[:,np.newaxis,:]
print(a9)
print(a10)

print(a6+a7)  
print(a6-a7)
print(a6.sum())   #元素相加
print(a6.prod())   #元素相乘

a6 = np.array([(1.1,2.6,3.4),(4.5,5.9,6),(7,8,9)])
print("mean:",np.mean(a6))   #平均数
print("var:",np.var(a6))    #方差
print("std:",np.std(a6))    #标准差
print("max:",np.max(a6))    #最大值
print("min:",np.min(a6))    #最小值
print("argmax:",np.argmax(a6))    #最大值下标
print("argmin:",np.argmin(a6))    #最小值下标
print("ceil:",np.ceil(a6))    #向上取整
print("floor:",np.floor(a6))    #向下取整
print("rint:",np.rint(a6))    #四舍五入

a6 = np.array([(1,5,3),(2,3,3),(7,8,9)])
a12 = np.sort(a6)    #排序，默认正序
print(a12)

m1 = np.array([(1,2),(3,4),(5,6)],dtype = float)
m2 = np.array([(5,6,3),(7,8,2)],dtype = float)
print(np.dot(m1,m2))   #矩阵1的最后一维 == 矩阵2的倒数第二维
print(m1@m2)

import torch
data = torch.tensor([[1,2],[3,4]])
print(data)

import numpy as np
a = np.array([[1,2,3],[4,5,6]])
data2 = torch.from_numpy(a)
print(data2)

import torch
data = torch.rand_like(data2,dtype=torch.float)
print(data)

data3 = torch.ones_like(data2)
print(data3)

data4 = torch.zeros_like(data2)
print(data4)

#shape = [2,3,]
shape = (2,3)
data5 = torch.rand(shape)
print(data5)
data6 = torch.ones(shape)
print(data6)
data7 = torch.zeros(shape)
print(data7)

m = torch.ones(5,3,dtype=torch.double)
print(m)
n = torch.rand_like(m,dtype=torch.float)
print(n)

print(torch.rand(3,3,3))
print(torch.randn(2,4))
print(torch.normal(mean=0.5,std=0.01,size=(5,6)))   #mu,sigma = 0.5,0.01 np.random.normal(mu,sigma,(5,6))
print(torch.linspace(start=1,end=10,steps=6))    #steps是份数       #np.range(1,10,0.5)  0.5是步幅

print(data.shape)
print(data.size())
print(data.dtype)
print(data.device)

print(torch.cuda.is_available())   #检测pytorch是否支持GPU

if torch.cuda.is_available():
    device = torch.device("cuda")
    data.to(device)
print(data)
print(data.device)

tensor = torch.rand([3,4,3])
print(tensor)
print(tensor[1])
print(tensor[1,1])
print(tensor[:,1])
#print(tensor[:,-1])
#print(tensor[:,-1,:])
print(tensor[:,:,-1])
print(tensor[...,-1])

tensor = torch.rand([3,4,3])
tensor[1,1] = 1
print(tensor)

ten1 = torch.cat([data5,data6],dim=1)    #dim在第几维上进行拼接
print(ten1)
ten2 = torch.cat([tensor,tensor],dim=2)
print(ten2) 

ten3 = torch.cat([tensor,tensor],dim=1)
print(ten3)

ten4 = torch.stack([tensor,tensor],dim=2)
print(ten4)

tensor = torch.arange(1,10,dtype=torch.double).reshape(3,3)
print(tensor)
t1 = tensor@tensor.T
print(t1)
t2 = tensor.matmul(tensor.T)
print(t2)
#t4 = torch.rand_like(tensor)
#torch.matmul(tensor,tensor.T,out=t4)
import numpy as np
t3 = np.dot(tensor,tensor.T)
print(t3)
t4 = torch.rand_like(tensor)
torch.matmul(tensor,tensor.T,out=t4)

z1 = tensor * tensor
print(z1)
z2 = tensor.mul(tensor)
print(z2)
z3 = torch.rand_like(tensor)
torch.mul(tensor,tensor,out=z3)

t = tensor.sum()
print(t,type(t))
tt = t.item()   #只是想得到结果值的时候
print(tt,type(tt))
ten = t.numpy()
print(ten,type(ten))

tensor = torch.arange(1,10,dtype=torch.double).reshape(3,3)
print(tensor)
# tensor.add_(10)
# tensor = tensor + 10
tensor += 10
print(tensor)

tensor = torch.arange(1,10,dtype=torch.double).reshape(3,3)
print(tensor)
t = tensor.numpy()
print(t)
tensor.add_(5)
print(tensor)
print(t)

import torch
A = torch.randn(5,5,requires_grad=True)
b = torch.randn(5,requires_grad=True)
c = torch.randn(5,requires_grad=True)
x = torch.randn(5,requires_grad=True)
print(A)
print(b)
print(c)
print(x)
result = A * x.T + b * x + c
print(result)
