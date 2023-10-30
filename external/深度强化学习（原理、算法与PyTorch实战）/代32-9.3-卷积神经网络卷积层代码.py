# 深度强化学习——原理、算法与PyTorch实战，代码名称：代32-9.3-卷积神经网络卷积层代码.py

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
############################################get_ipython().run_line_magic('matplotlib', 'inline')


im = Image.open('data/cat.png').convert('L') 	# 读入一张灰度图的图片
im = np.array(im, dtype='float32') 		# 将其转换为一个矩阵
am = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(im)
print(am)


# 可视化图片
plt.imshow(im.astype('uint8'), cmap='gray')


# 将图片矩阵转化为 pytorch tensor，并适配卷积输入的要求
print(am.shape)
im = torch.from_numpy(im.reshape((1, 1, im.shape[0], im.shape[1]))) 
am = torch.from_numpy(am.reshape((1, 1, am.shape[0], am.shape[1]))) 
print(im)
print(am)


# 使用 nn.Conv2d
conv1 = nn.Conv2d(1, 1, 3, bias=False) 		# 输入通道数，输出通道数，核大小，定义卷积
sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32') 	# 定义轮廓检测算子
sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))	 	# 适配卷积的输入输出
conv1.weight.data = torch.from_numpy(sobel_kernel) 	# 给卷积的 kernel 赋值
edge1 = conv1(Variable(im)) 				# 作用在图片上
#edge2 = conv1(Variable(am))
edge1 = edge1.data.squeeze().numpy() 			# 将输出转换为图片的格式
#edge2 = edge2.data.squeeze().numpy()
plt.imshow(edge1, cmap='gray')


# 使用 F.conv2d
sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32') 	# 定义轮廓检测算子
sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3)) 				# 适配卷积的输入输出
weight = Variable(torch.from_numpy(sobel_kernel))
edge2 = F.conv2d(Variable(im), weight) 					# 作用在图片上
edge2 = edge2.data.squeeze().numpy() 					# 将输出转换为图片的格式
plt.imshow(edge2, cmap='gray')


# 使用 nn.MaxPool2d
pool1 = nn.MaxPool2d(2, 2)
print('before max pool, image shape: {} x {}'.format(im.shape[2], im.shape[3]))
small_im1 = pool1(Variable(im))
small_im1 = small_im1.data.squeeze().numpy()
print('after max pool, image shape: {} x {} '.format(small_im1.shape[0], small_im1.shape[1]))
plt.imshow(small_im1, cmap='gray')


# F.max_pool2d
print('before max pool, image shape: {} x {}'.format(im.shape[2], im.shape[3]))
small_im2 = F.max_pool2d(Variable(im), 2, 2)
small_im2 = small_im2.data.squeeze().numpy()
print('after max pool, image shape: {} x {} '.format(small_im1.shape[0], small_im1.shape[1]))
plt.imshow(small_im2, cmap='gray')


#输入数据
cs = np.array([[0,0,0,1,0,1,2],[0,1,1,1,1,0,0],[0,1,1,2,2,0,1],[0,0,1,2,2,1,1],[0,0,0,1,1,0,1],[0,0,2,1,2,1,0],[1,0,1,2,0,0,1]],dtype='float32')
cs = torch.from_numpy(cs.reshape((1, 1, cs.shape[0], cs.shape[1])))
conv1 = nn.Conv2d(1, 1, 3, bias=False)
# 定义卷积核
conv1_kernel = np.array([[0, 0, 1], [0, 2, 0], [1, 0, 1]], dtype='float32')
# 适配卷积的输入输出
conv1_kernel = conv1_kernel.reshape((1, 1, 3, 3)) 
conv1.weight.data = torch.from_numpy(conv1_kernel)
final = conv1(Variable(cs))
print(final)


# 输入数据
cs = np.array([[[0,0,0,0,0,0,0],[0,1,0,1,0,1,0],[0,1,1,1,1,0,0],[0,2,1,0,2,1,0],[0,0,1,0,2,0,0],[0,2,1,2,0,2,0],[0,0,0,0,0,0,0]],[[0,0,0,0,0,0,0],[0,2,1,2,0,1,0],[0,1,2,0,0,1,0],[0,0,1,0,2,1,0],[0,2,0,1,2,0,0],[0,1,0,0,1,0,0],[0,0,0,0,0,0,0]],[[0,0,0,0,0,0,0],[0,0,0,1,2,0,0],[0,0,2,1,0,0,0],[0,1,0,0,0,1,0],[0,2,0,0,0,2,0],[0,1,1,2,1,0,0],[0,0,0,0,0,0,0]]],dtype='float32')
cs = torch.from_numpy(cs.reshape(1,cs.shape[0],cs.shape[1],cs.shape[2]))
conv2 = nn.Conv2d( in_channels=3, out_channels=2, kernel_size=3, stride=2, padding=0,bias=True)
# 定义卷积核
conv2_kernel = np.array([[[[-1, 0, 1], [0, 0, 0], [1, -1, 1]],[[-1,1,1],[0,1,0],[1,0,0]],[[1,-1,1],[-1,1,0],[0,1,0]]],[[[0, 0, 1], [1, -1, 1], [0, 0, 1]],[[1,0,1],[-1,0,-1],[0,-1,0]],[[0,1,1],[-1,-1,0],[1,1,0]]]], dtype='float32')
# 适配卷积的输入输出
conv2_kernel = conv2_kernel.reshape((2, 3, 3, 3)) 
# 定义偏置项
conv2_bias = np.array([1,0])
conv2.weight.data = torch.from_numpy(conv2_kernel)
conv2.bias.data = torch.from_numpy(conv2_bias)
final2 = conv2(Variable(cs))
print(final2)



