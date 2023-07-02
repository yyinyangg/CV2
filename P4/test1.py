from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import skimage.color
import torch

import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from matplotlib import pyplot as plt
from skimage import color

from utils import VOC_LABEL2COLOR
from utils import VOC_STATISTICS

from torchvision import models

#自己加的库
import random
import PIL
from torchvision import transforms

os.environ['VOC2007 HOME'] ="C:/Users/Aobo Tan/PycharmProjects/pythonProject_Computer_Vision/Assignment/Assignment-4/VOCtrainval_06-Nov-2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007"
root = os.environ['VOC2007 HOME']
root_of_splitFile = os.path.join(root,'ImageSets','Segmentation','train.txt')
root_of_image = os.path.join(root,'JPEGImages')
root_of_segmentation = os.path.join(root,'SegmentationClass')
import numpy as np

import random
from PIL import Image


def random_lines_from_file(file_path, num_lines):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    random_lines = random.sample(lines, num_lines)
    random_lines = [line.strip() for line in random_lines]

    return random_lines

split_file_path = root_of_splitFile  # 替换为实际的文件路径
num_lines = 20  # 要读取的行数
lines = random_lines_from_file(split_file_path, num_lines)
input_filenames = []
target_filenames = []

for line in lines:
    input_filenames.append (root_of_image+'/' +line.strip() + '.jpg')
    target_filenames.append(root_of_segmentation + '/' + line + '.png')
path_of_im = input_filenames[1]
path_of_gt = target_filenames[1]
print(path_of_im)




image1 = Image.open(path_of_im)
image2 = Image.open(path_of_gt)
background = image1.convert('YCbCr')
overlay = image2.convert('YCbCr')
converted_image1 = Image.blend(background,overlay,0.5)

#image1.show()
#converted_image1.show()
label2rgb = {idx: VOC_LABEL2COLOR[idx] if idx < len(VOC_LABEL2COLOR) else (224, 224, 192) for idx in range(len(VOC_LABEL2COLOR) + 235)}
np_jpeg = np.array(image1)

np_image_YCbCr = skimage.color.rgb2ycbcr(np_jpeg)
np_png = np.array(image2)
print("shape of np_png",np_png.shape)
H,W = np.shape(np_png)
label_rgb = np.zeros((H,W,3), dtype=np.uint8)
for i in range(H):
    for j in range(W):
        #print(np_png[i][j])
        #print(label2rgb[np_png[i][j]])
        label_rgb[i,j] = label2rgb[np_png[i][j]]
        #print(label_rgb[i,j])
label_rgb = label_rgb
label_rgb_ycbcr = skimage.color.rgb2ycbcr(label_rgb)
colored = np.zeros_like(np_jpeg)
colored[:,:,0] = np_image_YCbCr[:,:,0]
colored[:,:,1:] = label_rgb_ycbcr[:,:,1:]

colored_rgb= skimage.color.ycbcr2rgb(colored)
colored_rgb = np.clip(colored_rgb,0,255)
print(np_jpeg.dtype)
print(colored.dtype)

plt.imshow(np_image_YCbCr[:,:,2])
plt.colorbar()
plt.show()
plt.imshow(label_rgb)
plt.colorbar()
plt.show()
plt.imshow(colored.astype(np.uint8))
plt.colorbar()
plt.show()
print('test')
alpha = 0.5
#converted_image2 = (alpha * np_png + (1 - alpha) * np_jpeg).astype(np.uint8)

#colored = voc_label2color(np_jpeg,np_png)




#plt.imshow(np_jpeg)
#plt.show()
#plt.imshow(converted_image1)





#print(tensor_jpeg.shape)  # 输出: torch.Size([3, H, W])
#print(tensor_jpeg.dtype)  # 输出: torch.float32
#
#print(tensor_png.shape)  # 输出: torch.Size([1, H, W])
#print(tensor_png.dtype)  # 输出: torch.int64





