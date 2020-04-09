import numpy as np
from keras.datasets import mnist
import tensorflow as tf

# 0D tensor
x = np.array(12)
print(x)
print (x.ndim)  # dimension of the array

# 1D tensor
x = np.array([12, 3, 6, 89, 0])
print (x.ndim)
print (x.shape)

# 2D tensor
x = np.array([[12, 3, 6, 89, 0], [3, 6, 7, 9, 10], [7, 80, 4, 36, 2]])
print (x.ndim)
print (x.shape)

# 3D tensor
x = np.array([[[12, 3, 6, 89, 0],
               [3, 6, 7, 9, 10],
               [7, 80, 4, 36, 2]],
              [[1, 3, 6, 9, 0],
               [3, 6, 7, 8, 10],
               [7, 0, 4, 6, 2]],
              [[2, 3, 6, 8, 0],
               [3, 6, 7, 9, 1],
               [7, 8, 4, 6, 2]]]
             )
print (x.ndim)
print (x.shape)

# Using MNIST Hand written dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(train_images.ndim)
print(train_images.shape)
print(train_images.dtype)

digit = train_images[5]  # selected a specific digit alongside the first axis using the syntax train_images[i]
import matplotlib.pyplot as plt

plt.imshow(digit, cmap=plt.cm.binary)
#plt.show()

# select digit 10 to 100 (100 not inclded)
#images_10to100 = train_images[10:100, :, :]
#images_10to100 = train_images[10:100, 0:28, 0:28]
images_10to100 = train_images[10:100]
print(images_10to100.shape)

#select between any two indices along each tensor axis
myslice_1 = train_images[5:100, 0:14, 0:14]
print(myslice_1.shape)
myslice_2 = train_images[5:100, 14:, 14:]
print(myslice_2.shape)
myslice_3 = train_images[5:100, 7:-7, 7:-7 ]
print(myslice_3.shape)

#Batches of images
batch_128 = train_images[:128]
print(batch_128.shape)
batch_128_2 = train_images[128:256]
print(batch_128_2.shape)

