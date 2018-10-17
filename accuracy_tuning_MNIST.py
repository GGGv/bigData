#Homework 6 for CS 6220
#problem 1.1 accuracy tuning of a deep learning framework
#this code is largely based on tensorflow tutorial 
#https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/03C_Keras_API.ipynb
#original author: Magnus Erik Hvass Pedersen

import matplotlib.pyplot as plt 
import tensorflow as tf 
import numpy as np 
import math
# from tf.keras.models import Sequential
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten
from tensorflow.python.keras.models import Model

#load data
#mnist model is defined in code from TensorFlow_Tutorial
from TensorFlow_Tutorials.mnist import MNIST 
data=MNIST(data_dir="data/MNIST")
print("Size of:")
print("- TRaining-set:\t\t{}".format(data.num_train))
print("- Validation-set:\t{}".format(data.num_val))
print("- Test-set:\t\t{}".format(data.num_test))

#copy data dimensions
img_size_flat=data.img_size_flat
img_shape=data.img_shape
num_classes=data.num_classes
img_size=data.img_size
num_channels=data.num_channels
img_shape_full=data.img_shape_full

#construction of the Keras Model model
inputs=Input(shape=(img_size_flat,))
net=inputs
net=Reshape(img_shape_full)(net)
net=Conv2D(kernel_size=5,strides=1,filters=16,padding='same',
		   activation='relu',name='layer_conv1')(net)
net=MaxPooling2D(pool_size=2,strides=2)(net)
net=Conv2D(kernel_size=5,strides=1,filters=36,padding='same',
		   activation='relu',name='layer_conv2')(net)
net=MaxPooling2D(pool_size=2,strides=2)(net)
net=Flatten()(net)
net=Dense(128,activation='relu')(net)
net=Dense(num_classes,activation='softmax')(net)
outputs=net

#model compilation
model=Model(inputs=inputs,outputs=outputs)
model.compile(optimizer='rmsprop',
			   loss='categorical_crossentropy',
			   metrics=['accuracy'])

#training
model.fit(x=data.x_train,y=data.y_train,epochs=1,batch_size=128)

#evaluation
result=model.evaluate(x=data.x_test,y=data.y_test)
for name,value in zip(model.metrics_names,result):
	print(name,value)

model.summary()