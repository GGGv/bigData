import matplotlib.pyplot as plt 
import tensorflow as tf 
import numpy as np 
import math
# from tf.keras.models import Sequential
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten

#load data
from TensorFlow_Tutorials.mnist import MNIST 
data=MNIST(data_dir="data/MNIST")
print("Size of:")
print("- TRaining-set:\t\t{}".format(data.num_train))
print("- Validation-set:\t{}".format(data.num_val))
print("- Test-set:\t\t{}".format(data.num_test))

#copy data dimensions
#images are stored in one-dimensional arrays of this length
img_size_flat=data.img_size_flat
#tuple with height and width of images used to reshape arrays
img_shape=data.img_shape
## classes, one class for each of 10 digits
num_classes=data.num_classes
## pixels in each dimension of an array
img_size=data.img_size
## colour channels for the image: 1 channel for gray-scale
num_channels=data.num_channels
#tuple with height, width and depth used to reshape arrays
#this is used for reshaping in Keras
img_shape_full=data.img_shape_full

#helper function to plot images
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
        
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

def plot_example_errors(cls_pred):
    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Boolean array whether the predicted class is incorrect.
    incorrect = (cls_pred != data.y_test_cls)

    # Get the images from the test-set that have been
    # incorrectly classified.
    images = data.x_test[incorrect]
    
    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = data.y_test_cls[incorrect]
    
    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])


'''
#sequential model
#start construction of the Keras sequential model.
model=Sequential()
#add an input layer which is similar to a feed_dict in tf
#the input-shape must be a tuple containing the image-size
model.add(InputLayer(input_shape=(img_size_flat,)))
#the input is a flattened array with 784 elements,
#but the convolutional layers expect images with shape(28,28,1)
model.add(Reshape(img_shape_full))
#first convolutional layer with ReLU-activation and max-pooling
model.add(Conv2D(kernel_size=5,strides=1,filters=16,padding='same',
				 activation='relu',name='layer_conv1'))
model.add(MaxPooling2D(pool_size=2,strides=2))
#second convolutional layer with ReLU-activation and max-pooling
model.add(Conv2D(kernel_size=5,strides=1,filters=16,padding='same',
				 activation='relu',name='layer_conv2'))
model.add(MaxPooling2D(pool_size=2,strides=2))
#flatten the 4-rank output of the convolutional layers to 2-rank
#that can be input to a fully connected/dense layer
model.add(Flatten())
#first fully-connected layer with ReLU-activation
model.add(Dense(128,activation='relu'))
#last fully-connected layer with softmax-activation
model.add(Dense(num_classes,activation='softmax'))

#model compilation
from tensorflow.python.keras.optimizers import Adam
optimizer=Adam(lr=1e-3)
#use loss-function categorical_crossentropy
model.compile(optimizer=optimizer,
			  loss='categorical_crossentropy',
			  metrics=['accuracy'])

#training
model.fit(x=data.x_train,
		  y=data.y_train,
		  epochs=1,batch_size=128)

#evaluation
result=model.evaluate(x=data.x_test,y=data.y_test)
for name,value in zip(model.metrics_names,result):
	print(name,value)

#prediction
images=data.x_test[0:9]
cls_true=data.y_test_cls[0:9]
y_pred=model.predict(x=images)
cls_pred=np.argmax(y_pred,axis=1)
plot_images(images=images,cls_true=cls_true,cls_pred=cls_pred)
'''


#functional mdoel with the same network structrue as the previous one
inputs=Input(shape=(img_size_flat,))
#variable used for building the neural network
net=inputs
net=Reshape(img_shape_full)(net)
#first convolutional layer
net=Conv2D(kernel_size=5,strides=1,filters=16,padding='same',
		   activation='relu',name='layer_conv1')(net)
net=MaxPooling2D(pool_size=2,strides=2)(net)
#second convoutional layer
net=Conv2D(kernel_size=5,strides=1,filters=36,padding='same',
		   activation='relu',name='layer_conv2')(net)
net=MaxPooling2D(pool_size=2,strides=2)(net)
#flatten the output
net=Flatten()(net)
#first fully-connected layer with relu
net=Dense(128,activation='relu')(net)
#last fully-connected layer with softmax
net=Dense(num_classes,activation='softmax')(net)
#output of neural network
outputs=net

#model compilation
from tensorflow.python.keras.models import Model 
model2=Model(inputs=inputs,outputs=outputs)
model2.compile(optimizer='rmsprop',
			   loss='categorical_crossentropy',
			   metrics=['accuracy'])

#training
model2.fit(x=data.x_train,y=data.y_train,epochs=1,batch_size=128)

#evaluation
result=model2.evaluate(x=data.x_test,y=data.y_test)

for name,value in zip(model2.metrics_names,result):
	print(name,value)
