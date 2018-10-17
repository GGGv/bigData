#matplotlib inline
import matplotlib.pyplot as plt
import tensorflow as tf 
import numpy as np 
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta 
import math

#configuration of neural network
#convolutional layer 1
filter_size=5
num_filters1=16

#convolutional layer 2
filter_size2=5
num_filters2=36

#fully connected layer
fc_size=128

#configuration of neural network
#convolutional layer 1.
filter_size1=5
num_filters1=16
#convolutional layer 2.
filter_size2=5
num_filters2=36
#fully-connected layer.
fc_size=128

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

#helper-functions for creating new variables
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.05))
def new_biases(length):
    return tf.Variable(tf.constant(0.05,shape=[length]))

#helper-function for creating a new convolutional layer
def new_conv_layer(input,               #the previous layer
                   num_input_channels,  #num. channels in prev. layer
                   filter_size,         #width and height of each filter
                   num_filters,         #num. of filters
                   use_pooling=True):   #use 2x2 max_polling
    #shape of the filter-weights for the convolution.
    #this format is determined by the TensorFlow API
    shape=[filter_size,filter_size,num_input_channels,num_filters]
    #create new weights aka. filters with the given shape
    weights=new_weights(shape=shape)
    #create new biases, one for each filter
    biases=new_biases(length=num_filters)
    #create the TensorFlow operation for convolution
    #the first and last stride must always be 1,
    #because the first is for the image-number and 
    #the last is for the input-channel.
    #the padding is set to 'same' which means the input image
    #is padded with zeroes so the size of the output is the same
    layer=tf.nn.conv2d(input=input,
    				   filter=weights,
    				   strides=[1,1,1,1],
    				   padding='SAME')
    #add the biases to the results of the convolution.
    #a bias-value is added to each filter-channel
    layer+=biases
    #use pooling to down-sample the image resolution
    if use_pooling:
    	#this is 2x2 max-pooling, which means that we
    	#consider 2x2 windows and select the largest value
    	#in each window. then we move 2 pixels to the next window
    	layer=tf.nn.max_pool(value=layer,
    						 ksize=[1,2,2,1],
    						 strides=[1,2,2,1],
    						 padding='SAME')
    #rectified linear unit (ReLU)
    #it calculates max(x,0) for each input pixel x.
    #this adds some non-linearity to the formula and allows us
    #to learn more complicated functions
    layer=tf.nn.relu(layer)
    #we return bothe the resulting layer and the filter-weights
    #because we will plot the weights later.
    return layer, weights

#helper-function for flattening a layer
def flatten_layer(layer):
	#get the shape of the input layer.
	layer_shape=layer.get_shape()
	#the shape of the input layer is assumed to be:
	#later_shape=[num_images,img_height,img_width,num_channels]
	#num. features is: img_height*img_width*num_channels
	#we can use a function from tensorflow to calculate this.
	num_features=layer_shape[1:4].num_elements()
	#reshape the layer to [num_images,num_features].
	#note that we just set the size of the second dimension
	#to num_features and the size of the first dimension to -1
	#which means the size in that dimension is calculated 
	#so the total size of the tensor si unchanged from the reshaping
	layer_flat=tf.reshape(layer,[-1,num_features])
	#the shape of flattened layer is now:
	#[num_imgages,img_height*height_width*num_channels]
	return layer_flat,num_features

#helper-function for creating a new fully-connected layer
def new_fc_layer(input,			#the previous layer
				 num_inputs, 	#num. inputs from prev. layer.
				 num_outputs, 	#num. outputs.
				 use_relu=True):#use rectified linear unit
	#create new weights and biases
	weights=new_weights(shape=[num_inputs,num_outputs])
	biases=new_biases(length=num_outputs)
	#calculate the layer as the matrix multiplication of
	#the input and weights, and then add the bias-value
	layer=tf.matmul(input,weights)+biases
	#use ReLU
	if use_relu:
		layer=tf.nn.relu(layer)
	return layer

#placeholder variables
x=tf.placeholder(tf.float32,shape=[None,img_size_flat],name='x')
x_image=tf.reshape(x,[-1,img_size,img_size,num_channels])
y_true=tf.placeholder(tf.float32,shape=[None,num_classes],name='y_true')
y_true_cls=tf.argmax(y_true,axis=1)

#convolutional layer 1
layer_conv1,weights_conv1=\
	new_conv_layer(input=x_image,
				   num_input_channels=num_channels,
				   filter_size=filter_size1,
				   num_filters=num_filters1,
				   use_pooling=True)
layer_conv1

#convolutional layer 2
layer_conv2,weights_conv2=\
	new_conv_layer(input=layer_conv1,
				   num_input_channels=num_filters1,
				   filter_size=filter_size2,
				   num_filters=num_filters2,
				   use_pooling=True)
layer_conv2

#flatten layer
layer_flat, num_features = flatten_layer(layer_conv2)
#layer_flat
#num_features

#fully-connected layer 1
layer_fc1=new_fc_layer(input=layer_flat,
					   num_inputs=num_features,
					   num_outputs=fc_size,
					   use_relu=True)
#layer_fc1

#fully-connected layer 2
layer_fc2=new_fc_layer(input=layer_fc1,
					   num_inputs=fc_size,
					   num_outputs=num_classes,
					   use_relu=False)
#layer_fc2

#predicted classes
y_pred=tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, axis=1)

#cost-function to be optimized
cross_entropy=tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
													  labels=y_true)
cost=tf.reduce_mean(cross_entropy)

#optimization method
optimizer=tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

#performance measures
correct_prediction=tf.equal(y_pred_cls,y_true_cls)
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#tensorflow run
session=tf.Session()
session.run(tf.global_variables_initializer())

#helper-function to perform optimization iterations
train_batch_size=64
#counter for total number of iterations performed so far.
total_iterations=0
def optimize(num_iterations):
	#ensure we update the global variable rather than a local copy
	global total_iterations
	#start-time used for printing time-usage below.
	for i in range(total_iterations,
				   total_iterations+num_iterations):
		#get a batch of training examples.
		x_batch,y_true_batch,_=data.random_batch(batch_size=train_batch_size)
		feed_dict_train={x:x_batch,y_true:y_true_batch}
		session.run(optimizer,feed_dict=feed_dict_train)
		#print status every 100 iterations.
		if i%100==0:
			#calculate the accuracy on the training-set
			acc=session.run(accuracy,feed_dict=feed_dict_train)
			#message for printing
			msg="Optimization Iteration:{0:>6}, Training Accuracy:{1:>6.1%}"
			print(msg.format(i+1,acc))
	#update the total number of iteartions performed
	total_iterations+=num_iterations#ending time
	end_time=time.time()
	#difference between start and end-times
	time_dif=end_time-start_time
	#print time-usage
	print("Time Usage: "+str(timedelta(second=int(roung(time_dif)))))

 #helper-function to plot example errors

def plot_example_errors(cls_pred, correct):
    # This function is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set.

    # Negate the boolean array.
    incorrect = (correct == False)
    
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

#helper-function to plot confusion matrix
def plot_confusion_matrix(cls_pred):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the true classifications for the test-set.
    cls_true = data.y_test_cls
    
    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

#helper-function for showing the performance
# Split the test-set into smaller batches of this size.
test_batch_size = 256

def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):

    # Number of images in the test-set.
    num_test = data.num_test

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test)

        # Get the images from the test-set between index i and j.
        images = data.x_test[i:j, :]

        # Get the associated labels.
        labels = data.y_test[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: images,
                     y_true: labels}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    cls_true = data.y_test_cls

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)

#performance before any optimization
print_test_accuracy()
