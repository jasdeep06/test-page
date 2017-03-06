# CNNs in Tensorflow (cifar-10 implementation)(1/3)

Its been quite a while since I last posted as I was busy with exams at the college.Now that the carnage is over,you can expect posts in quick succession throughout the month.

In the [previous](https://jasdeep06.github.io/posts/basics-of-cnns/) post we discussed the cogs on which the system of Convolutional neural network(CNN) works.In this post we will implement a CNN in Tensorflow on the cifar-10 dataset.

As this is the first real implementation of this blog,I will discuss cifar-10 model in great detail(input-pipeline,training,evaluation etc.) and we will use these details in implementing other models as basic structure of models generally remains the same.

In this post we will discuss the representation of tensors and reshaping and slicing operations on them.We will also try to visualise cifar-10 dataset.

#### Lets get started!!!!

## Operating on tensors in Tensorflow


If you understand the concepts of multidimensional array,reshaping and slicing operations in tensorflow you can skip this section.
Before taking a look at reshaping and slicing operations in tensorflow,it would be great if we become comfortable with the representation of multidimensional arrays(or tensors).This would let us visualise dimensional information associated with each element of tensors which is essential to master the slicing operations.In all the discussions we will be using zero based indexing as is the case in python.

To understand shapes and dimensions of a tensor,we will use the method `tf.Tensor.get_shape()` which returns the shape of tensor in form of tuple of tensor dimensions.For example,an output of (2,1,4) would mean that the tensor has 3 dimensions(length of the tuple) and the zeroth dimension has 2 elements,first has 1 element and the second dimension has 4 elements.One of the tensor which satisfies this output can be given by-	
		
		T=[[[11,12,13,14]],[[15,16,17,18]]]

Now let us dissect above tensor representation to reach the conclusion about its shape.'Dimensions' in above notation is depicted by the square brackets.Encountering an opening bracket would mean going to the next higher dimension and encountering a closing brace would mean going down to immediate lower dimension.The 'comma' is used to separate elements of same dimension.

Let us traverse the above tensor from left to right.As we encounter three  consecutive opening brackets,we reach the second dimension which contains four elements 11,12,13 and 14.As we move further right,we encounter two closing braces which takes us back to zeroth dimension.Thus `[[11 12 13 14]]` was the zeroth element in our zeroth dimension.Similarly first element in the zeroth dimension would be `[[15 16 17 18]]`.
Let us try to visualise the allotment of indices to these tensor elements:
![tensor_visual](https://github.com/jasdeep06/jasdeep06.github.io/blob/master/posts/cnn-in-tensorflow/images/tensor_vis.png?raw=true)
Above figure is pretty self-explanatory.To get comfortable with indices let us write some code to access elements of above tensor `T`.While you analyse the output of code,compare them with above figure to make yourself see how elements of tensors are accessed.
	
	import tensorflow as tf
	T=tf.constant([[[11,12,13,14]],[[15,16,17,18]]])
	#start a interactive session
	sess=tf.InteractiveSession()
	print("Our tensor is",T.eval())
	print("Shape of the tensor is ",T.get_shape())
	print("T[0] is ",T[0].eval())   #equivalent to T[0,:,:]
	print("T[1,0] is",T[1,0].eval())    #equivalent to T[1,0,:]
	print("T[1,0,0] is ",T[1,0,0].eval())
	print("T[1,0,:] is",T[1,0,:].eval())
	
	
Output:

	Our tensor is [[[11 12 13 14]]
					[[15 16 17 18]]]
	Shape of the tensor is  (2, 1, 4)
	T[0] is  [[11 12 13 14]]
	T[1,0] is [15 16 17 18]
	T[1,0,0] is  15
	T[0,0,:] is [11 12 13 14]
	
By comparing the output of program with the figure above,we are able to visualise the arrangement of tensor elements which will be very helpful in reshaping and slicing operations.

### Reshaping in tensorflow

Reshaping operations in tensorflow are carried out by reshape function defined as-

	tf.reshape(tensor,shape,name=None)
	
The reshape function accepts the input `tensor` and reshapes it to the shape `shape`.Let us look at an example:
	
	import tensorflow as tf
	T=tf.constant([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]],[[13,14,15],[16,17,18]])
	reshaped=tf.reshape(T,[2,1,9])
	sess=tf.InteractiveSession()
	print(A.eval())

Output

	[[[1 2 3 4 5 6 7 8 9]] 
	[[10 11 12 13 14 15 16 17 18]]]

To visualise this reshaping,keep in mind that number of elements while reshaping remains same.There are 18 elements in the tensor `T`.While reshaping always concentrate on the innermost dimension(2nd dimension in this case).Here in the new shape,the innermost dimension must have 9 elements.Allot 9 elements in order to the innermost dimension and work your way outward.


### Slicing in tensorflow

Slicing operations in tensorflow are carried out by slice function defined as-

	tf.slice(input,begin,size,name=None)

The slice function extracts a slice of size `size` from `input` tensor starting at location specified by `begin`.Both `size` and `begin` are represented in form of vectors where `size[i]` is the number of elements of the i<sup>th</sup> dimension of input that you want to slice and begin[i] is the offset into the i<sup>th</sup> dimension of input that you want to slice from.Let us make this clear with help of some examples:
	
	import tensorflow as tf
	T=tf.constant([[[10,11,12],[13,14,15]],
					[[16,17,18],[19,20,21]],
					[[22,23,24],[25,26,27]]])
	sess=tf.InteractiveSession()
	S=tf.slice(T,[0,1,1],[1,1,2])
	print(S.eval())
	
Output:
	
	[[[14 15]]]
	
	
In above example our `begin` and `size` vectors were `[0,1,1]` and `[1,1,2]` respectively.The slice operation is operated by combination of these two vectors.The zeroth element of `begin` vector(i.e. 0) tells slice function to start slicing from 0<sup>th</sup> element of zeroth dimension of `T` and the zeroth element of `size` vector(i.e. 1) informs to take 1 element of the zeroth dimension of `T` which would give us `[[[10,11,12],[13,14,15]]]` to work with.The first element of `begin` vector(i.e. 1) tells slice function to start slicing from 1<sup>st</sup> element of first dimension of `T` and the first element of `size` vector(i.e. 1) informs to take 1 element of the first dimension of `T` which would give us `[[[13,14,15]]]` to work with.The second element of `begin` vector(i.e. 1) tells slice function to start slicing from 1<sup>st</sup> element of second dimension of `T` and the second element of `size` vector(i.e. 2) informs to take 2 elements of the second dimension of `T` which would give us `[[[14,15]]]` as our sliced tensor.


Thus slicing operations may be seen as cumulative effect of slicing commands in all the dimensions individually.Try to convince yourself,the outputs of examples below:


	import tensorflow as tf
	T=tf.constant([[[10,11,12],[13,14,15]],
					[[16,17,18],[19,20,21]],
					[[22,23,24],[25,26,27]]])
	sess=tf.InteractiveSession()
	S1 = tf.slice(T, [1, 1, 2], [1, 1, 1])
	print("The value of tf.slice(T,[1, 1, 2],[1, 1, 1]) is ",S1.eval())
	S2=tf.slice(T,[0,0,0],[3,2,1])
	print("The value of tf.slice(T,[0,0,0],[3,2,1]) is ",S2.eval())


Output:

	
	The value of tf.slice(T, [1, 1, 2], [1, 1, 1]) is  [[[21]]]
	The value of tf.slice(T,[0,0,0],[3,2,1]) is  
	[
	[[10]
  	 [13]]
  	
  	[[16]
  	[19]]
  	
  	[[22]
    [25]]
    ]

Now that we have taken a look at slicing operations,lets jump back to our cifar-10 implementation.


## Dataset


The cifar-10 dataset consists of 60000 32X32 colour images in 10 classes,with 6000 images per class.Out of these 60000 images,50000 are training images and 10000 are test images.

The dataset can be found [here](https://www.cs.toronto.edu/~kriz/cifar.html).You may also want to see this [tech-report](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf).We will use the binary dataset for implementation.


### Anatomy of dataset


The binary dataset consists of data distributed in 6 files of which 5 contain training data(10000 training images each) and the remaining file contains test data(10000 test images).

As each of these images are 32X32 colour images with 3 channels Red,Green and Blue(RGB),they will have 32X32X3=3072 image bytes(32X32=1024 for each R,G,B).Also each image is associated with a label(denoting its class) which occupies 1 byte.Thus each image will be defined by 3072(image)+1(label)=3073 bytes.

As each image file contains 10000 such images so they are 30730000 bytes long.The arrangement of images in these files can be visualised as:
![dataset_visual](https://github.com/jasdeep06/jasdeep06.github.io/blob/master/posts/cnn-in-tensorflow/images/dataset.png?raw=true)

Each image is arranged in form of rows with first byte as the label byte followed by 1024 pixel bytes each of Red,Green and Blue channels respectively.There are 10000 such rows.The image rows are consecutive i.e. there is nothing delimiting image rows.

Let us end this post here.The next post will be dedicated to input pipeline in tensorflow.We will try to read and process the cifar-10 dataset.The next post will be live in a couple of days.Stay tuned!!!

	
