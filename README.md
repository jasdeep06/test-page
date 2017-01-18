# Further into Backpropagation

In the [previous post](https://jasdeep06.github.io/posts/Lets-practice-backpropagation/) we applied chain rule(funkily called backpropagation!) to systems with complex functions.In this post we will apply backpropagation to neural networks.We will start with a two layered network and then extend our discussion to three layered network.Note that instead of derivation of mathematical formulaes we will focus on intuitive sense of it.We will look to explore implementation of backpropagation with multiple inputs represented in form of matrix of training data.


##### Lets get started!!!

Consider the network:

The figure consists of a two layered network.The first layer is the input layer and contains 3 nodes.The second layer is output layer and contains two nodes.In a standard neural network,the sigmoid layer is a part of output layer.For clarity of concept I have drawn it as a separate layer.The sigmoid layer quashes the output values in the range of 0 and 1.The two layers are connected to each other with weights which are represented by edges in the figure.Each node of first layer is connected to every node of second layer.

For those who don't know how neural networks work here is a short description(Note that this is just a very simple and crude explaination and is sufficient for this post.However,to appriciate the exact mechanism behind it consult  [other](https://en.wikipedia.org/wiki/Artificial_neural_network) resources too):
The input to the network is pumped through input layer.Here our inputs would be 3 dimensional as there are three nodes in our layer.These dimensions are also referred to as features.The inputs are multiplied with randomly initialized weights usually in form of matrix.The output of this matrix multiplication is subjected to a sigmoid function.The sigmoidal outputs are used to generate cost function.A cost function is a function that is a measure of deviation of our output from the actual value during the training of network.For this post we will use [cross-entropy](https://en.wikipedia.org/wiki/Cross_entropy) cost function.The nodes in output layer of our network represent different classes to which the input has to be classified.During the training of network we have a label corresponding to every input.This label represents the true class of that input.
###### Aim
Our aim would be adjust the value of weights such that our cost function in minimum(i.e. the deviation of our output from actual value is minimum).

So where do we start?If you are following the posts in series then you would know the answer to this!Thats right!Our update rules.Here we have to manipulate the values of weights to decrease the value of our cost thereby minimizing it.Thus updating weights to decrease the cost:

(dollar)\Large{W_{ij}}={W_{ij}}-h(mul)\frac{\partial C}{\partial W_{ij}}(dollar)

where (dbo)\Large{W_{ij}}(dbc) denotes weight connecting (dbo)\Large{i^th}(dbc) node in input layer to (dbo)\Large{j^th}(dbc) node in output layer.

We have to somehow find the value of (dbo)\frac{\partial C}{\partial W_{ij}}(dbc) and fill it in our update rule which would decrease the cost function(due to the minus sign).

Now that we know what we desire,lets analyse our cost function `C`.To do this we must first forward propagate through our network to analyse the dependencies of our cost function.Here we would take only one training example which would enable us to appriciate the intricacies better and from there we would extend this to multiple training examples.

#### Forward Propagation
The training example on which this analysis will be based is (dbo)\Large{x_1},{x_2},x_3}(dbc) which can be represented in form of 1X3 matrix as:
(dollar)\Large{X}=\begin{bmatrix}x_1 & x_2& x_3\end{bmatrix}(dollar)
The weights can also be represented in form of a 3x2 matrix as (dollar)\Large{W}=\begin{bmatrix}W_{11} & W_{12} \\W_{21} & W_{22}\\W_{31} & W_{32}\end{bmatrix}(dollar)
The output before application of sigmoid can easily be found out by multiplying the two matrix to generate a 1x2 output matrix:
(dollar)\begin{bmatrix}x_1 & x_2& x_3\end{bmatrix}(mul)\begin{bmatrix}W_{11} & W_{12} \\W_{21} & W_{22}\\W_{31} & W_{32}\end{bmatrix}=\begin{bmatrix}x_1W_{11}+x_2W_{21}+x_3W_{31} & x_1W_{12}+x_2W_{22}+x_3W_{32}\end{bmatrix}(dollar)

Note that the output matrix is a 1x2 matrix with two elements of which one belongs to the first node and other to the second node for a single training example.Let this matrix be represented as:
(dollar){y}=\begin{bmatrix}x_1W_{11}+x_2W_{21}+x_3W_{31} & x_1W_{12}+x_2W_{22}+x_3W_{32}\end{bmatrix}\equiv{\begin{bmatrix}y_1&y_2\end{bmatrix}}(dollar) where (dbo){y_1},{y_2},{y}(dbc) are placeholders to facilitate understanding.

Applying sigmoid on this matrix we get:

(dollar){y_o}=\begin{bmatrix}sigmoid(y_1)&sigmoid(y_2)\end{bmatrix}\equiv{\begin{bmatrix}y_{o1}&y_{o2}\end{bmatrix}}(dollar) where (dbo){y_{o1}},{y_{o2}},{y_o}(dbc) are placeholders.


### Cost function
(dbo){y_{o1}},{y_{o2}}(dbc) obtained from forward propagation are used in cost function.Here we are using [cross-entropy](https://en.wikipedia.org/wiki/Cross_entropy) cost function which is given by:

(dollar)\Large{C}=-\frac{1}{N}\sum_i{p_i(mul)log(q_i)}(dollar)

where (dbo){i}(dbc) is the number of catagories for classification(equivalent to number of nodes in output unit),(dbo){p_i}(dbc) is true label of that class and (dbo){q_i}(dbc) is predicted value and N is total number of training examples.

For this system (dbo){i}=2(dbc) (as number of nodes in output layer=2 i.e. 2 classification classes).(dbo)q_1=y_{o1}(dbc) and (dbo)q_2=y_{o2}(dbc) as they are the predicted value of the two classes.When we train our network against training data,the label corresponding to a training example would be known.The label represents the class that the training example belongs to.Here let us assume that the training example belongs to the first class.This would make the label values of all other classes to be zero and of the first class as 1.Thus here (dbo){p_1}=1(dbc) and (dbo){p_2}=0(dbc).Expanding our cost function for i=2,we get:
(dollar)\Large{C}=-\frac{1}{N}(mul)(p_1log(q_1)+p_2log(q_2))(dollar)
Putting corresponding values we get:

(dollar)\Large{C}=-(p_1(mul)log(y_{o1})+p_2(mul)log(y_{o2}))(dollar)

### Backpropagation
Let us first revisit the matrices that we have:
A pre sigmoidal output matrix (dollar){y}={\begin{bmatrix}y_1&y_2\end{bmatrix}}(dollar)
A post sigmoidal output matrix (dollar){y_o}{\begin{bmatrix}y_{o1}&y_{o2}\end{bmatrix}}(dollar)
Our weight matrix (dollar)\begin{bmatrix}W_{11} & W_{12} \\W_{21} & W_{22}\\W_{31} & W_{32}\end{bmatrix}(dollar)
Our input matrix (dollar){X}=\begin{bmatrix}x_1 & x_2& x_3\end{bmatrix}(dollar)

Remember that we had to find the value of (dbo)\frac{\partial C}{\partial W_{ij}}(dbc) to put into update rule which would decrease the cost.We will find this value by applying chain rule as we have done in previous posts but the only difference here will be that we would be dealing with matrices instead of individual variables.While applying chain rule we will focus on the parallelism in dealing with matrices and variables thereby making the transition to matrices smoother and intuitive.

If we look at our network figure and our cost function,we notice that our cost function is a function of (dbo){y_{o1}},{y_{o2}}(dbc) which is represented in (dbo){y_{o}}(dbc} matrix which is a function of (dbo){y_{1}},{y_{2}}(dbc) which is represented in (dbo){y}(dbc} matrix which is function of inputs and weights.

Let us move back through the system from cost function to input layer and write the chain rule.First we will move back from one node to other and alongside it we will represent layerwise movement in terms of matrices.

 - From cost function towards sigmoid layer
		 (dollar)\Large{C}=-((p_1(mul)log(y_{o1})+p_2(mul)log(y_{o2}))(dollar))
		 We can easily write the derivatives:
		 (dollar)\Large\frac{\partial C}{\partial y_{o1}}=-(\frac{p_1}{y_{o1})
		 (dollar)\Large\frac{\partial C}{\partial y_{o2}}=-(\frac{p_2}{y_{o2})
		 We can represent this in form of a matrix:
		 (dollar)\Large\frac{\partial C}{\partial y_{o}}=\begin{bmatrix} -(\frac{p_1}{y_{o1}})& -(\frac{p_2}{y_{o2}})\end{bmatrix}(dollar)
		 
 - Through the sigmoid layer-
	 From our experiences in previous posts we can easily write the sigmoid derivatives as:
	  (dollar)\Large\frac{\partial y_{o1}}{\partial y_1}=\sigma({y_1})(mul)(1-\sigma({y_1}))(dollar)
	  (dollar)\Large\frac{\partial y_{o2}}{\partial y_2}=\sigma({y_2})(mul)(1-\sigma({y_2}))(dollar)
	  We can represent this in matrix form:
	   (dollar)\Large\frac{\partial y_o}{\partial y}=\begin{bmatrix} \sigma({y_1})(mul)(1-\sigma({y_1}))&\sigma({y_2})(mul)(1-\sigma({y_2}))\end{bmatrix}(dollar)
 - Through the pre sigmoidal output layer towards the weights
	 We know the relations (dollar){y_1}=x_1W_{11}+x_2W_{21}+x_3W_{31}(dollar) and (dollar){y_2}=x_1W_{12}+x_2W_{22}+x_3W_{32}(dollar).From these we can easily find the derivatives:
	 (dollar)\Large\frac{\partial {y_1}}{\partial W_{11}}=x_1(dollar)
	 (dollar)\Large\frac{\partial {y_1}}{\partial W_{21}}=x_2(dollar)
	 (dollar)\Large\frac{\partial {y_1}}{\partial W_{31}}=x_3(dollar)
	 (dollar)\Large\frac{\partial {y_2}}{\partial W_{12}}=x_1(dollar)
	 (dollar)\Large\frac{\partial {y_2}}{\partial W_{22}}=x_2(dollar)
	 (dollar)\Large\frac{\partial {y_2}}{\partial W_{32}}=x_3(dollar)

Let us chain all the derivatives elementwise first.Then we will convert it into matrix representation. 
	 (dollar)\Large\frac{\partial {C}}{\partial W_{11}}=\frac{\partial C}{partial {y_{o1}}(mul)\frac{\partial y_{o1}}{\partial y_1}(mul)\frac{\partial {y_1}}{\partial W_{11}}(dollar)
	 Putting the respective values we get-
	 (dollar)\Large\frac{\partial {C}}{\partial W_{11}}=\frac{-p_1}{y_{o1}}(mul)\sigma({y_1})(mul)(1-\sigma({y_1}))(mul)x_1
	 Similarly we can write this for all the W's and place them in a matrix as-
	 (dollar)\Large\frac{\partial {C}}{\partial W}=\begin{bmatrix} -(\frac{p_1}{y_{o1}})(mul)\sigma({y_1})(mul)(1-\sigma({y_1}))(mul)x_1& -(\frac{p_2}{y_{o2}})(mul)\sigma({y_2})(mul)(1-\sigma({y_2}))(mul)x_1\\-(\frac{p_1}{y_{o1}})(mul)\sigma({y_1})(mul)(1-\sigma({y_1}))(mul)x_2&-(\frac{p_2}{y_{o2}})(mul)\sigma({y_2})(mul)(1-\sigma({y_2}))(mul)x_2\\-(\frac{p_1}{y_{o1}})(mul)\sigma({y_1})(mul)(1-\sigma({y_1}))(mul)x_3&-(\frac{p_2}{y_{o2}})(mul)\sigma({y_2})(mul)(1-\sigma({y_2}))(mul)x_3\end{bmatrix}(dollar)
Observe the above matrix.It is nothing but the combination (dbo){X^T}*{\frac{\partial C}{\partial y_{o}}.\frac{\partial y_o}{\partial y}(dbc)

where T denotes transpose of matrix and " * " denotes matrix product while " . "denotes element-wise product.
Now we can put this expression in our update rule:
(dollar)\Large{W}=W-h*{X^T}*{\frac{\partial C}{\partial y_{o}}.\frac{\partial y_o}{\partial y}(dollar)
The python representation can be given as:

    import numpy as np
	import random

	def sigmoid(x):
    return 1/(1+np.exp(-x))
	def derivative_sigmoid(x):
    return np.multiply(sigmoid(x),(1-sigmoid(x)))

	#initialization
	X=np.matrix("2,4,-2")
	W=np.random.normal(size=(3,2))
	#label
	ycap=[0]
	#number of training of examples
	num_examples=1
	#step size
	h=.01
	#forward-propogation
	y=np.dot(X,W)
	y_o=sigmoid(y)
	#loss calculation
	loss=-np.sum(np.log(y_o[range(1),ycap]))
	print(loss)
	#backprop starts
	temp1=np.copy(y_o)
	#implementation of derivative of cost function with respect to y_o
	temp1[range(num_examples),ycap]=1/-(temp1[range(1),ycap])
	temp=np.zeros_like(y_o)
	temp[range(num_examples),ycap]=1
	#derivative of cost with respect to y_o
	dcost=np.multiply(temp,temp1)
	#derivative of y_o with respect to y
	dy_o=derivative_sigmoid(y)
	#element-wise multiplication
	dgrad=np.multiply(dcost,dy_o)
	dw=np.dot(X.T,dgrad)
	#weight-update
	W-=h*dw
	#forward prop again with updated weight to find new loss
	y=np.dot(X,W)
	yo=sigmoid(y)
	loss=-np.sum(np.log(yo[range(1),ycap]))
	print(loss)

Above program shows only one iteration of backpropagation and can be extend to multiple iterations to minimize the cost function.

