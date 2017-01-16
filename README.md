# Lets practice Backpropagation

In the [previous post](https://jasdeep06.github.io/posts/into-backpropagation/) we went through a system of nested nodes and analysed the update rules for the system.We also went through the intuitive notion of backpropagation and figured out that it is nothing but applying chain rule over and over again.In this post we will apply backpropogation to systems with complex functions so that the reader gets comfortable with chain rule and its applications to complex systems.

##### Lets get started!!!

Lets start with a single node system but this time with a complex function:
![sigmoid.png](https://github.com/jasdeep06/jasdeep06.github.io/blob/master/posts/futher-into-backpropagation/images/sigmoid.png?raw=true)

This system can be represented as:
	
    import numpy as np
	def sigmoid(x):
    	return 1/(1+np.exp(-x))
	a=-2
	f=sigmoid(a)
	print(f) #outputs 0.1192

#### Aim
Our aim essentially remains the same as in previous posts viz:we have to increase the value of output(`f`) by manipulating the values of input(`a`).

The node in above figure represents a sigmoid unit.Sigmoid function is expressed as-
(dollar)\Large{\sigma({x})=\frac{1}{1+e^{-x}}(dollar)

>*If you analyse the sigmoid function,you will notice that irrespective of the input value,this function gives a value between 0 and 1 as output.It is a nice property to have as it gives a sense of probabilistic values of inputs.It was once the most popular activation function used in neural network design.*

Where do we start?Once again let us look at our update rule:
(dollar)\Large{a}={a}+{h}(mul)\frac{\df}{\da}(dollar)

(Here we are using total derivative((db)\Large\d(db)) instead of partial derivative((db)\Large\partial(db)) because here our node is a function of only one variable i.e. `a`.And thats the only difference there is between the two.Rest every thing remains the same.)

We just have to find value of the derivative (db)\Large\frac{df}{da}(db).
If you are aware of basic rules of calculus(Refer [Derivetive rules](https://www.mathsisfun.com/calculus/derivatives-rules.html)) then you can easily find the derivative of sigmoid with respect to a.If you are new to calculus then just remember for now that derivative of sigmoid funtion is given by:


(dollar)\Large\frac{d\sigma}{da}=(\sigma({a}))(mul)(1-\sigma({a}))(dollar)

Let us put this value in our update rule:
(dollar)\Large{a}={a}+h(mul)(\sigma({a}))(mul)(1-\sigma({a}))(dollar)

Using this update rule in python:

	import numpy as np
    def sigmoid(x):
    	return 1/(1+np.exp(-x))
	def derivative_sigmoid(x):
    	return sigmoid(x)*(1-sigmoid(x))
	a=-2
	h=.1
	a=a+h*derivative_sigmoid(a)
	f=sigmoid(a)
	print(f)  #outputs 0.1203
    
The above program outputs 0.1203 which is greater than 0.1192.It worked!!!

Let us take the discussion one notch above.Consider the system:
![nested-sigmoid](https://github.com/jasdeep06/jasdeep06.github.io/blob/master/posts/futher-into-backpropagation/images/nested-sigmoid.png?raw=true)
The system consists of three inputs `a`,`b` and `c`.The former two pass through an addition node and give output `d` which products with the input`c` to generate `d` which is passed thwough sigmoid node to give final output `f`.Let us represent this system in python:

	import numpy as np
	def addition(x,y):
    	return x+y
	def product(x,y):
    	return x*y
	def sigmoid(x):
    	return 1/(1+np.exp(-x))

	a=1
	b=-2
	c=-3
	d=addition(a,b)
	e=product(c,d)
	f=sigmoid(e)
	print(f)  #outputs 0.952574
    
Our aim essentially remains the same viz:to tweak the values of input `a`,`b` and `c` in order to increase the value of `f`.Once again like our previous approaches,let us look at our update rules:

(dollar)\Large{a}={a}+{h}(mul)\frac{\partial f}{\partial a}(dollar) 
(dollar)\Large{b}={b}+{h}(mul)\frac{\partial f}{\partial b}(dollar)
(dollar)\Large{c}={c}+{h}(mul)\frac{\partial f}{\partial c}(dollar)

We have to somehow find the values of the derivatives (db)\Large\frac{\partial f}{\partial a}(db),(db)\Large\frac{\partial f}{\partial b}(db) and (db)\Large\frac{\partial f}{\partial c}(db).

Using chain rule described in [previous post](https://jasdeep06.github.io/posts/into-backpropagation/) we can write:
(dollar)\Large\frac{\partial f}{\partial a}=\frac{\partial f}{\partial e}(mul)\frac{\partial e}{\partial d}(mul)\frac{\partial d}{\partial a}(dollar)
(dollar)\Large\frac{\partial f}{\partial b}=\frac{\partial f}{\partial e}(mul)\frac{\partial e}{\partial d}(mul)\frac{\partial d}{\partial b}(dollar)
(dollar)\Large\frac{\partial f}{\partial c}=\frac{\partial f}{\partial e}(mul)\frac{\partial e}{\partial c}(dollar)

>*This is a good example to get an intuition about chain rule.Observe how in order to compute derivatives of `f` with respect to various inputs,we are just travelling to those inputs from `f` and multiplying(chaining) the derivatives that we encounter as we reach the input.*

Lets start finding the values of derivatives:

Let us traverse the system from output to input i.e. backward.While crossing the sigmoid node the value of (db)\Large\frac{\partial f}{\partial e}(db) can be written easily as (db)\Large(\sigma({e}))(mul)(1-\sigma({e}))(db).Further while crossing the product node the values of (db)\Large\frac{\partial e}{\partial d}(db) and (db)\Large\frac{\partial e}{\partial c}(db) can easily be written as (db)\Large{c}(db) and (db)\Large{d}(db) respectively.Further while crossing the addition node the values of (db)\Large\frac{\partial e}{\partial d}(db) and (db)\Large\frac{\partial e}{\partial c}(db) can be easily written as (db)\Large{1}(db) and (db)\Large{1}(db) respectively.If you are having trouble in getting your head around these derivatives I suggest you to have a look at [first](https://jasdeep06.github.io/posts/towards-backpropagation/) and the [second](https://jasdeep06.github.io/posts/into-backpropagation/) post of this series.

Writing our update rules we get:
(dollar)\Large\frac{\partial f}{\partial a}=(\sigma({e}))(mul)(1-\sigma({e}))(mul){c}(mul){1}(dollar)
(dollar)\Large\frac{\partial f}{\partial b}=(\sigma({e}))(mul)(1-\sigma({e}))(mul){c}(mul){1}(dollar)
(dollar)\Large\frac{\partial f}{\partial c}=(\sigma({e}))(mul)(1-\sigma({e}))(mul){d}(dollar)

Let us represent this in python:

import numpy as np
	def addition(x,y):
    	return x+y
	def product(x,y):
    	return x*y
	def sigmoid(x):
    	return 1/(1+np.exp(-x))
	def derivative_sigmoid(x):
    	return sigmoid(x)*(1-sigmoid(x))

	#initialization
	a=1
	b=-2
	c=-3
	#forward-propogation
	d=addition(a,b)
	e=product(c,d)
	#step size
	h=0.1
	#derivatives
	derivative_f_wrt_e=derivative_sigmoid(e)
	derivative_e_wrt_d=c
	derivative_e_wrt_c=d
	derivative_d_wrt_a=1
	derivative_d_wrt_b=1
	#backward-propogation (Chain rule)
	derivative_f_wrt_a=derivative_f_wrt_e*derivative_e_wrt_d*derivative_d_wrt_a
	derivative_f_wrt_b=derivative_f_wrt_e*derivative_e_wrt_d*derivative_d_wrt_b
	derivative_f_wrt_c=derivative_f_wrt_e*derivative_e_wrt_c
	#update-parameters
	a=a+h*derivative_f_wrt_a
	b=b+h*derivative_f_wrt_b
	c=c+h*derivative_f_wrt_c
	d=addition(a,b)
	e=product(c,d)
	f=sigmoid(e)
	print(f)  #prints 0.9563

The output of above program is 0.9563 which is greater than 0.9525.

