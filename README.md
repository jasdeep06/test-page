# Into-Backpropagation

In the [previous post](https://jasdeep06.github.io/posts/towards-backpropagation/),we learnt to appericiate the beauty of derivatives and their effect on update rule which is given by-

$$\Large{a}={a}+{h}*\frac{\partial f}{\partial a}$$
$$\Large{b}={b}+{h}*\frac{\partial f}{\partial b}$$

where 
$$\Large{f}={f}({a},{b})$$



Although the case of single node system helps us capture the intuition behind the effect of change in input on the outpute of node,it is pretty useless if considered in isolation.We need to scale up our network.Let us consider case of nested nodes as shown below-

![nested](https://github.com/jasdeep06/jasdeep06.github.io/blob/master/posts/into-backpropagation/images/nested.png?raw=true)

Here there are two nodes,out of which the first one(from the left) accepts two inputs `a` and `b` and performs addition operation on them to return the output `d`.The second node accepts `d` and a new input `c` as inputs and performs product operation on them to gives `f` as final output.
This system can be represented in python:
		    
    def product(x,y):
    	return x*y
    def addition(x,y):
    	return x+y

	a=5
	b=-3
	c=-2
	d=addition(a,b)
	f=product(d,c)	#outputs -4

#### Aim
Our aim is still the same as was in last post viz;we want to manipulate the values of our inputs `a`,`b`,`c` in such a way that the value of output `f` increases.

Not only will we achieve the above aim but in that process we will slowly slide into backpropagation and go through the concept intuitively.Note that this post will be slightly more mathematical than the last one but all the concepts used are from the described intuitivly in the [previous post](https://jasdeep06.github.io/posts/towards-backpropagation/).

###### Lets get started!!

This nested system might seem a bit intimidating at first.Where do we start?Well,we know the update rules from the last post that involve derivatives of output with respect to input.Let us list down these update rules for our inputs `a`,`b` and `c`-

$$\Large{a}={a}+{h}*\frac{\partial f}{\partial a}$$
$$\Large{b}={b}+{h}*\frac{\partial f}{\partial b}$$
$$\Large{c}={c}+{h}*\frac{\partial f}{\partial c}$$

We somehow want to compute the three derivatives (\\frac{\partial f}{\partial a}\),(\\frac{\partial f}{\partial b}\) and (\\frac{\partial f}{\partial c}\)

Let us look at the relations among various variables in our system.We can easily write-

$$\Large{f}={d}*{c}$$
$$\Large{d}={a}+{b}$$

Now let us use the analytical gradient to calculate derivatives from the above relations.(Refer [Derivative rules](https://www.mathsisfun.com/calculus/derivatives-rules.html)).

Consider the relation $$\Large{f}={d}*{c}$$

Differentiating this relation we get-

$$\Large\frac{\partial f}{\partial d}={c}$$
$$\Large\frac{\partial f}{\partial c}={d}$$

Differentiating the relation $$\Large{d}={a}*{b}$$ we get-

$$\Large\frac{\partial d}{\partial a}={1}$$
$$\Large\frac{\partial d}{\partial b}={1}$$

Observe that derivatives for addition node is 1.This makes intuitive sense too.If you try to increase input to an addition node by a quantity h,then the output value will increase by same quantity.Thus normalised change i.e. the derivative is 1.

We now have the values of (\\frac{\partial f}{\partial d}\),(\\frac{\partial f}{\partial c}\),(\\frac{\partial d}{\partial a}\)and (\\frac{\partial d}{\partial b}\).We somehow have to use these values to compute the values of (\\frac{\partial f}{\partial a}\),(\\frac{\partial f}{\partial b}\) and (\\frac{\partial f}{\partial c}\)





