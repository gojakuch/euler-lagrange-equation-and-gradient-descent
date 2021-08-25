# Solving Basic Variational Calculus Problems Numerically with Gradient Descent
## Abstract and Disclaimer
This repository contains an implementation of a method I have come up with for approximating solutions to certain Euler-Lagrange equations in the Julia Language. I will elaborate on the method and the intuition further in this text, however, as of now, everything I know about mathematics is a result of my self-study and I am not a professional mathematician yet (in fact, when I made this I was barely even an undergraduate) which means that this is not the best numerical method for such tasks. It could also be a duplicate of some other method. It is simply what I thought of right after I heard about the Calculus of Variations. In case you would like to look through something better, see the last section with helpful web pages.
## Problem Setup
The simplest problem of 1-dimensional Variational Calculus is to minimize a functional <img src="https://latex.codecogs.com/gif.latex?I" title="I" /> which is defined by
1) <img src="https://latex.codecogs.com/gif.latex?I&space;=&space;\int_{a}^{b}F[x,y,y']dx" title="I = \int_{a}^{b}F[x,y,y']dx" />\
to find an optimal function <img src="https://latex.codecogs.com/gif.latex?y" title="y(x)" />. In the expression above <img src="https://latex.codecogs.com/gif.latex?a" title="a" /> and <img src="https://latex.codecogs.com/gif.latex?b" title="b" /> are points between which we minimize our functional.\
By setting the derivative of <img src="https://latex.codecogs.com/gif.latex?I" title="I" /> to <img src="https://latex.codecogs.com/gif.latex?0" title="0" /> and performing some intriguing manipulations, which are beyond our interest for now, we get the Euler-Lagrange Equation that our function <img src="https://latex.codecogs.com/gif.latex?y" title="y(x)" /> must satisfy:
2) <img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;F}{\partial&space;y}-\frac{d}{dx}(\frac{\partial&space;F}{\partial&space;y'})&space;=&space;0" title="\frac{\partial F}{\partial y}-\frac{d}{dx}(\frac{\partial F}{\partial y'}) = 0" />
## The Method
The basic idea is to turn our Variational Calculus problem into a minimization task which we can solve numerically using simple gradient descent.
We can rewrite our solution function <img src="https://latex.codecogs.com/gif.latex?y" title="y(x)" /> as:\
3) <img src="https://latex.codecogs.com/gif.latex?y(x)&space;=&space;\sum_{n=0}^{\infty}k_nx^n" title="y(x) = \sum_{n=0}^{\infty}k_nx^n" />\
However, in terms of approximating the solution <img src="https://latex.codecogs.com/gif.latex?\hat{y}" title="\hat{y}(x)" />, we will obviously pick a constant <img src="https://latex.codecogs.com/gif.latex?N" title="N" /> such that:\
4) <img src="https://latex.codecogs.com/gif.latex?\hat{y}(x)&space;=&space;\sum_{n=0}^{N}k_nx^n" title="\hat{y}(x) = \sum_{n=0}^{N}k_nx^n" /> and <img src="https://latex.codecogs.com/gif.latex?\hat{y}(x)\approx&space;y(x)" title="\hat{y}(x)\approx y(x)" />\
The <img src="https://latex.codecogs.com/gif.latex?k_0...k_N" title="k_0...k_N" /> coefficients are, of course, unknown.\
Now, let us investigate a function\
5) <img src="https://latex.codecogs.com/gif.latex?\epsilon(x)&space;=&space;\left&space;|\frac{\partial&space;F}{\partial&space;\hat{y}}-\frac{d}{dx}(\frac{\partial&space;F}{\partial&space;\hat{y}'})&space;\right&space;|" title="\epsilon(x) = \left |\frac{\partial&space;F}{\partial&space;\hat{y}}-\frac{d}{dx}(\frac{\partial F}{\partial \hat{y}'}) \right |" />\
which is based upon the equation (2) and measures how different is our approximation <img src="https://latex.codecogs.com/gif.latex?\hat{y}" title="\hat{y}(x)" /> from <img src="https://latex.codecogs.com/gif.latex?y" title="y(x)" /> at certain <img src="https://latex.codecogs.com/gif.latex?x" title="x" />. For convenience purposes, we will consider it a function of multiple variables, as technically for us, not only it is dependent on <img src="https://latex.codecogs.com/gif.latex?x" title="x" /> but also on <img src="https://latex.codecogs.com/gif.latex?k_0...k_N" title="k_0...k_N" />. Thus, generally:\
6) <img src="https://latex.codecogs.com/gif.latex?\epsilon(x,&space;\overrightarrow{k})&space;=&space;\left&space;|\frac{\partial&space;F}{\partial&space;\hat{y}_{\overrightarrow{k}}}-\frac{d}{dx}(\frac{\partial&space;F}{\partial&space;\hat{y}_{\overrightarrow{k}}'})&space;\right&space;|" title="\epsilon(x, \overrightarrow{k}) = \left |\frac{\partial&space;F}{\partial&space;\hat{y}_{\overrightarrow{k}}}-\frac{d}{dx}(\frac{\partial F}{\partial \hat{y}_{\overrightarrow{k}}'}) \right |" />\
where <img src="https://latex.codecogs.com/gif.latex?\vec{k}" title="\vec{k}" /> is a vector of all <img src="https://latex.codecogs.com/gif.latex?k_0...k_N" title="k_0...k_N" /> (again, for convenience purposes).\
If we minimize <img src="https://latex.codecogs.com/gif.latex?\epsilon&space;(x,&space;\vec{k})" title="\epsilon(x, \overrightarrow{k})" /> with respect to <img src="https://latex.codecogs.com/gif.latex?\vec{k}" title="\vec{k}" /> (that is, we will try to find <img src="https://latex.codecogs.com/gif.latex?\vec{k}_\text{min}" title="\vec{k}_min" /> for which <img src="https://latex.codecogs.com/gif.latex?\epsilon" title="\epsilon(x, \vec{k})" /> is minimal over all <img src="https://latex.codecogs.com/gif.latex?x&space;\in&space;[a;&space;b]" title="x \in [a; b]">) we can, unfortunately, get a function that will not solve our initial problem. Therefore, the actual function we minimize is:\
7) <img src="https://latex.codecogs.com/gif.latex?l(x,&space;\vec{k})&space;=&space;\epsilon(x,\vec{k})&plus;\left&space;|&space;\frac{x-x_a}{x_b-x_a}(\hat{y}(x_b)-y_b)&space;\right&space;|&plus;\left&space;|&space;\frac{x_b-x&plus;x_a}{x_b-x_a}(\hat{y}(x_a)-y_a)&space;\right&space;|" title="l(x, \vec{k}) = \epsilon(x,\vec{k})+\left | \frac{x-x_a}{x_b-x_a}(\hat{y}(x_b)-y_b) \right |+\left | \frac{x_b-x+x_a}{x_b-x_a}(\hat{y}(x_a)-y_a) \right |" />\
where we add two terms to the <img src="https://latex.codecogs.com/gif.latex?\epsilon" title="\epsilon(x, \overrightarrow{k})" /> so that our function is pinalized for not passing through points <img src="https://latex.codecogs.com/gif.latex?a" title="a" /> and <img src="https://latex.codecogs.com/gif.latex?b" title="b" />.
So our initial problem is reduced to\
8) <img src="https://latex.codecogs.com/gif.latex?\text{argmin}_{\vec{k}}[l(x,&space;\vec{k})]" title="\text{argmin}_{\vec{k}}[l(x, \vec{k})]" />
the solution of which we can approximate with gradient descent.
## Results
To test the method, let us run it on the simplest possible <img src="https://latex.codecogs.com/gif.latex?F" title="F" />:\
<img src="https://latex.codecogs.com/gif.latex?F&space;=&space;\sqrt{1&plus;y'^2}" title="F = \sqrt{1+y'^2}" />\
which should give us a simple linear solution (<img src="https://latex.codecogs.com/gif.latex?y" title="y(x)" /> is a line from <img src="https://latex.codecogs.com/gif.latex?a" title="a" /> to <img src="https://latex.codecogs.com/gif.latex?b" title="b" />). If we plot the true solution against the one found by this method, we get\
![loss+solution](plotall.png)
![loss+solution](plotall2.png)
## Implementation
I have implemented the method in the Julia Programming Language. The code consists of three parts:
- ELSolution.jl - defines the ELSolution callable wrapper type.
- VarCalcSolver.jl - defines the solveEulerLagrangeGD function.
- Test.jl - uses previous files and plots the solution.
For more details, visit the files directly. All the definitions have been provided with a documentation.
## Helpful links
- [Introduction to Variational Calculus (Video)](https://www.youtube.com/watch?v=VCHFCXgYdvY)
- [The Deep Ritz method: A deep learning-based
numerical algorithm for solving variational problems](https://arxiv.org/pdf/1710.00211.pdf)
- [Basic numerical methods for VarCalc](https://encyclopediaofmath.org/wiki/Variational_calculus,_numerical_methods_of)
