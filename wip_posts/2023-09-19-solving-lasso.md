---
layout: distill
title:  Solving Lasso Regression
description: Comparing optimization algorithms for Lasso
tags: optimization
giscus_comments: false
date: 2023-09-19

authors:
  - name: Govind Chari
    url: "https://govindchari.com/"
    affiliations:
      name: University of Washington Autonomous Controls Lab

bibliography: mybib.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Introduction
  - name: Reformulating as Quadratic Program
  - name: ISTA
  - name: FISTA
  - name: ADMM
  - name: Results
---

## Introduction
Lasso regression is and important regularization technique for linear regression that can also perform variable selection. What this means is that solutions to the lasso problem tend to be sparse (contain zeros) which allows us to rule out certain independent variables in our model. A great resource to familiarize yourself with lasso is [this video](https://www.youtube.com/watch?v=GaXfqoLR_yI&ab_channel=SteveBrunton).

Lasso is of incredible importance in statistics, signal processing, compressed sensing, and image processing. In this post we will look at a variety of optimization technique for solving the lasso problem. 

The lasso optimization problem is an unconstrained optimization problem which can be written as follows:

$$
\begin{split}
    \underset{x}{\text{minimize}} 
    \quad & \frac{1}{2}\|Ax-b\|_2^2 + \lambda \|x\|_1 \\
\end{split}
$$

where $A \in \mathbb{R}^{m \times n}$

A first thought to solve this problem might be gradient descent, however the objective function is non-smooth (due to the l1 penalty), so we cannot use gradient descent.
A second thought is to use subgradient descent, which is a generalization of gradient descent to non-smooth functions. This would work, but subgradient descent has an extremely slow worst-case convergence rate of $\mathcal{O}(1 / \sqrt{t})$ (meaning you need four times the iterations to double the accuracy) so we will look at better algorithms.

## Reformulating as Quadratic Program
One of the easiest things to do would be reformulating this problem as a Quadratic Program (QP) as follows:

$$
\begin{split}
    \underset{x}{\text{minimize}} 
    \quad & \frac{1}{2}\|Ax-b\|_2^2 + \lambda \sum_{i=1}^n t_i \\
    \text{subject to} 
    \quad & -t_i \leq x_i \leq t_i \quad \forall i \in [1,n]
\end{split}
$$

We can then feed this into a QP solver such as OSQP and then get an answer. This would would but it feel wasteful to turn an unconstrained problem into a constrained one and then using a generic QP solver. There should be better algorithms that exploit the structure of our problem where we have a smooth plus a non-smooth term.

## ISTA
ISTA or iterative shrinking threshold algorithm is an application of the proximal gradient method to the Lasso problem.

The proximal gradient method solves problems of the following form, where $f$ is differentiable

$$
\begin{split}
    \underset{x}{\text{minimize}} 
    \quad & f(x) + g(x) \\
\end{split}
$$

The algorithm looks as follows:

$$
x^{k+1} = \boldsymbol{\text{prox}}_{\eta g}(x^k - \eta \nabla f(x^k))
$$

where $\eta$ is the gradient descent stepsize for $f$ which will be the inverse of the Lipschitz constant of $f$.

The proximal operator $\boldsymbol{\text{prox}}_{\eta g}$ is a generalization of the projection operation and is defined as follows

$$
\boldsymbol{\text{prox}}_{\eta f}(v) = \underset{x}{\text{argmin}} \left(f(x) + \frac{1}{2\eta}\|x-v\|_2^2\right) 
$$

You can think of the proximal operator as returning a point which balances minimizing the function and staying close to the current point. The proximal operator for many function are well known in closed form. For more information on proximal operators and algorithms using proximal operators, refer to <d-cite key="Parikh2014Proximal"></d-cite>.

The idea behind the proximal gradient method is to perform a gradient descent step assuming we are just going to be minimizing the smooth function $f$ and then do an evaluation of the proximal operator for $g$ which can be interpreted as a gradient descent step on the smoothed version of $g$. More technically we do a gradient descent step on the Moreau envelope of $g$.

Alternating between these two steps, we eventually minimize our original objective function.

For Lasso we will take, 

$$
\begin{split}
    f(x) &= \frac{1}{2}\|Ax-b\|_2^2 \\
    g(x) &= \|x\|_1 \\
\end{split}
$$

It can be shown that the proximal operator for the l1 norm is the soft threshold operator

$$
\boldsymbol{\text{prox}}_{\eta \|\cdot\|_1}(v) = \mathcal{S}_\eta(v) = \text{sign}(v)\max(|v|-\eta,0)
$$

We can now write out the ISTA iterates as follows

$$
x^{k+1} = \mathcal{S}_{\lambda/L}\left(x^k - \frac{1}{L}A^\top(Ax^k-b)\right)
$$

Where $L$ is the maximum eigenvalue of $A^\topA$. This can quickly be computed via [power iteration](https://en.wikipedia.org/wiki/Power_iteration).

It can be shown that this algorithm has a worst-case convergence rate of $\mathcal{O}(1 / t)$ meaning that if we double the number of iterations, we double the accuracy of the solution. This is already better than subgradient method, but is not the best we can do.

## FISTA
ISTA was used for a while, but it can be painfully slow to converge. In 2009 Beck and Teboulle introduced FISTA (Fast Iterative Shrinking Threshold Algorithm) where they used momentum to accelerate ISTA and were able to achieve the optimal worst-case convergence rate of $\mathcal{O}(1 / t^2)$ <d-cite key="Beck2009Fast"></d-cite>.

## ADMM

## Results
What we have 

***