---
layout: distill
title: Resource list
description: Various useful resources for learning optimization and computing
tags: optimization
giscus_comments: false
date: 2025-03-09

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
  - name: Optimization
  - name: Computing
  - name: Math
---

This post is a collection of good resources I have used during my studies to learn topics in convex optimization and computer systems and my review of each resource. This is intended to be a living document, and I will update this list every once in a while.

Disclaimer: I am not an expert in optimization or computing, but I am writing this post because I feel like I have spent enough time learning both to be able to recommend good resources.

## Optimization
This section mostly focuses on convex optimization.

### Convex Optimization by Boyd and Vandenberghe

This textbook which can freely be found [here](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf) is the classic introduction resource to convex optimization. This textbook has corresponding lectures by Stephen Boyd which can be found [here](https://www.youtube.com/playlist?list=PL3940DD956CDF0622). Additionally, Professor Boyd teaches this course at Stanford University and the course webpage which has all the lecture notes can be found [here](https://web.stanford.edu/class/ee364a/).

The textbook has three parts: theory, application, and algorithms. The theory part of the textbook mostly focuses on what is convexity, why is it important, and what are operations that preserve convexity. Then it covers the classical convex optimization problems such as linear programs, quadratic programs, second-order cone programs, and semidefinite programs. Finally, it covers duality and the Karush-Kuhn-Tucker (KKT) conditions.

I think this book is a good introduction to convex optimization, but it is not my favorite textbook to learn optimization algorithms from. Although part 3 of the textbook does cover optimization algorithms, it mostly focuses on primal interior point methods, which are not used much by popular convex solvers. Additionally, I do not care much for the presentation of duality, it feels quite prescriptive in my opinion and a bit unmotivated. Boyd presents duality as a "structured way to create lower bounds for optimization problems", which although that is true, I believe the real importance of duality in convex optimization is to aid in algorithm design.

Nevertheless, I recommend any beginner to read the first part of this book before reading any other textbook on this list.

### Numerical Optimization by Nocedal and Wright

This is one of the best textbooks for learning about optimization algorithms. The beginning of the book discusses optimization algorithms for unconstrained nonconvex problems. To be honest I have not read much of that part. Chapters 12-17 are a goldmine for learning about algorithms for constrained convex optimization. I also love the presentation of duality in Chapter 12 far more than Boyd's treatment. In this textbook duality is presented as a tool in algorithm design.

I also enjoy the historical perspective this textbook brings on optimization algorithms. First, the simplex method for linear programming is discussed, then its drawbacks (worst case exponential runtime) are presented, then interior point methods are presented. This textbook in my opinion, has the best explanation for how primal and primal-dual interior point methods work. My only complaint is that the primal-dual interior point method is only described for linear programming and quadratic programming and does not include second-order cone programming and semidefinite cone programming. The book also discusses active set methods for quadratic programming and presents the material is a very clear way.

My only complaints about this book is that primal-dual interior point methods are not described for second-order cone programming and semidefinite programming. Also operator splitting methods such as forward-backwards splitting, ADMM, and PDHG are not discussed.

### Large-Scale Convex Optimization by Ryu and Yin
This textbook can be freely found [here](https://large-scale-book.mathopt.com/LSCOMO.pdf) with accompanying videos [here](https://www.youtube.com/@large-scaleconvexoptimizat2973/videos).

This is my favorite textbook of all time by a long shot, and is the best resource I have found to understand operator-splitting methods. I have always heard about projected gradient descent, the proximal point method, ISTA, augmented Lagrangian methods, Douglas-Rachford Splitting, ADMM, PDHG etc, and always viewed them as separate concepts in my mind. However, this textbook unifies all of these algorithms as fixed-point iterations with some averaged operator. Even reading the first three chapters will be an extremely eye-opening experience to those interested in operator-splitting methods.

### Convex Analysis and Nonsmooth Optimization by Drusvyatskiy
[This set of course notes](https://sites.math.washington.edu/~ddrusv/crs/Math_516_2020/bookwithindex.pdf) discusses the fundemental concepts in convex analysis in a rigorous way. I used these notes in my Convex Analysis class and enjoyed them.

### Convex Analysis and Monotone Operator Theory by Baushke and Combettes
Disclaimer: I have not read this textbook, but it has been highly praised for rigorously discussed Monotone operator theory which is fundamental for rigorously analyzing operator splitting methods. 

## Computing

### The Cherno
The YouTube playlist is [here](https://www.youtube.com/playlist?list=PLlrATfBNZ98dudnM48yfGUldqGD0S4FFb). This is the best video series to learn C++. I love that he discusses not only C++ syntax, but what is happening under the hood in terms how data and the program is stored in memory.

### Building an 8-bit computer
[This video series](https://www.youtube.com/playlist?list=PLowKtXNTBypGqImE405J2565dvjafglHU) by Ben Eater is a fantastic introduction to computer architecture. He starts from simple and/or/not gates and builds an 8-bit computer on a breadboard while explaining everything you need to know along the way.

Although this series does not discuss important topics for performance such as cache and branch prediction, this is a fantastic video series for a beginner who wants to understand how a CPU works.

### Computer Systems Programming
This [video series](https://scs.hosted.panopto.com/Panopto/Pages/Sessions/List.aspx#folderID=%22b96d90ae-9871-4fae-91e2-b1627b43e25e%22) is from the 15-213: Introduction to Computer Systems course at Carnegie Mellon University and follows [this textbook](https://www.cs.cmu.edu/~213/textbook.html).

This is my favorite resource to learn about lower level concepts in computing such as assembly, memory cache, and writing more performant code.

### Performance Aware Programming 
[This course](https://www.computerenhance.com/p/table-of-contents) by Casey Muratori is fantastic for understanding how to write code in a performance oriented way. I am currently working through it.

## Math
### Elementary Analysis by Kenneth Ross
I learned basic analysis from this textbook. The book is well written, I like the examples and I like that the textbook is not overly rigorous like Rudin. I think it is a good intro to analysis.

### Introductory Functional Analysis by Erwin Kreyszig
I have skimmed the first few chapters. The author does a good job of motivating the need for the abstraction of metric and Banach spaces. Most of the material in the first two chapters was pretty easy to grasp with knowledge of proofs, linear algebra, and analysis. I started reading this textbook so I could understand Convex Analysis and Monotone Operator Theory by Baushke and Combettes, as they work in Hilbert spaces. I also think this material is pretty interesting in its own right.

---
