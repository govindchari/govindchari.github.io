---
layout: distill
title:  Custom Convex Solvers
description: Why you should not use off the shelf convex solvers if you care about performance
tags: math
giscus_comments: false
date: 2023-05-09

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

---

## Introduction

Off the shelf solvers such as ECOS, SCS, Mosek, or Gurobi are good for prototyping and modest performance, but if you care about solving one problem class extremely quickly then these solvers will not give you maximum performance. In this post we will investigate a particular problem class (robust portfolio optimization) and solver (PIPG) to show how we can beat all off the shelf SOCP solvers by tuning our solver to our problem class. 
## Robust Portfolio Optimization
Formulate problem

Talk about classical case then extend to chance constraints

page 216 
https://web.stanford.edu/~boyd/papers/pdf/socp.pdf


## PIPG
PIPG natively handles quadratic objectives. Most SOCP solvers need to put the quadratic objective in the constraints which results in a larger KKT system to solve.

We will use PIPG which I have an upcoming blog post about (cite xPIPG)
You take a step in your gradient direction augmented with the constraint gradient direction, then you do projection and dual update
PIPG uses projections to exactly satisfy set D constraints at every iteration

Parse problem into PIPG canonical form

Three constraints: SOC on all of x, non-negative orthant constraints on all elements of x, and equality constraint on x
Try putting SOC into D and no shorting into K and vice versa.

## Results
***