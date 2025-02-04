---
layout: post
title: Released QOCO and QOCOGEN
date: 2025-01-10
inline: false
---

QOCO is a C-based solver for second-order cone programs with quadratic objectives based on a primal-dual interior point method. This solver is fast, robust and easy to use.

QOCOGEN is a custom code generator which takes in an SOCP problem family and generates a customized C solver (called qoco_custom) for this problem family which implements the same algorithm as QOCO. This customized solver is library-free, only uses static memory allocation, and can be a few times faster than QOCO. QOCO can be called from C/C++, Python, and Matlab, and can be called from the parser CVXPY.

***

The website describing these solvers and how to use them can be found [here](https://qoco-org.github.io/qoco/index.html). The paper is forthcoming.