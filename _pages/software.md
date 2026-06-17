---
layout: page
permalink: /software/
title: software
description: Software I have worked on.
nav: true
nav_order: 2
---

### Main Developer

- [QOCO](https://github.com/qoco-org/qoco): QOCO is an open-source solver for quadratic objective SOCPs written in C. It can be called from CVXPY.

- [QOCOGEN](https://github.com/qoco-org/qocogen): QOCOGEN is an open-source custom solver generator for quadratic objective SOCPs written in C. It generates a custom solver written in C (called QOCO Custom) which exploits the sparsity structure of the problem to achieve even faster solvetimes. QOCOGEN can be called from CVXPYgen.

### Contributor

I have also made open-source contributions to the following software:

- [CVXPY](https://www.cvxpy.org/): A domain specific language for optimization problems in Python that allows users to specify optimization problems in a way that is nearly identical to the math.

- [CVXPYgen](https://github.com/cvxgrp/cvxpygen): A code generation tool that sits on top of CVXPY and generates an embeddable solver for a users specified problem class.
