# Project

# DGSolver: Diffusion Generalist Solver with Universal Posterior Sampling for Image Restoration

<em>Hebaixu Wang, Jing Zhang, Haonan Guo, Di Wang, Jiayi Ma and Bo Du</em>.

[Paper]() |  [Github Code](https://github.com/MiliLab/DGSolver)

## Abstract

Diffusion models have achieved remarkable progress in universal image restoration. While existing methods speed up inference by reducing sampling steps, substantial step intervals often introduce cumulative errors. Moreover, they struggle to balance the commonality of degradation representations and restoration quality. To address these challenges, we introduce \textbf{DGSolver}, a diffusion generalist solver with universal posterior sampling. We first derive the exact ordinary differential equations for generalist diffusion models and tailor high-order solvers with a queue-based accelerated sampling strategy to improve both accuracy and efficiency. We then integrate universal posterior sampling to better approximate manifold-constrained gradients, yielding a more accurate noise estimation and correcting errors in inverse inference. Extensive experiments show that DGSolver outperforms state-of-the-art methods in restoration accuracy, stability, and scalability, both qualitatively and quantitatively.

## Overview

<img src="./assets/method.png" width="100%">

## Visualization

<img src="./assets/visualization.png" width="100%">

### Contributor

Baixuzx7 @ wanghebaixu@gmail.com

### Copyright statement

The project is signed under the MIT license, see the [LICENSE.txt](https://github.com/MiliLab/DGSolver/LICENSE.md)
