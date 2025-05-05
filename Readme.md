# Project

# DGSolver: Diffusion Generalist Solver with Universal Posterior Sampling for Image Restoration

<em>Hebaixu Wang, Jing Zhang, Haonan Guo, Di Wang, Jiayi Ma and Bo Du</em>.

[Paper](https://arxiv.org/abs/2504.21487) |  [Github Code](https://github.com/MiliLab/DGSolver)

## Abstract

Diffusion models have achieved remarkable progress in universal image restoration. While existing methods speed up inference by reducing sampling steps, substantial step intervals often introduce cumulative errors. Moreover, they struggle to balance the commonality of degradation representations and restoration quality. To address these challenges, we introduce \textbf{DGSolver}, a diffusion generalist solver with universal posterior sampling. We first derive the exact ordinary differential equations for generalist diffusion models and tailor high-order solvers with a queue-based accelerated sampling strategy to improve both accuracy and efficiency. We then integrate universal posterior sampling to better approximate manifold-constrained gradients, yielding a more accurate noise estimation and correcting errors in inverse inference. Extensive experiments show that DGSolver outperforms state-of-the-art methods in restoration accuracy, stability, and scalability, both qualitatively and quantitatively.

## Overview

<img src="./assets/method.png" width="100%">

## Visualization

<img src="./assets/visualization.png" width="100%">

## Datasets Information

| Task                     | Dataset                        | Synthetic/Real      | Download Links |
|--------------------------|--------------------------------|---------------------|----------------|
| **Deraining**            | DID                            | Synthetic           | URL()              |
|                          | DeRaindrop                     | Real                | URL()          |
|                          | Rain13K                        | Synthetic           | URL()        |
|                          | Rain_100H                      | Synthetic           | URL()           |
|                          | Rain_100L                      | Synthetic           | URL()            | 
|                          | GT-Rain                        | Real                | URL()         | 
|                          | RealRain-1k-H                  | Real                | URL()           | 
|                          | RealRain-1k-L                  | Real                | URL()            |
| **Low-light Enhancement**| LOL                            | Real                | URL()            |
|                          | MEF                            | Real                | URL()             |
|                          | VE-LOL-L                       | Synthetic/Real      | URL()        | 
|                          | NPE                            | Real                | URL()              | 
| **Desnowing**            | CSD                            | Synthetic           | URL()          | 
|                          | Snow100K-Real                  | Real                | URL()          |
| **Dehazing**             | SOTS                           | Synthetic           | URL()          | 
|                          | ITS_v2                         | Synthetic           | URL()         | 
|                          | D-HAZY                         | Synthetic           | URL()          |
|                          | NH-HAZE                        | Real                | URL()          |
|                          | Dense-Haze                     | Real               | URL()          |
|                          | NHRW                           | Real                | URL()          | 
| **Deblur**               | GoPro                          | Synthetic           | URL()          | 
|                          | RealBlur                       | Real                | URL()          | 

## Model Checkpoint

[Google Cloud]()
[Baidu Cloud]()

## Mixed-Datasets Links

[Google Cloud]()
[BaiduYun](https://pan.baidu.com/s/1Gkrf_a0cZ7vSXyFN4kPKaA?pwd=3di3)

### Contributor

Baixuzx7 @ wanghebaixu@gmail.com

### Copyright statement

The project is signed under the MIT license, see the [LICENSE.txt](https://github.com/MiliLab/DGSolver/LICENSE.md)
