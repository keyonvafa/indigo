---
title: "Discrete Flows: Invertible Generative Models of Discrete Data"
layout: post
date: 2019-10-09 15:28
headerImage: false
projects: true
hidden: true # don't count this post in blog pagination
author: keyonvafa
externalLink: false
---

---


While normalizing flows have led to significant advances in modeling high-dimensional continuous distributions, their applicability to discrete distributions remains unknown. In this work, we show that flows can in fact be extended to discrete events—and under a simple change-of-variables formula not requiring log-determinant-Jacobian computations. Discrete flows have numerous applications. We consider two flow architectures: discrete autoregressive flows that enable bidirectionality, allowing, for example, tokens in text to depend on both left-to-right and right-to-left contexts in an exact language model; and discrete bipartite flows that enable efficient non-autoregressive generation as in RealNVP. Empirically, we find that discrete autoregressive flows outperform autoregressive baselines on synthetic discrete distributions, an addition task, and Potts models; and bipartite flows can obtain competitive performance with autoregressive baselines on character-level language modeling for Penn Tree Bank and text8.

D. Tran, K. Vafa, K. K. Agrawal, L. Dinh, and B. Poole. **Discrete Flows: Invertible Generative Models of Discrete Data**. _Advances in Neural Information Processing Systems (to appear)_. Vancouver (Canada), December 2019.

_To appear at NeurIPS 2019. <a href="https://arxiv.org/abs/1905.10347">arXiv link here.</a>_

---

![Gaussian mixture with flows]({{ site.url }}/assets/images/projects/gaussian_mixture_flows.png)
<figcaption class="caption">Learning a discretized mixture of Gaussians with maximum likelihood. Discrete flows help capture the multi-dimensional modes, which a factorized distribution cannot.</figcaption>



