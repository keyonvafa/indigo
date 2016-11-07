---
title: "Training and Inference for Deep Gaussian Processes"
layout: post
date: 2016-11-07 14:38
headerImage: false
projects: true
hidden: true # don't count this post in blog pagination
author: keyonvafa
externalLink: false
---

---

![Deep GP Posterior]({{ site.url }}/assets/images/step_1_2_layers_preds.png)

---

For my undergraduate thesis, advised by <a href="http://people.seas.harvard.edu/~srush/">Alexander Rush</a>, I explore deep Gaussian processes (deep GPs), a class of models for regression that combines Gaussian processes (GPs) with deep architectures. Exact inference on deep GPs is intractable, and while researchers have proposed variational approximation methods, these models are difficult to implement and do not extend easily to arbitrary kernels. We introduce the Deep Gaussian Process Sampling algorithm (DGPS), which relies on Monte Carlo sampling to circumvent the intractability hurdle and uses pseudo data to ease the computational burden. We build the intuition for this algorithm by defining and discussing GPs and deep GPs, going over their strengths and limitations as models. We then apply the DGPS algorithm to various data sets, and show that deeper architectures are better suited than single-layer GPs to learn complicated functions, especially those involving non-stationary data, although training becomes more difficult due to limitations of local maxima. Throughout, our goal is not only to introduce a novel inference technique, but also to make deep Gaussian processes more accessible to the machine learning community at large. This work would have been impossible without the generous help and support of <a href="http://finale.seas.harvard.edu/">Finale Doshi-Velez</a>, <a href="https://www.cs.toronto.edu/~duvenaud/">David Duvenaud</a>, and <a href = "https://jmhl.org/">José Miguel Hernández-Lobato<a/>.

My thesis is <a href="{{site.base_url}}/files/thesis.pdf">available here</a>. A blog post and Github tutorial are upcoming.

---

## Slides

The slides below summarize the Deep Gaussian Process Sampling algorithm and highlight our results.

<iframe src="//www.slideshare.net/slideshow/embed_code/key/rt9RbzCsJZkkAl" width="560" height="310" frameborder="0" marginwidth="0" marginheight="0" scrolling="no" style="border:1px solid #CCC; border-width:1px; margin-bottom:5px; max-width: 100%;" allowfullscreen> </iframe>

---
