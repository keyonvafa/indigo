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

For my undergraduate thesis, advised by <a href="http://people.seas.harvard.edu/~srush/">Alexander Rush</a>, I explore deep Gaussian processes (deep GPs), a class of models for regression that combines Gaussian processes (GPs) with deep architectures. Exact inference on deep GPs is intractable, and while researchers have <a href="http://www.jmlr.org/proceedings/papers/v31/damianou13a.pdf">proposed</a> <a href="https://arxiv.org/pdf/1412.1370v1.pdf">variational</a> <a href="https://arxiv.org/pdf/1511.06455.pdf">approximation</a> <a href="http://jmlr.org/proceedings/papers/v48/bui16.pdf">methods<a/>, these models are difficult to implement and do not extend easily to arbitrary kernels. We introduce the Deep Gaussian Process Sampling algorithm (DGPS), which relies on Monte Carlo sampling to circumvent the intractability hurdle and uses pseudo data to ease the computational burden. We build the intuition for this algorithm by defining and discussing GPs and deep GPs, going over their strengths and limitations as models. We then apply the DGPS algorithm to various data sets, and show that deeper architectures are better suited than single-layer GPs to learn complicated functions, especially those involving non-stationary data, although training becomes more difficult due to limitations of local maxima. Throughout, our goal is not only to introduce a novel inference technique, but also to make deep Gaussian processes more accessible to the machine learning community at large. This work would have been impossible without the generous help and support of <a href="http://finale.seas.harvard.edu/">Finale Doshi-Velez</a>, <a href="https://www.cs.toronto.edu/~duvenaud/">David Duvenaud</a>, and <a href = "https://jmhl.org/">José Miguel Hernández-Lobato<a/>.

My thesis is <a href="{{site.base_url}}/files/thesis.pdf">available here</a>. This led to a <a href='http://www.approximateinference.org/accepted/Vafa2016.pdf'>workshop paper</a> at the <a href='http://www.approximateinference.org/'>Advances in Approximate Bayesian Inference workshop</a> at NIPS. A blog post and Github tutorial are upcoming.

---

![Deep GP Posterior]({{ site.url }}/assets/images/step_1_2_layers_preds.png)
<figcaption class="caption">Posterior draws from a standard GP and a 2-layer deep GP trained on noisy step function data. The 2-layer deep GP is better equipped to model the non-stationarity of the data and capture our uncertainty outside of the input region.</figcaption>

---

## Slides

The slides below summarize the Deep Gaussian Process Sampling algorithm and highlight our results.

<iframe src="//www.slideshare.net/slideshow/embed_code/key/rt9RbzCsJZkkAl" width="560" height="310" frameborder="0" marginwidth="0" marginheight="0" scrolling="no" style="border:1px solid #CCC; border-width:1px; margin-bottom:5px; max-width: 100%;" allowfullscreen> </iframe>

