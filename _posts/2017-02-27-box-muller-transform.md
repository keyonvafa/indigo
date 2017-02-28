---
title: The Box-Muller Transform
layout: post
date: 2017-02-27 21:00
headerImage: false
blog: true
star: false
author: keyonvafa
description: Sampling normal random variables using uniform draws.  
---

Every statistician has a favorite way of generating samples from a distribution (not sure if I need a citation for this one). From <a href='https://en.wikipedia.org/wiki/Rejection_sampling'>rejection sampling</a> to <a href='https://arxiv.org/pdf/1206.1901.pdf'>Hamiltonian Monte Carlo</a>, there are countless methods to choose from (my personal favorite is ``rnorm``).

One of the most interesting and counterintuitive sampling techniques is the Box-Muller transform. I'm not sure how widely it's used today, but given two samples from a uniform distribution, it can generate two _independent_ samples from a standard normal distribution.

<!--Given a uniform sample $$U \sim \text{Unif}(0,1)$$, we can generally sample from a distribution with cdf $$F$$ by taking $$F^{-1}(U)$$. Since we cannot write the normal cdf in closed form, we must rule out the inverse cdf method.-->

The idea behind the Box-Muller transform is to imagine two independent samples $$X, Y \sim \mathcal{N}(0,1)$$ plotted in the Cartesian plane, and then represent these points as polar coordinates. Recall, to transform to polar, we need the distance $$R$$ between $$(X,Y)$$ and the origin along with $$\theta$$, the angle this line segment makes with the x-axis.

We start with the distance between the origin, $$R = \sqrt{X^2 + Y^2}$$. For simplicity, we work with $$R^2 = X^2 + Y^2$$. The sum of two independent squared standard normals follows a <a href='https://en.wikipedia.org/wiki/Chi-squared_distribution'>chi-squared distribution</a> with 2 degrees of freedom. It is also a <a href='https://en.wikipedia.org/wiki/Chi-squared_distribution#Gamma.2C_exponential.2C_and_related_distributions'>known fact</a> that a chi-squared distribution with 2 degrees of freedom is equivalent to a $$\text{Gamma}(1,\frac{1}{2})$$ random variable, which is itself <a href='http://stats.stackexchange.com/questions/27908/sum-of-exponential-random-variables-follows-gamma-confused-by-the-parameters'>equivalent</a> to a $$\text{Expo}(\frac{1}{2})$$ variable. Finally, we can express an exponential random variable as the <a href='http://math.stackexchange.com/questions/199614/distribution-of-log-x-if-x-is-uniform'>log of a uniform</a>. More succinctly,

$$
R^2 \sim \chi^2_{df=2} \sim \text{Gamma}\left(1,\frac{1}{2}\right) \sim \text{Expo}\left(\frac{1}{2}\right) \sim -2\log U_1
$$

where $$U_1 \sim \text{Unif}(0,1).$$

What about the angle, $$\theta$$? If we write the joint density of $$X$$ and $$Y$$, we can see

$$
f_{X,Y}(x,y) = \frac{1}{2\pi} e^{-\frac{X^2}{2}}e^{-\frac{Y^2}{2}} = \frac{1}{2\pi}e^{-\frac{(X^2+Y^2)}{2}} = \frac{1}{2\pi}e^{-\frac{R^2}{2}}.
$$

Thus, once we have $$R^2$$, the squared distance between $$(X,Y)$$ and the origin, the joint distribution of $$X$$ and $$Y$$ is uniform. That is, as long as $$(X,Y)$$ is a pair satisfying $$X^2 + Y^2 = R^2$$, it can be any point on the circle with radius $$R$$. As a result, we can simply take $$\theta = 2\pi U_2$$, where $$U_2 \sim \text{Unif}(0,1).$$

Putting all these results together, if we take $$R = \sqrt{-2\log U_1}$$ and $$\theta = 2\pi U_1$$ for $$U_1, U_2 \sim \text{Unif}(0,1)$$, we have the polar coordinates for two independent standard normal draws. Thus, converting back to Cartesian, we have 

$$
X = R\cos\theta = \sqrt{-2\log U_1}\cos(2\pi U_2)\\
Y = R\sin\theta = \sqrt{-2\log U_1}\sin(2\pi U_2).
$$

This is straightforward to implement in R:

```R
nsims = 10000
samples = rep(NA,nsims*2)
for (sim in 1:nsims) {
  us = runif(2)
  R = sqrt(-2*log(us[1]))
  theta = 2*pi*us[2]
  samples[2*sim] = R*cos(theta)
  samples[2*sim-1] = R*sin(theta) 
}
```
Using the above code, I compared the histogram of Box-Muller samples to those using `rnorm`, which were nearly identical:

![Box-Muller Samples]({{site.base_url}}/assets/images/box_muller_blog/box_muller_samples.png)

_Interesting, but this is nothing more than a cool sampling trick, right?_ Wrong. If we represent normal random variables in Box-Muller form, it can become easier to prove results about the normal distribution. 

For example, consider the problem of proving that for independent draws $$X,Y \sim \mathcal{N}(0,1)$$, $$X+Y$$ is independent of $$X-Y$$, and both distributed as $$\mathcal{N}(0,2)$$. A proof that doesn't require the use of pdfs involves representing $$X$$ and $$Y$$ in Box-Muller form (I first saw this solution in <a href='http://www.people.fas.harvard.edu/~blitz/Site/Home.html'>Joe Blitzstein's</a> class <a href='https://locator.tlt.harvard.edu/course/colgsas-111696'>Stat 210</a>, which I encourage any Harvard student who's reading this to take). Let $$R^2 \sim \chi^2_{df=2}$$ and $$U \sim \text{Unif}(0,1)$$, as in the representation above. Thus, $$X = R\cos(\theta) = R\cos(2\pi U)$$, and $$Y = R\sin(\theta) = R\cos(2\pi U)$$. This form gives us

$$
X + Y = R\cos(2\pi U) + R\sin(2\pi U) = \sqrt{2}R\sin(2\pi U + \pi/4)\\
X - Y = R\cos(2\pi U) - R\sin(2\pi U) = \sqrt{2}R\cos(2\pi U + \pi/4)
$$

Note that we use the trigonometric identities for $$\cos(\alpha + \beta)$$ and $$\sin(\alpha + \beta)$$ in the derivation. The final form should look familiar -- we've recovered the Box-Muller representation, albeit with some modifications. The $$\sqrt{2}$$ in front scales the standard normal so it now has a variance of 2. Additionally, note that we are using $$2\pi U + \pi/4$$ as $$\theta$$ instead of $$2\pi U$$. However, we do not have to worry about it as it still results in a uniform sample over the possible angles.

Thus, $$X+Y$$ and $$X-Y$$ are independent draws from the distribution $$\mathcal{N}(0,2)$$.

<!--between the x-axis and the line segment connecting the origin and $$(X,Y)$$. -->

<!--I first came across the method in a class taught by <a href='http://www.people.fas.harvard.edu/~blitz/Site/Home.html'>Joe Blitzstein</a>, and a conversation today with another PhD student inspired me to write up a short tutorial.-->
