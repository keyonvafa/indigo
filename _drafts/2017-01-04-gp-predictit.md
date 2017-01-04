---
title: "Betting on Politics with Gaussian Processes"
layout: post
date: 2017-01-04 05:19
headerImage: false
blog: true
star: false
author: keyonvafa
description: Applying Guassian Proccesses to PredictIt
---

Happy new year everyone! 

In this post, I'll provide a quick tutorial on using Gaussian processes for regression and walk through one of my favorite applications: betting on political data on the website PredictIt. This post is intended for anyone who has taken an intro probability class and has basic machine learning experience, as my main goal is to provide intuition for Gaussian processes. Thus, it is not meant to be exhaustive at all. Be sure to check out Carl Rasmussen and Christopher Williams's excellent textbook _Gaussian Processes for Machine Learning_ (<a href="http://www.gaussianprocess.org/gpml/">available for free online</a>) for a more comprehensive reference. 

We begin by overviewing Gaussian processes (GPs). If you would like to skip the overview and go straight to the betting example, jump ahead to the [next section](#betting-on-predictit-with-gaussian-proccesses). 

## Gaussian Process Tutorial

What is a Gaussian process? Frequently, it is referred to as the infinite-dimensional extension of the multivariate normal distribution (CITE?). This may be confusing, because we typically don't observe random variables with infinitely many components. However, when we work with GPs, the intuition is that we observe some finite-dimensional subset of infinite-dimensional data, and this finite subset follows a multivariate normal distribution, as would every finite subset.

For example, suppose we measure the temperature every day of the year at noon, resulting in a 365-dimensional vector. In reality, temperature is a continuous process, and the choice to take a measurement every day at noon is arbitrary. What would happen if we took the temperature in the evening instead? What if we took measurements every hour or every week? If we model the data with a GP, we are assuming that each of these possible data collection schemes would yield data from a multivariate normal distribution. 

Thus, it makes sense to think of a GP as a function. Formally, a function $$\boldsymbol f$$ is a GP if any finite set of values $$f(\boldsymbol x_1), \dots, f(\boldsymbol x_n)$$ has a multivariate normal distribution, where the inputs $$\{\boldsymbol x_n\}_{n=1}^N$$ correspond to objects (typically vectors) from any arbitrarily sized domain. For example, in the temperature example, $$\{\boldsymbol x_n\}_{n=1}^{365}$$ correspond to the days of the year, and $$f(\boldsymbol x_n)$$ indicates the temperature measurement at day $$n$$. 

A GP is specified by a mean function $$m(\boldsymbol x)$$ and a covariance function $$k(\boldsymbol x, \boldsymbol x')$$, otherwise known as a _kernel_. That is, for any $$x, x'$$, $$m(\boldsymbol x) = E[f(\boldsymbol x)]$$ and $k(\boldsymbol x, \boldsymbol x') = \text{Cov}(f(\boldsymbol x),f \boldsymbol (x'))$. The shape and smoothness of our function is determined by the covariance function, as it controls the correlation between all pairs of output values. Thus, if $k(\boldsymbol x, \boldsymbol x')$ is large when $\boldsymbol x$ and $\boldsymbol x'$ are near one another, the function will be more smooth, while smaller kernel values imply a more jagged function.

Given a mean function and a kernel, we can sample from any GP. Say we want to evaluate the function at $$N$$ inputs, each of which has dimension $$D$$. We first create a matrix $$\boldsymbol X \in \mathbb{R}^{N \times D}$$, where each row corresponds to an input we would like to sample from. We then evaluate the mean function at all inputs, denoted by $$\boldsymbol{m}_{\boldsymbol X}$$ (a vector of length $$N$$), and the _kernel matrix_ corresponding to $$\boldsymbol X$$, denoted by $$\boldsymbol K_{\boldsymbol X,\boldsymbol X}$$, defined by

$$
\boldsymbol K_{\boldsymbol X\boldsymbol X} = \begin{pmatrix} k(\boldsymbol x_1, \boldsymbol x_1) & \cdots & k(\boldsymbol x_1, \boldsymbol x_N)\\ \vdots & \ddots & \vdots \\ k(\boldsymbol x_N, \boldsymbol x_1) & \cdots & k(\boldsymbol x_N, \boldsymbol x_N)\end{pmatrix}.
$$

More generally, for any two sets of input data, $$\boldsymbol X$$ and $$\boldsymbol X'$$, we define $$\boldsymbol K_{\boldsymbol X \boldsymbol X'}$$ to be the matrix where the $$(i,j)$$ element is $$k(\boldsymbol x_i, \boldsymbol x_j')$$.  Finally, we can sample a random vector $$\boldsymbol f$$ from a multivariate normal distribution: $$\boldsymbol f \sim \mathcal N(\boldsymbol m_{\boldsymbol X}, \boldsymbol K_{\boldsymbol X \boldsymbol X})$$. By construction, $$E(f(\boldsymbol x_n)) = m(\boldsymbol x_n)$$ for all $$n$$ and $$\text{Cov}(f(\boldsymbol x_n), f(\boldsymbol x_m)) = k(\boldsymbol x_n, \boldsymbol x_m)$$ for all pairs $$n,m$$. Because this vector has a multivariate normal distribution, all subsets also follow a multivariate distribution, fulfilling the definition of a GP.

Choosing an appropriate kernel may not be a straightforward task. The only requirement is that the kernel be a positive-definite function that maps two inputs, $$\boldsymbol x$$ and $$\boldsymbol x'$$, to a scalar, so that $$\boldsymbol K_{\boldsymbol X \boldsymbol X}$$ is a valid covariance matrix. Thus, it is typical to choose a kernel that can approximate a large variety of functions. I won't go over different types of kernels here, but the <a href="http://www.cs.toronto.edu/~duvenaud/cookbook/index.html">kernel cookbook</a> by David Duvenaud provides a great overview of popular kernels for GPs.

For example, consider single-dimensional inputs $$\{x_n\}$$ with a constant mean function at 0 and the following kernel:

$$
k(x,x') = h^2\left(1+\frac{(x-x')^2}{2\alpha l^2}\right)^{-\alpha},
$$

where $$k, \alpha,$$ and $$l$$ are all positive real numbers, referred to as hyper-parameters. This is known as the _rational quadratic_ covariance function (RQ). I won't go into much detail about this particular kernel, but note that it only depends on the inputs via their difference $$(x-x')$$, meaning the shape of the function is constant throughout the input space. Additionally, as $$x$$ and $$x'$$ are closer to one another, the covariance is larger, resulting in continuity. Below are samples drawn from a GP with a rational quadratic kernel and various kernel parameters, with $$h$$ fixed at 1:

![GP Samples]({{site.base_url}}/assets/images/gp_predictit_blog/gp_samples.pdf)

Note that after importing `numpy` and defining the RQ covariance, these plots are generated in Python by

```python
    plot_xs = np.reshape(np.linspace(-5, 5, 300), (300,1))
    sampled_funcs = np.random.multivariate_normal(np.ones(len(plot_xs)), rq_covariance(params,plot_xs,plot_xs), \
                        size=10)
    ax.plot(plot_xs, sampled_funcs.T)
```

Typically, we would like to estimate function values of a GP conditioned on some training data, rather than merely sampling functions. We are typically given a set of inputs $$\boldsymbol X \in \mathbb{R}^{N \times D}$$ and corresponding outputs $$\boldsymbol f \in \mathbb{R}^n$$, and we would like to estimate the outputs $$\boldsymbol f_*$$ for a set of new inputs $$\boldsymbol X_*$$. In the simplest, noise-free case, we can model $$\boldsymbol f$$ as a GP. What does this mean? We have observed data $$(\boldsymbol f)$$ and unobserved data $$(\boldsymbol f_*)$$ coming from a GP, so we know that concatenating $$\boldsymbol f$$ and $$\boldsymbol f_*$$ results in a multivariate normal with the following mean and covariance structure:

$$
\begin{pmatrix} \boldsymbol f \\ \boldsymbol f_* \end{pmatrix} \sim \mathcal N \left( \begin{pmatrix} \boldsymbol m_{\boldsymbol X} \\ \boldsymbol m_{\boldsymbol X_*} \end{pmatrix}, \begin{pmatrix} \boldsymbol K_{\boldsymbol X \boldsymbol X} &  \boldsymbol K_{\boldsymbol X \boldsymbol X_*} \\  \boldsymbol K_{\boldsymbol X_* \boldsymbol X} &  \boldsymbol K_{\boldsymbol X_* \boldsymbol X_*}\end{pmatrix}\right).
$$

Note that if this notation is unfamiliar, we're just concatenating vectors and matrices. For example, the mean parameter of this multivariate normal is $$\boldsymbol m_{\boldsymbol X}$$ concatenated with $$\boldsymbol m_{\boldsymbol X_*}$$.

Now, because $$\boldsymbol f$$ is observed, we can model $$\boldsymbol f_*$$ using the conditional distribution of a multivariate normal, given by:

$$
p(\boldsymbol f_*| \boldsymbol X_*, \boldsymbol X, \boldsymbol f) = \mathcal N(\boldsymbol m_{\boldsymbol X_*} +\boldsymbol K_{\boldsymbol X_* \boldsymbol X} \boldsymbol K_{\boldsymbol X \boldsymbol X}^{-1}(\boldsymbol f - \boldsymbol m_{\boldsymbol X_*}),  \boldsymbol K_{\boldsymbol X_* \boldsymbol X_*} -  \boldsymbol K_{\boldsymbol X_* \boldsymbol X} \boldsymbol K_{\boldsymbol X \boldsymbol X}^{-1} \boldsymbol K_{\boldsymbol X \boldsymbol X_*}).
$$

This is a known result of the multivariate normal distribution; if this is result is unfamiliar, this <a href="http://stats.stackexchange.com/questions/30588/deriving-the-conditional-distributions-of-a-multivariate-normal-distribution">Stack Exchange answer</a> gives a pretty neat derivation. Thus, we not only have an estimate of function values, but we also have complete knowledge of the predictive covariance in closed form, making it possible to assess uncertainty. This will come in handy for betting on political data. 

A subtle note is that typically we do not have access to the function values themselves, but rather noisy observations $$y_n = f(\boldsymbol x_n) + \epsilon_n$$ where $$\epsilon_n \sim \mathcal N(0, \sigma^2_\epsilon)$$ i.i.d. We can incorporate this noise into our model by adding $$\sigma_\epsilon^2$ to every diagonal term in $$\boldsymbol K_{\boldsymbol X \boldsymbol X}$$, which corresponds to an updated kernel.

<a href='http://www.gaussianprocess.org/gpml/chapters/RW2.pdf'>Rasmussen and Williams</a> show that the likelihood incorporates a tradeoff between fit and model complexity, so overfitting tends to be less significant a problem in GP regression. A downside, however, is that every iteration of optimization requires the inversion of an $N \times N$ matrix, which is $\mathcal O(N^3)$. 

## Betting on PredictIt with Gaussian Processes

Now that we're all experts on GPs, let's use them to make money. One of my favorite websites is <a href='https://www.predictit.org/'> PredictIt</a>, which provides prediction markets for betting on politics. Many of these markets are not explicitly quantitative, such as <a href="https://www.predictit.org/Contract/4500/Will-there-be-a-Putin-Trump-meeting-in-US-in-Trump's-first-100-days#dat">Will there be a Putin-Trump meeting in U.S. in Trump's first 100 days?</a>. However, certain markets involve predicting polling numbers or election winners, which are possible to model with machine learning techniques.

In this post, I'm going to focus on the market <a href='https://www.predictit.org/Market/2845/What-will-congressional-job-approval-be-on-January-9'>What will congressional job approval be on January 9?</a> (today is January 4). The congressional job approval is taken from a polling average aggregated on the website <a href='http://www.realclearpolitics.com/epolls/other/congressional_job_approval-903.html'>RealClearPolitics<a/>, which averages polls from Gallup, Monmouth, and Economist/YouGov, among others. There are five possible buckets for the average job approval on January 9, and users can either "Buy Yes" or "Buy No" on each outcome. 

![Congress PredictIt Screenshot]({{site.base_url}}/assets/images/gp_predictit_blog/congress_predictit_screenshot.pdf)

For example, in the screenshot above, we can "Buy Yes" on "14.0% - 14.4%" for $0.27. We will then be rewarded with $0.73 if the average congressional job approval is between 14.0% and 14.4%, and we will lose our $0.27 otherwise. Thus, if we believe that the probability of the job approval being in this range is larger than 27%, we should buy this share. Similarly, we can "Buy No" for $0.79, and if the job approval is _not_ in this range we will be rewarded with $0.21.



> This is a
> blockquote
> with
>
> two paragraphs

This is a list:
* A
* B
* C

If your list items span multiple paragraphs, intend the items with three spaces.
Here the list is enumerated.

1.   This is the first sentence.

     This is the second sentence.

2.   And so on...

3.   And so forth...

This is **bold text**.
This is _italic text_.
This is ~~strikeout text~~.

This is an inline equation: $$a^2 + b^2 = c^2$$. You have to use two
dollar signs to open and close, not one like in Latex.
To have a centered equation, write it as a pararaph that starts and
ends with the two dollar signs:

$$
p(\theta \, | \, y) \propto p(\theta) \, 
p(y \, | \, \theta).
$$

I don't think you can do align blocks yet.

This is `inline  code`. 
Code blocks are intended paragraphs with four spaces:

```python
F = lambda n: ((1+np.sqrt(5))**n - (1-np.sqrt(5))**n) / (2**n * np.sqrt(5))
```
This is a figure. Note that `site.base_url` refers to the homepage.
In this case, `abc.png` is located in the `img` folder under root.

![ABC]({{site.base_url}}/img/abc.png)

### References
I've just been copying and pastying references as follows: 

[1] Meeds, Edward, Robert Leenders, and Max Welling. "Hamiltonian ABC." _arXiv preprint arXiv:1503.01916_ (2015). [link](http://arxiv.org/pdf/1503.01916)
...

### Footnotes
Here's my trick for footnotes. You can write HTML inside markdown, so I just create a
div with the id footnotes and then add a link[<sup>1</sup>](#footnotes)

<div id="footnotes"></div>
1. like this.
2. ...

