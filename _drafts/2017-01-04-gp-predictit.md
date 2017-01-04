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

In this post, I'll provide a quick tutorial on using Gaussian processes for regression and walk through one of my favorite applications: betting on political data on the website PredictIt. This post is intended for anyone who has taken an intro probability class and has basic machine learning experience, as my main goal is to provide intuition for Gaussian processes. Thus, it is not meant to be exhaustive at all. Be sure to check out Carl Rasmussen and Christopher Williams's excellent textbook Gaussian Processes for Machine Learning (<a href="http://www.gaussianprocess.org/gpml/">available for free online</a>) for a more comprehensive reference. 

We begin by overviewing Gaussian processes (GPs). If you would like to skip the overview and go straight to the betting example, go to [click on this link](#my-multi-word-header). 

What is a Gaussian process? Frequently, it is referred to as the infinite-dimensional extension of the multivariate normal distribution (CITE?). This may be confusing, because we typically don't observe random variables with infinitely many components. However, when we work with GPs, the intuition is that we observe some finite-dimensional subset of infinite-dimensional data, and this finite subset follows a multivariate normal distribution, as would every finite subset.

### My Multi Word Header


# This is a first level heading

## This is a second level heading

## This is a third level heading

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

