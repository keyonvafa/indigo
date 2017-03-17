---
title: Variational Inference for Bayesian Probit Regression
layout: post
date: 2017-03-16 20:00
headerImage: false
blog: true
star: false
author: keyonvafa
description: A brief tutorial covering variational inference for Bayesian probit regression.
---

Variational inference has become one of the most important approximate inference techniques for Bayesian statistics, but it has taken me a long time to wrap my head around the central ideas (and I'm still learning). Since I've found that going through examples is the most efficient way to learn, I thought I would go through a single example in this post, performing variational inference on Bayesian probit regression.

I'm going to assume the reader is somewhat familiar with the basic ideas behind variational inference. If you've never seen variational infererence before, I strongly recommend <a href ='https://arxiv.org/pdf/1601.00670.pdf'>this tutorial</a> by <a href='http://www.cs.columbia.edu/~blei/'>David Blei</a>, <a href='http://www.proditus.com/'>Alp Kucukelbir</a>, and <a href='https://www.stat.berkeley.edu/~jon/'>Jon McAuliffe</a>. These <a href='https://www.cs.princeton.edu/courses/archive/fall11/cos597C/lectures/variational-inference-i.pdf'>course notes</a> from David Blei are also very <a href='https://www.youtube.com/watch?v=eXiwYUCe_bY'>handy</a>.

## Variational Inference: A (Very) Brief Overview 

Bayesian statistics often requires computing the conditional density $$p(\boldsymbol z \vert \boldsymbol x)$$ of latent variables $$\boldsymbol z = z_{1:m}$$ given observed variables $$\boldsymbol x = x_{1:n}$$. Since this distribution is typically intractable, variational inference learns an approximate distribution $$q(\boldsymbol z)$$ that is meant to be "close" to $$p(\boldsymbol z \vert \boldsymbol x)$$, using <a href='https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence'>Kullback-Leibler divergence</a> as a measure.  

Thus, there are two steps. The first comes from providing a form for the variational distribution, $$q(\boldsymbol z)$$. The most frequently used form comes from the _mean-field variational family_, where $$q$$ factors into conditionally independent distributions each governed by some set of parameters, $$q(\boldsymbol z) = \prod_{j=1}^m q_j(z_j)$$. Once we have specified the factorization of the distribution, we are still required to figure out the optimal form of each factor, both in terms of its family and parameters (although these can be conisdered the same thing). Thus, the second step is optimizing $$KL(q \vert \vert p)$$.

It turns out the optimal form of each factor is straightforward: $$q_j^*(z_j) \propto \exp\left\{E_{-j}[\log p(\boldsymbol z, \boldsymbol x)]\right\}$$, where $$E_{-j}[\cdot]$$ refers to the expectation when omitting variable $$z_j$$. To minimize $$KL(q \vert \vert p)$$, we cycle between latent factors $$q_j$$ and update the mean (with respect to the current parameters) according to the equation above. If these results are unfamiliar, definitely check out <a href ='https://arxiv.org/pdf/1601.00670.pdf'>the tutorial</a> I mentioned earlier.   

## Variational Inference for Bayesian Probit Regression

Consider a probit regression problem, where we have data $$\boldsymbol x \in \mathbb{R}^{N \times 1}$$ and a binary outcome $$\boldsymbol y \in \{0,1\}^{N}$$. In probit regression, we assume $$p(y_i = 1) = \Phi(a + bx_i)$$, where $$a$$ and $$b$$ are unknown and random, with a uniform prior, and $$\Phi(\cdot)$$ is the standard normal CDF. To simplify things, we can introduce variables $$z_i \sim \mathcal{N}(a+bx_i,1)$$ so $$y_i = 1$$ if $$z_i > 0$$ and $$y_i = 0$$ if $$z_i \leq 0$$. 

The first step is writing down the log posterior density $$\log p(a,b,\boldsymbol z \vert \boldsymbol y)$$ up to a constant. It is straightforward to see

$$
\log p(a, b, \boldsymbol z \vert \boldsymbol y) \propto \sum_{i=1}^n  y_i \log I(z_i > 0) + (1-y_i)\log(I(z_i \leq 0)) - \frac{\sum_{i=1}^n (z_i - (a+bx_i))^2}{2}.
$$

The next step is defining our variational distribution $$q$$. We will provide one factor for each $$z_i$$, along with indendent factors for $$a$$ and $$b$$ each. Therefore, $$q$$ consists of $$n + 2$$ independent factors:

$$
q(a, b, \boldsymbol z) = q_a(a) q_b(b) \prod_{j=1}^m q_j(z_j).
$$

To learn the optimal form of each factor, we use the rule described above. That is, consider a single $$z_j$$. The optimal distribution is therefore $$q_j^*(z_j) \propto \exp \left\{E_{a,b,\boldsymbol z_{-j}}[\log p(a, b, \boldsymbol z \vert \boldsymbol y)]\right\}$$. Writing this out, we see 

$$
E_{a,b,\boldsymbol z_{-j}}[\log p(a, b, \boldsymbol z \vert \boldsymbol y)] \propto y_j \log I(z_j > 0) + (1-y_j)\log I(z_j \leq 0) - \frac{E_{a,b}(z_j-(a+bx_i))^2}{2}.
$$

Thus, after exponentiating, we have that the ideal form is a truncated normal distribution. That is, $$q_j(z_j) \sim \mathcal N^+(E(a)+E(b)x_i,1)$$ if $$y_j = 1$$ and $$q_j(z_j) \sim \mathcal N^-(E(a)+E(b)x_i,1)$$ if $$y_j = 0$$, where $$\mathcal N^+$$ and $$\mathcal N^-$$ are normal distributions truncated to be positive and negative, respecitively. 

Similarly, for $$a$$, we have $$E_{b,\boldsymbol z}[\log p(a, b, \boldsymbol z \vert \boldsymbol y)] \propto E_{b,\boldsymbol z}\left(-\frac{\sum_{i=1}^n (z_i - (a+bx_i))^2}{2}\right)$$. Removing terms that do not depend on $$a$$ and completing the square, we have the optimal form as $$q_a(a) \sim \mathcal N\left(\frac{\sum_{i=1}^n [E(z_i)-E(b)x_i]}{n},\frac{1}{n}\right)$$.

Finally, for $$b$$, we have $$E_{a,\boldsymbol z}[\log p(a, b, \boldsymbol z)] \propto E_{a, \boldsymbol z}\left(-\frac{\sum_{i=1}^n (z_i - (a+ bx_i))^2}{2}\right)$$. Again removing the terms that do not depend on $$b$$ and completing the square, we have the following optimal form:

$$
q_b(b) \sim \mathcal N \left(\frac{\sum_{i=1}^n x_i[E(z_i)-E(a)]}{\sum_{=1}^n x_i^2}, \frac{1}{\sum_{i=1}^n x_i^2}\right).
$$

Now that we know the form of all the factors, it's time to optimize. To do this, we set each parameter to the mean of its optimal factored distribution. The updates can take the following form in R:

```R
update_M_zj = function(M_a,M_b,j) {
  mu = M_a + M_b*x[j]
  if (y[j] == 1) {
    return(mu + dnorm(-1*mu)/(1-pnorm(-1*mu)))
  } else {
    return(mu - dnorm(-1*mu)/(pnorm(-1*mu)))
  }
}
update_M_a = function(M_z,M_b) {
  return(sum(M_z-M_b*x)/n)
}
update_M_b = function(M_z,M_a) {
  return(sum(x*(M_z-M_a))/sum(x^2))
}
```

Thefore, a single updating step would look like

```R
for (i in 1:n) {
  M_z[iteration] = update_M_zj(M_a,M_b,i)
}
M_a = update_M_a(M_z,M_b)
M_b = update_M_b(M_z,M_a)
as[iteration] = M_a
bs[iteration] = M_b
```

Again, variational inference is an incredibly powerful tool, and I cannot overstate how helpful the links I posted above are in understanding all of this. Hopefully this tutorial clears up some of the confusion about variational inferece. 






