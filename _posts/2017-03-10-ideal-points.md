---
title: Ideal Points of US Senators
layout: post
date: 2017-03-09 2:00
headerImage: false
blog: true
star: true
author: keyonvafa
description: Bayesian ideal point modeling to assess the political views of US senators based off voting records.  
---

<a href='http://k7moa.com/pdf/Upside_Down-A_Spatial_Model_for_Legislative_Roll_Call_Analysis_1983.pdf'>Popularized by Keith Poole and Howard Rosenthal</a>, ideal point modeling is a powerful way to extract the relative ideologies of politicans based solely on their voting records. <a href='http://www.acrwebsite.org/search/view-conference-proceedings.aspx?Id=9188'>A</a> <a href='http://www.stat.columbia.edu/~gelman/research/published/171.pdf'>lot</a> <a href='https://www.cs.princeton.edu/~blei/papers/GerrishBlei2011.pdf'>has</a> <a href='http://pablobarbera.com/static/barbera_twitter_ideal_points.pdf'>been</a> <a href='https://www.jstor.org/stable/1558585'>written</a> on ideal point models, so I'm not going to add anything new, but I wanted to give a brief overview of the Bayesian perspective.

First, some results. The following plot shows the ideal points (essentially inferred ideologies) of US senators based solely on roll call voting from 2013-2015 (scroll over the points to see names): 

<iframe width="1000" height="300" frameborder="0" scrolling="no" src="https://plot.ly/~keyonvafa/58.embed"></iframe>

More extreme scores (i.e. away from zero) represent more extreme political views. While the liberal-conservative spectrum is not explicitly encoded into the model, the model picks this up naturally from voting patterns. On the far left are some of the most liberal members of the US Senate, such as <a href='https://en.wikipedia.org/wiki/Brian_Schatz'>Brian Schatz</a>, while the far right has some of the most conservative members, such as <a href='https://en.wikipedia.org/wiki/Jim_Risch'>Jim Risch</a> and <a href='https://en.wikipedia.org/wiki/Ted_Cruz'>Ted Cruz</a>. In the middle are senators sometimes referred to as <a href='https://en.wikipedia.org/wiki/Democrat_In_Name_Only'>DINOs</a> and <a href='https://en.wikipedia.org/wiki/Republican_In_Name_Only'>RINOs</a>, such as <a href='https://en.wikipedia.org/wiki/Joe_Manchin'>Joe Manchin</a>, <a href='https://en.wikipedia.org/wiki/Susan_Collins'>Susan Collins</a>, and <a href='https://en.wikipedia.org/wiki/Lisa_Murkowski'>Lisa Murkowski</a>.

The basic model is as follows. Consider a legislator $$u$$ and a particular bill $$d$$. The vote $$u$$ places on $$d$$ is denoted as a binary variable, $$v_{ud} = 1$$ for Yea and $$v_{ud} = 0$$ for Nay. Each legislator has an _ideal point_ $$x_u$$; a value of 0 is political neutrality, whereas large values in either direction indicate more political extremism in the respective direction. Every bill has its own _discrimination_ $$b_d$$, which is on the same scale as the ideal points for legislators. If $$x_u*b_d$$ is high, the legislator is likely to vote for the bill, and if the value is low, the legislator is less likely to vote. Finally, each bill also has an offset $$a_d$$ that indicates how popular the bill is overall, regardless of political affiliation. Formally, the model is as follows:

$$
P(v_{ud} = 1) = \sigma(x_ib_d + a_d),
$$

where $$\sigma(\cdot)$$ is some sigmoidal function, such as the inverse-logit or the standard normal CDF. If a senator didn't vote on a particular bill, this data is considered missing at random. 

Inference requires learning the vectors $$X, B$$, and $$A$$. I took a Bayesian approach and put (independent) normal priors on each variable. I then used an EM algorithm derived by <a href='http://imai.princeton.edu/research/files/fastideal.pdf'>Kosuke Imai et al</a>. The E-Step and M-Step are described in full detail in the paper, and I followed their setup, except I removed senators with less than 50 votes, and I stopped after 500 iterations.

All my code is available <a href='https://github.com/keyonvafa/ideal-point-blog'>here</a>. 
