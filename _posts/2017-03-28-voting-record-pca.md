---
title: US Senators and PCA
layout: post
date: 2017-03-28 0:00
headerImage: false
blog: true
star: false
author: keyonvafa
description: A follow-up to ideal point modeling, comparing with PCA
---

A couple of weeks ago, I wrote a <a href='http://keyonvafa.com/ideal-points/'>blog post about modeling ideal points of US senators</a>. I wanted to follow up (very briefly), since I was curious about comparing the Bayesian method there with Principal Component Analysis (PCA).

Here are the (new) results performing PCA on the voting record:

<iframe width="1000" height="300" frameborder="0" scrolling="no" src="https://plot.ly/~keyonvafa/114.embed"></iframe> 

Here are the (older) results using ideal point modeling:

<iframe width="1000" height="300" frameborder="0" scrolling="no" src="https://plot.ly/~keyonvafa/58.embed"></iframe> 

It's interesting to compare the methods (the scale on the x-axis is irrelevant). Both models do a good job of capturing the more moderate senators, since <a href='https://en.wikipedia.org/wiki/Susan_Collins'>Susan Collins</a>, <a href='https://en.wikipedia.org/wiki/Lisa_Murkowski'>Lisa Murkowski</a>, and <a href='https://en.wikipedia.org/wiki/Kelly_Ayotte'>Kelly Ayotte</a> are in the middle in both methods. The furthest left senator using PCA is <a href='https://en.wikipedia.org/wiki/Maria_Cantwell'>Maria Cantwell</a>, who is also pretty far left with ideal points. Meanwhile, the furthest right senator with PCA is <a href='https://en.wikipedia.org/wiki/Tom_Coburn'>Tom Coburn</a> (whose <a href='https://en.wikipedia.org/wiki/Tom_Coburn'>Wikipedia page</a> describes him as "the godfather of the modern conservative, austerity movement"), yet he is further left than 8 senators with ideal point modeling. 

Overall, I was surprised by how similar these results were, given how differently the two methods are motivated. Ideal point modeling yields scores for every bill and senator (along with a predictive interpretation), while PCA can reduce the voting data to any dimension to capture senator voting habits (not to mention it's much faster). I would definitely be interested in exploring these methods with more rigor.  