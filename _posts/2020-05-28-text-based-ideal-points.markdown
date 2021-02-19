---
title: "Text-Based Ideal Points"
layout: post
date: 2020-05-28 16:00
headerImage: false
projects: true
hidden: true # don't count this post in blog pagination
author: keyonvafa
externalLink: false
---

---

[[PDF](https://www.aclweb.org/anthology/2020.acl-main.475.pdf)] [[Code](https://github.com/keyonvafa/tbip)] [[Tutorial](https://colab.research.google.com/drive/1_KkVI2lGtPdgsHSKDIMhSLCKkHvBQ4LO?usp=sharing)] [[Slides]({{site.base_url}}/assets/slides/tbip_slides.pdf)] [[Video](https://slideslive.com/38929238/textbased-ideal-points)]


Ideal point models analyze lawmakers' votes to quantify their political positions, or ideal points. But votes are not the only way to express a political position. Lawmakers also give speeches, release press statements, and post tweets. [In this paper](https://www.aclweb.org/anthology/2020.acl-main.475/), we introduce the text-based ideal point model (TBIP), an unsupervised probabilistic topic model that analyzes texts to quantify the political positions of its authors. We demonstrate the TBIP with two types of politicized text data: U.S. Senate speeches and senator tweets. Though the model does not analyze their votes or political affiliations, the TBIP separates lawmakers by party, learns interpretable politicized topics, and infers ideal points close to the classical vote-based ideal points. One benefit of analyzing texts, as opposed to votes, is that the TBIP can estimate ideal points of anyone who authors political texts, including non-voting actors. To this end, we use it to study tweets from the 2020 Democratic presidential candidates. Using only the texts of their tweets, it identifies them along an interpretable progressive-to-moderate spectrum.

<!-- [PyTorch](https://github.com/keyonvafa/tbip/blob/master/pytorch/tbip.py) and [Tensorflow](https://github.com/keyonvafa/tbip/blob/master/tbip.py) implementations available on [Github](https://github.com/keyonvafa/tbip). -->

[Notebook with tutorial available on Colab](https://colab.research.google.com/drive/1_KkVI2lGtPdgsHSKDIMhSLCKkHvBQ4LO?usp=sharing).


The plots below show examples of ideological topics for U.S. Senate speeches (2015-2017). Move the slider to see how the ideological topic changes as a function of ideal point:

<iframe width="900" height="600" frameborder="0" scrolling="no" src="//plotly.com/~keyonvafa/256.embed?&link=false"></iframe>

<iframe width="900" height="600" frameborder="0" scrolling="no" src="//plotly.com/~keyonvafa/252.embed?&link=false"></iframe>

<iframe width="900" height="600" frameborder="0" scrolling="no" src="//plotly.com/~keyonvafa/250.embed?&link=false"></iframe>

<iframe width="900" height="600" frameborder="0" scrolling="no" src="//plotly.com/~keyonvafa/254.embed?&link=false"></iframe>

---

K. Vafa, S. Naidu, and D. Blei. [**Text-Based Ideal Points**](https://www.aclweb.org/anthology/2020.acl-main.475/). In _Proceedings of ACL_, 2020.


<!-- ---

<iframe width="900" height="600" frameborder="0" scrolling="no" src="//plotly.com/~keyonvafa/228.embed"></iframe> -->

<!-- 
![Senate speech ideal point comparisons]({{ site.url }}/assets/images/projects/senate_ideal_point_comparisons.jpg)
<figcaption class="caption">The ideal points learned by the TBIP for senator speeches and tweets are highly correlated with the classical vote ideal points. Senators are coded by their political party (Democrats in blue circles, Republicans in red x’s). Although the algorithm does not have access to these labels, the TBIP almost completely separates parties.</figcaption>

 -->

