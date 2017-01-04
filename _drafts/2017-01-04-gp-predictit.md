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

_This is the second part of a two part blog post on Gaussian processes. If you would like an overview of Gaussian processes, head over to the first part._

## Betting on PredictIt with Gaussian Processes

One of my favorite websites is <a href='https://www.predictit.org/'> PredictIt</a>, which provides prediction markets for betting on politics. Many of these markets are not explicitly quantitative, such as <a href="https://www.predictit.org/Contract/4500/Will-there-be-a-Putin-Trump-meeting-in-US-in-Trump's-first-100-days#dat">Will there be a Putin-Trump meeting in U.S. in Trump's first 100 days?</a>. However, certain markets involve predicting polling numbers or election winners, which are possible to model with machine learning techniques.

In this post, I'm going to focus on the market <a href='https://www.predictit.org/Market/2845/What-will-congressional-job-approval-be-on-January-9'>What will congressional job approval be on January 9?</a> (today is January 4). The congressional job approval is taken from a polling average aggregated on the website <a href='http://www.realclearpolitics.com/epolls/other/congressional_job_approval-903.html'>RealClearPolitics<a/>, which averages polls from Gallup, Monmouth, and Economist/YouGov, among others. There are five possible buckets for the average job approval on January 9, and users can either "Buy Yes" or "Buy No" on each outcome. 

![Congress PredictIt Screenshot]({{site.base_url}}/assets/images/gp_predictit_blog/congress_predictit_screenshot.pdf)

For example, in the screenshot above, we can "Buy Yes" on "14.0% - 14.4%" for $0.27. We will then be rewarded with $0.73 if the average congressional job approval is between 14.0% and 14.4% on January 9, and we will lose our $0.27 otherwise. Thus, if we believe that the probability of the job approval being in this range is larger than 27%, we should buy this share. Similarly, we can "Buy No" for $0.79, and if the job approval is _not_ in this range we will be rewarded with $0.21.

In R, I scraped the past 1000 days of approval data from the RealClearPolitics website. If I train a GP on these data points, I can not only make predictions for the average job approval for January 9, but I can also use the predictive variance to assess my certainty of these estimates. This makes it possible to calculate the probability of being in any of the five buckets on PredictIt under my model. Thus, if one of my probabilities is significantly different from any of the market values, I should go ahead and buy shares and make $$$ (assuming, of course, that this model is correct).

A few words of caution before proceeding: this model is not correct. There are many possible ways to model time series data, and a GP is just one of them. Any model that can generate predictions and uncertainty estimates would be exciting to try, and I used a GP because it's one of my favorite models. Additionally, we are trying to model polling averages, so treating each poll individually should provide more fruitful estimates. Finally, the prices on PredictIt reflect more knowledge than simply the past 1,000 days; they take into account current events, such as whether Congress just did something unpopular (such as <a href='http://www.nytimes.com/2017/01/02/us/politics/with-no-warning-house-republicans-vote-to-hobble-independent-ethics-office.html'>trying to remove an ethics committee</a>), and the schedule of poll releases. Our model uses only prior data, so somehow accounting for these factors would make it more robust.

Once I scraped the data, I removed consecutive days where the average didn't change. I did this becuase there are some days where no new polls are added to the aggregate, so I removed them instead of accounting for these days in the model. The data is below:

![Raw Data]({{site.base_url}}/assets/images/gp_predictit_blog/raw_data.png)

I chose to use a constant mean function and the rational quadratic covariance function (RQ) as my kernel. This is an area where the model can be significantly improved: I would imagine it's more appropriate to use some combination of kernels, especially if there is a linear or periodic effect. However, the RQ kernel appeared to provide sensible results, so I stuck with it.

I then used R's `optim` command to optimize the log-likelihood $$p(\boldsymbol y | \boldsymbol X, \boldsymbol \theta)$$ where $$\boldsymbol y$$ is the polling data averages, $$\boldsymbol X$$ consists of the days corresponding to the polling data, and $$\boldsymbol \theta$$ are the RQ hyper-parameters along with the noise-scale $$\sigma^2_{\epsilon}$$ and the constant mean parameter. Because some of these parameters were constrained to be positive, I parameterized them by their log when optimizing, which could be any real value.

After choosing my hyper-parameters, I calculated the predictive mean for the past 1000 days and the next 100 along with the 95% confidence interval using the predictive covariance:

![Predictions]({{site.base_url}}/assets/images/gp_predictit_blog/predictions.png)

The dashed red-line represents the predictive mean at each time point, and the shaded purple represents the 95% confidence interval. As we can see, the data fits nicely, and, as desired, the uncertainty of the estimates increases as we look at the next 100 days. If anything, there appears to be a slight downward trend as we head into the future, perhaps reflecting that most recently, the approval ratings have been decreasing. 

Finally, I calculated my estimates for each PredictIt bucket for January 9:

| Approval Rating | "Yes" Price   | "Yes" Probability  | "No" Price | "No" Probability |
| :-------------: |:-------------:| :-----------------:|:----------:|:----------------:|
| 15.0% or higher | $.04          | 20%                | $.97       | 80%              |
| 14.5% - 14.9%   | $.75          | 31%                | $.33       | 69%              |
| 14.0% - 14.4%   | $.27          | 30%                | $.79       | 70%              |
| 13.5% - 13.9%   | $.11          | 14%                | $.96       | 86%              |
| 13.4% or lower  | $.03          | 4%                 | $.97       | 96%              |

Thus, the most under-valued markets are buying "Yes" for "15.0% or higher" ($.04 for 20%) and buying "No" for "14.5% - 14.9%" ($.33 for 69%). The remaining markets more-or-less align. 

I find it interesting that the "Yes" price is so high for "14.5% - 14.9%" compared to my model. The most recent polling average was 14.5 for January 3, which is right on the border of the second and third buckets (recall we're trying to make estimates for January 9). My guess is that this reflects some outside knowledge such as no polls being conducted in the next 5 days. 

Regardless, I bought 50 shares of "Yes" for "15.0% or higher" (for a total of $2.00) and 12 shares of "No" for "14.5% - 14.9" (for a total of $3.96). I'll be sure to provide updates with how much money I win/lose. I'd love to try out more elaborate kernels or even a <a href="{{site.base_url}}/deep-gaussian-processes/">deep Gaussian Process</a> in future posts and see how these models fare.

All my code is available here (LINK). 
