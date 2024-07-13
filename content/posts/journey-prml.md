---
title: "A Journey through Pattern Recognition and Machine Learning"
date: 2020-09-10T00:00:00+00:00
description: "A journey through the book Pattern Recognition and Machine Learning from Christopher Bishop."
katex: false
draft: false
tags: [thoughts, data-analysis]
---

On July 10, 2018 I committed myself to read and do all the exercises of the book Pattern Recognition and Machine Learning from Christopher Bishop. After seven hundred and ninety two days, I can finally say I accomplished my goal.

One of the main reasons why I decided to study the book from cover to cover was to demystify machine learning (ML). Reading through the book taught me to question the underlying assumptions behind every model, to understand from a mathematical point of view why some models fail, and to not fret the mathematics behind ML. Most importantly, I understood why ML is hard, to be humbled by it and to not take models as recipes that we can use as we please.

Whilst studying the book, I had a small side project that consisted in clocking every time I opened the book using the [Toggl application](https://toggl.com/). The goal of this blog post, therefore, is not to discuss the intricate chapters of the book, but rather to tell a story with data on how I managed to finish the book, my experiences and what data can tell me about my own patterns.

## Part1: Temporal Analysis

### Daily Hours

I started studying PRML on July 10, 2018 at 17:21pm. The first time I opened the book I did not know that I was going to read it from cover to cover. In fact, it was not until one month later, on august 27, 2018, the day of my birthday, that I decided to pursue a thorough study of the book. I do not recall much of that day, other than, according to my notes, I studied for 18 minutes. Interestingly though, my following two birthdays saw a rise the in the number of hours that I clocked:

|year|clocked hours|
|----|-------------|
|2018|0.3          |
|2019|1.25         |
|2020|2.43         |


Around that time, I decided to study for two hours every single day until the day I finished the book. Considering the 792 days that took me to study the book from cover to cover, I should’ve gotten 1,584 hours worth of study, but instead, I got 1,542.72 hours. A close 97% of my goal.

The following graph shows the total number of clocked hours (top plot) and my deficit of hours (bottom plot), i.e., the difference between the number of hours that I should’ve studied v.s. the actual number of hours that I studied.

![https://imgur.com/NwbZcnR.png](https://imgur.com/NwbZcnR.png)

An interesting observation from the graph above is to note the total number of days in which I did not clock a single hour of study is 48. Before starting doing this analysis I could’ve sworn that I had less than 5 days in which I did not study!

Another interesting observation from the previous graph is that I was on a time-deficit of almost 150 hours pre-covid19. But during the pandemic I managed to recover more than 100 hours worth of study, which led me to finish the book with a 43-hour deficit; or almost 21 days worth of study.

Without consideration of the days in which I did not clock my studies, I get an average study time of 2.17 hours per day. If I do consider them, I get an average of around 1.94 hours of study per day which, of course, is the 97% of 2.

The following graphs shows the distribution of hours of study per day without taking into account the 48 days in which I did not clock my studies. It shows that, on an average day, I would’ve studied between 1.4 and 3 hours.

![https://imgur.com/NZ17ZtR.png](https://imgur.com/NZ17ZtR.png)

distribution-clocked-hours

### Time of study

My decision to study two hours per day stemmed from the fact that I was working full-time on my startup. Even though I had more freedom to choose when I decided to start studying, I was constraint to my deadlines, meetings with clients, other models I was working on, etc. This, not only restricted how much could I study in one day, but also when did I start studying. In this section, I discuss my patterns in regards of my times of study.

The following histogram represents the distribution of times that I started clocking a study session. As we can see, it is centered at around 10 am, but it has a significant standard deviation of 4 hours. Meaning that roughly, on a normal day, I could’ve been studying anywhere from 6am to 2pm. Of course, these statistics do not truly represent what would happen on a normal day since the distribution seems to be right skewed and possibly multimodal.

![https://imgur.com/tnWVzvI.png](https://imgur.com/tnWVzvI.png)

time-of-start

In order to better understand my patterns of study we can consider the distribution of times that I started studying, grouped by weekday represented in the graph below. Clearly, Monday through Friday seem to be the days whose distributions look closest. Saturdays and Sundays however, diverge from the rest. Saturday, on the one hand, seems to have a mean that is closer to the early hours of the morning, while on the other hand, Sunday seems to be the weekday with higher probability of study in later hours of the day.

![https://imgur.com/mmQDQBF.png](https://imgur.com/mmQDQBF.png)

An additional way to distill how much time I dedicated myself to study is to consider the following *heatmap* which represents the total number of hours that I studied, grouped by day of the week and the rounded hour in which I started studying.

![https://imgur.com/2SwWDxF.png](https://imgur.com/2SwWDxF.png)

It came as a surprise to learn that my favorite day and hour of study was on Saturdays at 6am. In other timeframe did I get above 50 hours worth of study. Conversely, all workweek days seem pretty standard and not deviate much from the norm. On Sundays had the highest probability of studying after 9am. Finally, as with every dataset, there must be some outlier datapoints. In my case, it was the hours of study in very early in the morning, particularly on Thursdays.

## Part 2: Chapter Analysis

As it is with any other topic in mathematics, not al subjects in a book have the same difficulty. One rough estimation on how hard each chapter was for me is to consider the amount of hours that I spent on that chapter.

```
Chapter 1: Introduction                        128.755833
Chapter 2: Probability distributions           208.143611
Chapter 3: Linear Models for Regression        105.791389
Chapter 4: Linear Models for Classification    102.149167
Chapter 5: Neural Networks                     200.052120
Chapter 6: Kernel Methods                       61.676944
Chapter 7: Sparse Kernel Machines               79.763056
Chapter 8: Graphical Models                     96.968333
Chapter 9: Mixture Models and EM                79.607500
Chapter 10: Approximate Inference              226.593056
Chapter 11: Sampling Methods                    27.502500
Chapter 12: Continuous Latent Variables         91.390833
Chapter 13: Sequential Data                    104.260833
Chapter 14: Combining Models                    30.071667
Name: study-hours, dtype: float64
```

Before starting my analysis, I had the hypothesis that harder chapters must, of course, take longer to complete. What I did not consider, however, was the length of each chapter (which is easy to measure), the amount of hours that I dedicated myself to a particular exercise in the chapter (which I did not measure), and finally the *depth* of each chapter (which is hard to measure). Chapter 2, for example, was about topic I was already familiar with, namely, probability distributions. But despite this fact, it was the second chapter with most clocked hours. I did not feel that the chapter was particularly difficult, but the way in which the author talked about the subject made pay particularly attention to the details.

The following graph corrects for the number of pages in each chapter by dividing the total number of hours spent per chapter by the total number of pages in that chapter.

![https://imgur.com/5PrKNfY.png](https://imgur.com/5PrKNfY.png)

hours spent per chapter per page

Looking at the graph tells a slightly different story, but not the complete one. Chapter 14, for instance, was a high-pace chapter since it involved many concepts that were presented in earlier parts of the book, which made the reading easy to do and the exercises relatively easy to complete.

To finalize with this analysis, I can combined the previous two analyses to get to my final graph: the time spent on each chapter as a function of the date that I started studying it. The y-axis represents the total number of hours clocked per week, and the x-axis represents the end of each week.

![https://imgur.com/6DenEap.png](https://imgur.com/6DenEap.png)

Looking at this graphs makes me quickly forget that the datapoints represent me. Between every two dashed lines in the graph there was a Gerardo experiencing something: a great weekend at the beach, a rough day at work, a day of worries or happiness. It sometimes easy to forget that as we work with data about people, there are stories that we are not taking account for. Latent variables that could better explain our patterns of behavior.

## Conclusion

The book Pattern Recognition and Machine Learning became my day-to-day life for more than two years. Looking back at my experiences with the book in the form of data is, to me, the only reasonable way to make an homage to such an inspiring book.

Furthermore, the analyses of my patterns in the form of hard data can help me overcome my next big goal. There are many things I would’ve done differently such as being more rigorous on how my data was extracted or write in words how I feeling on a particular day. But the key lesson that I learned from this experience is that **consistency can take me anywhere**.

### Further comments

- All the code used to make this post can be found in [this](https://github.com/gerdm/misc/blob/master/2020-09/bishop-analysis.ipynb) notebook