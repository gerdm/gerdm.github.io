---
title: "An outlier-robust EWMA"
date: 2024-07-13
description: "WoLF for robust estimation of an exponentially weighted moving average."
katex: true
draft: true
tags: [kalman-filter, wolf, ewma]
tldr: "An outlier robust exponentially weighted moving average via the WoLF method."
---

Our weighted-observation likelihood filter (WolF) got accepted at ICML 2024.
Our method is a provably-robust and easy to implement method that *robustifies* the Kalman filter (KF)
by replacing the classical Gaussian likelihood assumption with a loss function.
Despite its simplicity, the terminology behind the WoLF method (or the KF more generally), might not be familiar to everyone
and the method might seem a bit abstract for newcomers.

Thus, to show the practical utility our method,
I will show how to create a robust variant of the KF that is familiar to many: the exponentially weighted moving average (EWMA).

This post is organlised as follows:
first, I recap the EWMA.
Then, I introduce the one-dimensional state-space model (SSM) and show that the EWMA is a special case of the Kalman filter in one dimension.
Next, I derive the WoLF method for an EWMA.
I conclude this post by showing a numerical experiment that illustrates the robustness of the WoLF method in one-dimensional financial data.

# The exponentially weighted moving average (EWMA)
Given a sequence of observations $y_{1:t} = (y_1, \ldots, y_t)$, the EWMA is defined as
$$
z_t = \beta y_t + (1-\beta) z_{t-1}
$$
where $\beta \in (0,1]$ is the smoothing factor.
Higher levels of $\beta$ give more weight to recent observations.

# The Kalman filter in one dimension

# The WoLF method for the EWMA

# Numerical experiments

# Conclusion