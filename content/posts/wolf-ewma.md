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
by replacing the Gaussian likelihood with a loss function.
Despite its simplicity, the terminology behind the WoLF method (or the KF more generally), might not be familiar to everyone
and the method might seem a bit abstract for some.

Thus, to show the practical utility our method to a more general audience, in this post,
I will show a classical use case of the KF that is familiar to many: the exponentially weighted moving average (EWMA).
First, I recap the EWMA.
Then, I introduce the KF in 1d and show how it is a generalization of the EWMA.
Next, I show how the WoLF method can be used to robustify the EWMA.
I conclude this post by showing some numerical experiments that illustrate the robustness of the WoLF method in one-dimensional financial data.