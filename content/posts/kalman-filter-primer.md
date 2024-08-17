---
title: "On signal-plus-noise models"
date: 2024-08-04
description: "Primer on the Kalman filter primer."
katex: true
draft: true
tags: [kalman-filter, summary]
tldr: "An introduction to signal-plus-noise models and their best linear estimates."
---

In the book "A Kalman Filter Primer" by R.L. Eubank,
the author provides a _no-frills_ introduction to the Kalman filter.

# Signal plus noise models
The story of the Kalman filter begins with signal plus noise models.
By this, we mean that the observations {{< math >}}$y_{1:t}${{< /math >}}
can be written as the sum of two components:
a predictable component
{{< math >}}$f_{1:t}${{< /math >}}
and an unpredictable component
{{< math >}}${\bf e}_{1:t}${{< /math >}}.
{{< math >}}
$$
y_{1:t} =
    \underbrace{f_{1:t}}_{\text{signal}} + 
    \underbrace{{\bf e}_{1:t}}_{\text{noise}}.
$$
{{< /math >}}
By predictable, we mean that the that the covariance between the measurement and the signal is (not necessarily) non-diagonal.
By unpredictable, we mean that the covariance between the measurement and the noise is diagonal.
This means

{{< math >}}
$$
\begin{aligned}
    {\rm Cov}(y_t, f_j) &\neq 0, \\
    {\rm Cov}(y_t, e_j) &= 0.
\end{aligned}
$$
{{< /math >}}
for all $t \neq j$,
and
{{< math >}}
$$
    {\rm Cov}(e_t, e_t) = {\bf w}_t,
$$
{{< /math >}}
for all $t = 1, \ldots, T$. Here ${\bf w}_t$ is the covariance matrix of the noise at time $t$.

In other words, having $t \neq j$,
information of the signal at time $j$ could be contained in the observation at time $t$.
However, information of the noise at time $j$ is not contained in the observations at time $t$.


Having access to $y_{1:j}$ for some {{< math >}}$j \in \{1, \ldots, T\}${{< /math >}},
we seek to find the best linear estimate of the signal at time $t$.
Suppose
{{< math >}}$\mathbb{E}[f_k] = 0${{< /math >}} and
{{< math >}}$\mathbb{E}[{\bf e}_k] = 0${{< /math >}}
for all $k = 1, \ldots, T$.
Then, the best linear estimate of the signal at time $t$ is given by
{{< math >}}
$$
    {\bf A}_\text{opt}
    = \argmin_{\bf A}\mathbb{E}\left[\|f_t - {\bf A}\,y_{1:j}\|^2\right]
    = {\rm Cov}(f_t, y_{1:j})\,{\rm Var}(y_{1:j})^{-1}.
$$
{{< /math >}}
So that the best linear estimate of the signal at time $t$ is
{{< math >}}
$$
\begin{aligned}
    f_{t|j}
    &= {\bf A}_\text{opt}\,y_{1:j} \\
    &= {\rm Cov}(f_t, y_{1:j})\,{\rm Var}(y_{1:j})^{-1}\,y_{1:j}.
\end{aligned}
$$
{{< /math >}}

Furthermore,
the error variance-covariance matrix of the best linear estimate is
{{< math >}}
$$
    S_{t|j} = {\rm Var}(f_t - f_{t|j}) = {\rm Var}(f_t) - {\bf A}_\text{opt}\,{\rm Var}(y_{1:j})\,{\bf A}_\text{opt}^\intercal.
$$
{{< /math >}}


# Introduction of innovations

# State-space formulation

# The fundamental covariance structure