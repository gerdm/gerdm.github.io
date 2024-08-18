---
title: "On signal-plus-noise models"
date: 2024-08-04
description: "Primer on the Kalman filter primer."
katex: true
draft: true
tags: [kalman-filter, summary]
tldr: "An introduction to signal-plus-noise models and their best linear estimates."
---

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
    {\rm Cov}(y_t, e_j) &= \mathbb{1}(t = j)\,{\bf w}_t,
\end{aligned}
$$
{{< /math >}}
for all {{< math >}}$t,j \in \{ 1, \ldots, T\}${{< /math >}}.
Here ${\bf w}_t$ is the covariance matrix of the noise at time $t$.

<!-- In other words, having $t \neq j$,
information of the signal at time $j$ could be contained in the observation at time $t$.
However, information of the noise at time $j$ is not contained in the observations at time $t$. -->


Suppose
{{< math >}}$\mathbb{E}[f_k] = 0${{< /math >}} and
{{< math >}}$\mathbb{E}[{\bf e}_k] = 0${{< /math >}}
for all $k = 1, \ldots, T$.
Then, having access to $y_{1:j}$, for some {{< math >}}$j \in \{1, \ldots, T\}${{< /math >}},
the best linear predictor (BLUP) of the signal at time $t$ is given by
{{< math >}}
$$
    {\bf A}_\text{opt}
    = \argmin_{\bf A}\mathbb{E}\left[\|f_t - {\bf A}\,y_{1:j}\|^2\right]
    = {\rm Cov}(f_t, y_{1:j})\,{\rm Var}(y_{1:j})^{-1}.
$$
{{< /math >}}
As a consequence, the best linear estimate of the signal at time $t$,
having access to $y_{1:j}$, is
{{< math >}}
$$
\begin{aligned}
\tag{BLUP.1}
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
One of the main challenges in computing ${\rm (BLUP.1)}$ above is that its computation
grows quadratically with the number of observations $y_{1:j}$.
This is because of the term ${\rm Var}(y_{1:j})^{-1}$.

To overcome the above, we introduce the concept of an innovation.
The innovation at time $t$ is defined as
{{< math >}}
$$
\tag{i.1}
    \varepsilon_t = y_t - \sum_{k=1}^{t-1} {\rm Cov}(y_t, \varepsilon_k)\,{R}_k^{-1}\,\varepsilon_k.
$$
{{< /math >}}
with $R_k = {\rm Var}(\varepsilon_k)$.

It can be shown that {{< math >}}${\rm Cov}(\varepsilon_t, \varepsilon_k) = 0${{< /math >}} for all $t \neq k$.
So that
{{< math >}}
$$
    {\rm Var}(\varepsilon_{1:t}) = {\rm diag}(R_1, \ldots, R_t).
$$
{{< /math >}}
As a consequence,
{{< math >}}
$$
    y_{1:j} = {\bf L}\,\varepsilon_{1:j},
$$
{{< /math >}}
with
{{< math >}}
$$
    {\bf L}_{t,j} =
    \begin{cases}
    {\rm Cov}(y_t, \varepsilon_j)\,R_j^{-1} & \text{if } j < t, \\
    {\bf I} & \text{if } j = t, \\
    {\bf 0 } & \text{if } j > t.
    \end{cases}
$$
{{< /math >}}

Then, the BLUP of the signal up to index $j$ can be written as
{{< math >}}
$$
\begin{aligned}
\tag{BLUP.2}
    f_{t|j}
    &= {\rm Cov}(f_t, {\bf L}\,\varepsilon_{1:j})\,{\rm Var}({\bf L}\,\varepsilon_{1:j})^{-1}\,{\bf L}\,\varepsilon_{1:j}\\
    &= \sum_{k=1}^j {\rm Cov}(f_t, \varepsilon_k)\,R_k^{-1}\,\varepsilon_k.
\end{aligned}
$$
{{< /math >}}

And the error variance-covariance matrix of the BLUP takes the form
{{< math >}}
$$
\begin{aligned}
    S_{t|j} = {\rm Var}(f_t) - \sum_{k=1}^j{\rm Cov}(f_t, \varepsilon_k)\,R_k^{-1}\,{\rm Cov}(\varepsilon_k, f_t).
\end{aligned}
$$
{{< /math >}}