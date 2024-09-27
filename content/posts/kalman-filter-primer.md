---
title: "Notes on the Kalman filter (I): signal plus noise models"
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
{{< math >}}${e}_{1:t}${{< /math >}}.
{{< math >}}
$$
\underbrace{y_{1:t}}_\text{measurement} =
    \underbrace{f_{1:t}}_{\text{signal}} + 
    \underbrace{{e}_{1:t}}_{\text{noise}}.
$$
{{< /math >}}
By predictable, we mean that the that the covariance between the measurement and the signal is (not necessarily) non-diagonal.
By unpredictable, we mean that the covariance between the measurement and the noise is diagonal.
This means

{{< math >}}
$$
\begin{aligned}
    {\rm Cov}(y_t, f_j) &\neq 0, \\
    {\rm Cov}(y_t, e_j) &= \mathbb{1}(t = j)\,{\bf R}_t,
\end{aligned}
$$
{{< /math >}}
for all {{< math >}}$t,j \in \{ 1, \ldots, T\}${{< /math >}}.
Here ${\bf R}_t$ is the covariance matrix of the noise at time $t$, which
by definition, is positive definite.

Finally, suppose
{{< math >}}$\mathbb{E}[f_k] = 0${{< /math >}} and
{{< math >}}$\mathbb{E}[{e}_k] = 0${{< /math >}}
for all $k = 1, \ldots, T$.

# The best-linear unbiased predictor (BLUP)
Suppose, $j,t\in{\cal T}$.
We seek to find the matrix ${\bf A} \in \reals^{d\times j}$ that maps from measurements $y_{1:j}$
to the signal $f_t$ so that
{{< math >}}
$$
    f_t \approx {\bf A}\,y_{1:j}.
$$
{{< /math >}}
That is, we want the matrix ${\bf A}$ that weights all observations up to time $j$
to make an estimate of the signal at time $t$.

Our notion of approximation is based on the matrix ${\bf A}$ that
minimises the expected L2 error
between the signal $f_t$ and the linear predictor
{{< math >}}${\bf A}\,y_{1:j}${{< /math >}}:
{{< math >}}
$$
    {\bf A}_\text{opt}
    = \argmin_{\bf A}\mathbb{E}\left[\|f_t - {\bf A}\,y_{1:j}\|^2\right]
    = {\rm Cov}(f_t, y_{1:j})\,{\rm Var}(y_{1:j})^{-1}.
$$
{{< /math >}}
As a consequence, the best linear unbiased predictor (BLUP) of the signal at time $t$,
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
One of the main challenges in computing ${\rm (BLUP.1)}$ above is that the computation
grows quadratically with the number of observations $y_{1:j}$.
This is because of the term ${\rm Var}(y_{1:j})^{-1}$,
which is an $j\times j$ positive definite matrix.

To overcome the above, we introduce the concept of an innovation.
The innovation at time $t$ is defined as
{{< math >}}
$$
\tag{I.1}
    \varepsilon_t =
    \begin{cases}
        y_1 & \text{for } t = 1,\\
        y_t - \sum_{k=1}^{t-1} {\rm Cov}(y_t, \varepsilon_k)\,{S}_k^{-1}\,\varepsilon_k & \text{for } t \geq 2,
    \end{cases}
$$
{{< /math >}}
and $S_k = {\rm Var}(\varepsilon_k)$.
It can be shown that {{< math >}}${\rm Cov}(\varepsilon_t, \varepsilon_k) = 0${{< /math >}} for all $t \neq k$.
So that
{{< math >}}
$$
    {\rm Var}(\varepsilon_{1:t}) = {\rm diag}(S_1, \ldots, S_t).
$$
{{< /math >}}

It can also be shown that the relationship between innovations and measurements satisfy
{{< math >}}
$$
\tag{I.2}
    y_{1:j} = {\bf L}\,\varepsilon_{1:j},
$$
{{< /math >}}
with
{{< math >}}
$$
    {\bf L}_{t,j} =
    \begin{cases}
    {\rm Cov}(y_t, \varepsilon_j)\,S_j^{-1} & \text{if } j < t, \\
    {\bf I} & \text{if } j = t, \\
    {\bf 0 } & \text{if } j > t.
    \end{cases}
$$
{{< /math >}}

Using $(\text{BLUP.1})$ and $(\text{I.2})$, we see that the BLUP of the signal $f_t$ given $y_{1:j}$
is given by
{{< math >}}
$$
\begin{aligned}
\tag{BLUP.2}
    f_{t|j}
    &= {\rm Cov}(f_t, y_{1:j})\,{\rm Var}(y_{1:j})^{-1}y_{1:j}\\
    &= {\rm Cov}(f_t, {\bf L}\,\varepsilon_{1:j})\,{\rm Var}({\bf L}\,\varepsilon_{1:j})^{-1}\,{\bf L}\,\varepsilon_{1:j}\\
    &= \sum_{k=1}^j {\rm Cov}(f_t, \varepsilon_k)\,S_k^{-1}\,\varepsilon_k.
\end{aligned}
$$
{{< /math >}}

And the error variance-covariance matrix of the BLUP takes the form
{{< math >}}
$$
\begin{aligned}
    S_{t|j} = {\rm Var}(f_t) - \sum_{k=1}^j{\rm Cov}(f_t, \varepsilon_k)\,S_k^{-1}\,{\rm Cov}(\varepsilon_k, f_t).
\end{aligned}
$$
{{< /math >}}

Equation $(\text{BLUP.2})$ above highlights a key property when working with innovations in estimating the BLUP:
the number of computations to estimate $f_{t|j}$ becomes *linear* in time.

# Filtering, prediction, smoothing, and fixed-lag smoothing
Armed with $(\text{BLUP.2})$, we define some important quantities of interest when dealing with signal plus noise models.

## Filtering
The term *filtering* refers to the action of filtering-out the noise $e_t$ to estimate the signal $f_t$.
This is defined as
{{< math >}}
$$
\tag{F.1}
    f_{t | t} = \sum_{k=1}^t {\rm Cov}(f_t, \varepsilon_k)\,S_k^{-1}\,\varepsilon_k.
$$
{{< /math >}}
This quantity is one of the most important in the signal-processing theory.


## Smoothing
This quantity refers to the estimate of $f_t$, with $t < T$, having observed $y_{1:T}$.
Contrary to the filtering equation $(\text{F.1})$, which is *online*, the smoothing operation
waits until all measurements have been observed to make an estimate of the signal.
{{< math >}}
$$
\tag{F.2}
    f_{t | T} = \sum_{k=1}^T {\rm Cov}(f_t, \varepsilon_k)\,S_k^{-1}\,\varepsilon_k.
$$
{{< /math >}}


## Fixed-lag smoothing
This quantity is a middle ground between the filtering, which is *online* and smoothing, which is *offline*.
The idea behind an $i$-step fixed-lag smoother is to estimate the signal $f_t$ after observing $y_{1:t+i}$.
In this sense, we only have to wait $i$ steps before making an estimate of the signal $f_t$.
{{< math >}}
$$
\tag{F.3}
    f_{t | t + i} = \sum_{k=1}^{t+i} {\rm Cov}(f_t, \varepsilon_k)\,S_k^{-1}\,\varepsilon_k.
$$
{{< /math >}}


## Prediction
Finally, this quantity estimates the expected future signal $f_{t+i}$, given $y_{1:t}$.
Here, $i \geq 1.$
{{< math >}}
$$
\tag{F.4}
    f_{t + i |t} = \sum_{k=1}^{t} {\rm Cov}(f_{t+1}, \varepsilon_k)\,S_k^{-1}\,\varepsilon_k.
$$ 
{{< /math >}}

# A note on the signal-plus-noise model
Note that the BLUP terms $(\text{F.1 -- F.4})$ depend on the quantities
${\rm Cov}(f_t, \varepsilon_k)$ and
$S_k = {\rm Var}(\varepsilon_k)$
for {{< math >}}$j,k \in \{1, \ldots, T\} = {\cal T}$.{{< /math >}}
In this sense, when dealing with signal-plus-noise models,
we assume that the experiments we run have length of size $T$.
Because of this, we have access to multiple runs of the process,
which allows us to approximate $(\text{BLUP.2})$ and $(\rm{I.2})$.
We show an example of this idea below.

# Example: a data-driven BLUP
In this example, we provide a data-driven approach to estimate the BLUP at multiple timesteps.
We assume we have access to a *train phase* where the quantities
${\rm Cov}(f_t, \varepsilon_k)$ and $S_k$ are found;
and a test-phase where BLUP estimates
$(\text{F.1 -- F.4})$
are obtained given an unseen run.