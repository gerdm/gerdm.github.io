---
title: "Notes on the Kalman filter (I): signal plus noise models"
date: 2024-08-04
description: "Primer on the Kalman filter primer."
katex: true
draft: true
tags: [kalman-filter, summary]
tldr: "An introduction to signal-plus-noise models and their best linear unbiased estimates."
---

# Signal plus noise models
The story of the Kalman filter begins with signal plus noise models.
A signal plus noise model assumes that a sequence of observations
{{< math >}}$y_{1:T} = (y_1, \ldots, y_T)${{< /math >}}
can be written as the sum of two components:
a predictable component
{{< math >}}$f_{1:T}${{< /math >}}
and an unpredictable component
{{< math >}}${e}_{1:T}${{< /math >}}.
We assume $(y_t, f_t, e_t) \in {\mathbb R}^d$ for all
{{< math >}}$t \in {\cal T} = \{1, \ldots, T\}${{< /math >}}
and $d \geq 1$.

We write a signal-plus-noise process as
{{< math >}}
$$
\tag{S.1}
\underbrace{y_{1:T}}_\text{measurement} =
    \underbrace{f_{1:T}}_{\text{signal}} + 
    \underbrace{{e}_{1:T}}_{\text{noise}}.
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
for all {{< math >}}$t,j \in {\cal T}${{< /math >}}.
Here ${\bf R}_t$ is the covariance matrix of the noise at time $t$, which
by definition, is positive definite.
Finally, suppose
{{< math >}}$\mathbb{E}[f_t] = 0${{< /math >}} and
{{< math >}}$\mathbb{E}[{e}_t] = 0${{< /math >}}
for all $t \in {\cal T}$.

In this post, we assume that we have access to multiple trials of $(\text{S.1})$,
each lasting $T$ steps which produces samples $y_{1:T}$.

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
Computing ${\rm (BLUP.1)}$ requires a cubic amount of operations as a function of time $j$;
this is because of the term ${\rm Var}(y_{1:j})^{-1}$,
which is an $j\times j$ positive definite matrix that we have to invert.

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

# The BLUP under innovations

## Proposition
Suppose we have access to the innovations $\varepsilon_{1:j}$ derived from measurements $y_{1:j}$ for some $j\in{\cal T}$.
Let
{{< math >}}
$$
\tag{G.1}
    {\bf K}_{t,k} = {\rm Cov}(f_t, \varepsilon_k)\,S_k^{-1}  
$$
{{< /math >}}
be the *gain* matrix for the signal at time $t$, given the innovation $\varepsilon_k$.

The BLUP of the signal $f_t$ given $\varepsilon_{1:j}$ can be written
as the sum of linear combinations of gain matrices and innovations:
{{< math >}}
$$
    \tag{BLUP.2}
    f_{t|j} = \sum_{k=1}^j {\bf K}_{t,k}\,\varepsilon_k.
$$
{{< /math >}}
Furthermore, the error-variance covariance matrix of the BLUP takes the form
{{< math >}}
$$
\begin{aligned}
    {\rm Var}(f_t - f_{t|j})
    = {\rm Var}(f_t) - \sum_{k=1}^j {\bf K}_{t,k}\,S_k\,{\bf K}_{t,k}^\intercal.
\end{aligned}
$$
{{< /math >}}

### Proof
Using $(\text{BLUP.1})$ and $(\text{I.2})$, we see that
{{< math >}}
$$
\begin{aligned}
    {\bf A}_\text{opt}
    &= {\rm Cov}(f_t, y_{1:j})\,{\rm Var}(y_{1:j})^{-1}\\
    &= {\rm Cov}(f_t, {\bf L}\,\varepsilon_{1:j})\,{\rm Var}({\bf L}\,\varepsilon_{1:j})^{-1}\\
    &= {\rm Cov}(f_t, \varepsilon_{1:j})\,{\bf L}^\intercal\,\{{\bf L}{\rm Var}(\varepsilon_{1:j}){\bf L}^\intercal\}^{-1}\\
    &= {\rm Cov}(f_t, \varepsilon_{1:j})\,{\bf L}^\intercal\,{\bf L}^{-\intercal}\, {\rm Var}(\varepsilon_{1:j})^{-1}{\bf L}^{-1}\\
    &= {\rm Cov}(f_t, \varepsilon_{1:j})\, {\rm Var}(\varepsilon_{1:j})^{-1}{\bf L}^{-1}\\
    &= {\rm Cov}(f_t, \varepsilon_{1:j})\, {\rm Diag}(S_1, \ldots, S_j)^{-1}{\bf L}^{-1}.
\end{aligned}
$$
{{< /math >}}
Then, the BLUP of the signal $f_t$ given $y_{1:j}$ is
{{< math >}}
$$
\begin{aligned}
    f_{t|j}
    &= {\bf A}_\text{opt}\,y_{1:j}\\
    &= {\rm Cov}(f_t, y_{1:j})\,{\rm Var}(y_{1:j})^{-1}\,y_{1:j}\\
    &= {\rm Cov}(f_t, \varepsilon_{1:j})\, {\rm Diag}(S_1, \ldots, S_j)^{-1}{\bf L}^{-1}\,{\bf L}\,\varepsilon_{1:j}\\
    &= {\rm Cov}(f_t, \varepsilon_{1:j})\, {\rm Diag}(S_1, \ldots, S_j)^{-1}\,\varepsilon_{1:j}\\
    &= \sum_{k=1}^j {\rm Cov}(f_t, \varepsilon_k)\,S_k^{-1}\,\varepsilon_k\\
    &= \sum_{k=1}^j {\bf K}_{t,k}\,\varepsilon_k.
\end{aligned}
$$
{{< /math >}}

Furthermore, the error variance-covariance matrix of the BLUP takes the form
{{< math >}}
$$
\begin{aligned}
    &{\rm Var}(f_t - f_{t|j})\\
    &= {\rm Var}(f_t) - {\bf A}_\text{opt}\,{\rm Var}\,(y_{1:j})\,{\bf A}_\text{opt}^\intercal\\
    &= {\rm Var}(f_t) -
    {\rm Cov}(f_t, \varepsilon_{1:j})\, {\rm Var}(\varepsilon_{1:t})^{-1}{\bf L}^{-1}
    \{{\bf L}{\rm Var}(\varepsilon_{1:j}){\bf L}^\intercal\}
    \left({\rm Cov}(f_t, \varepsilon_{1:j})\, {\rm Var}(\varepsilon_{1:t})^{-1}{\bf L}^{-1}\right)^\intercal\\
    &= {\rm Var}(f_t) -
    {\rm Cov}(f_t, \varepsilon_{1:j})\, {\rm Var}(\varepsilon_{1:t})^{-1}{\bf L}^{-1}
    {\bf L}{\rm Var}(\varepsilon_{1:j}){\bf L}^\intercal
    \left({\rm Cov}(f_t, \varepsilon_{1:j})\, {\rm Var}(\varepsilon_{1:t})^{-1}{\bf L}^{-1}\right)^\intercal\\
    &= {\rm Var}(f_t) -
    {\rm Cov}(f_t, \varepsilon_{1:j})\, {\rm Var}(\varepsilon_{1:t})^{-1}{\bf L}^{-1}
    {\bf L}{\rm Var}(\varepsilon_{1:j}){\bf L}^\intercal\,
    {\bf L}^{-\intercal}\,{\rm Var}(\varepsilon_{1:t})^{-1}\,{\rm Cov}(\varepsilon_{1:j}, f_t)\\
    &= {\rm Var}(f_t) -
    {\rm Cov}(f_t, \varepsilon_{1:j})\, {\rm Var}(\varepsilon_{1:t})^{-1}{\rm Var}(\varepsilon_{1:j})\,
    {\rm Var}(\varepsilon_{1:t})^{-1}\,{\rm Cov}(\varepsilon_{1:j}, f_t)\\
    &= {\rm Var}(f_t) -
    {\rm Cov}(f_t, \varepsilon_{1:j})\, {\rm diag}(S_1, \ldots, S_j)^{-1}
    {\rm diag}(S_1, \ldots, S_j)\,
    {\rm diag}(S_1, \ldots, S_j)^{-1}\,{\rm Cov}(\varepsilon_{1:j}, f_t)\\
    &= {\rm Var}(f_t) - \sum_{k=1}^j{\rm Cov}(f_t, \varepsilon_k)\,S_k^{-1}S_k\,S_k^{-1}\,{\rm Cov}(\varepsilon_k, f_t)\\
    &= {\rm Var}(f_t) - \sum_{k=1}^j \left({\rm Cov}(f_t, \varepsilon_k)\,S_k^{-1}\right)S_k\,\left({\rm Cov}(f_t, \varepsilon_k)\,S_k^{-1}\right)^\intercal\\
    &= {\rm Var}(f_t) - \sum_{k=1}^j {\bf K}_{t,k}\,S_k\,{\bf K}_{t,k}^\intercal.
\end{aligned}
$$
{{< /math >}}
{{< math >}} $$ \ \tag*{$\blacksquare$} $$ {{< /math >}}

Equation $(\text{BLUP.2})$ highlights a key property when working with innovations in estimating the BLUP:
the number of computations to estimate $f_{t|j}$ becomes *linear* in time.

# Filtering, prediction, smoothing, and fixed-lag smoothing
Depending on the choice of $j$, and armed with Arrmed with $(\text{BLUP.2})$,
we define some important quantities of interest when dealing with signal plus noise models.
The following quantities all seek to estimate the BLUP of the signal $f_t$.
However, they all consider different time-frames of reference.

## Filtering
The term *filtering* refers to the action of filtering-out the noise $e_t$ to estimate the signal $f_t$.
This is defined as
{{< math >}}
$$
\tag{F.1}
    % f_{t | t} = \sum_{k=1}^t {\rm Cov}(f_t, \varepsilon_k)\,S_k^{-1}\,\varepsilon_k.
    f_{t | t} = \sum_{k=1}^t {\bf K}_{t,k}\,\varepsilon_k.
$$
{{< /math >}}
This quantity is one of the most important in the signal-processing theory.
Given a run of the experiment ran up to time $t$, it produces the best estimate of the signal,
given the measurements $y_{1:t}$.

We exemplify filtering in the table below.
The top row shows the point in time to compute the estimate,
the *signal* row shows in $\color{#00B7EB}\text{cyan}$ the target signal;
finally the bottom row shows in $\color{orange}\text{orange}$ the measurements to consider to estimate the BLUP.

{{< math >}}
$$
\begin{array}{c|ccccc}
\text{BLUP} & & & & f_{t|t} & & & \\
\text{signal} & f_{t-3} & f_{t-2} & f_{t-1} & \color{#00B7EB}{f_{t}} & f_{t+1} & f_{t+2} & f_{t+3}\\
\text{measurement} & \color{orange}{y_{t-3}} & \color{orange}{y_{t-2}} & \color{orange}{y_{t-1}} & \color{orange}{y_{t}} & y_{t+1} & y_{t+2} & y_{t+3}\\
\hline
\text{time} & t-3 & t-2 & t-1 & t & t+1 & t+2 & t+3
\end{array}
$$
{{< /math >}}
As we see, filtering considers all measurements up to time $t$ to make an estimate of the signal at time $t$.

## Prediction
This quantity estimates the expected future signal $f_{t+i}$, given $y_{1:t}$.
Here, $i \geq 1.$
{{< math >}}
$$
\tag{F.2}
    % f_{t + i |t} = \sum_{k=1}^{t} {\rm Cov}(f_{t+1}, \varepsilon_k)\,S_k^{-1}\,\varepsilon_k.
    f_{t + i | t} = \sum_{k=1}^t {\bf K}_{t+i,k}\,\varepsilon_k.
$$ 
{{< /math >}}


We exemplify a two-step-ahead prediction in the table below
{{< math >}}
$$
\begin{array}{c|ccccc}
\text{BLUP} & & & & f_{t+2|t} & & & \\
\text{signal} & f_{t-3} & f_{t-2} & f_{t-1} & f_{t} & f_{t+1} & \color{#00B7EB}{f_{t+2}} & f_{t+3}\\
\text{measurement} & \color{orange}{y_{t-3}} & \color{orange}{y_{t-2}} & \color{orange}{y_{t-1}} &  \color{orange}{y_{t}} & y_{t+1} & y_{t+2} & y_{t+3}\\
\hline
\text{time} & t-3 & t-2 & t-1 & t & t+1 & t+2 & t+3
\end{array}
$$
{{< /math >}}
As shown in the table above, a two-step-ahead prediction at time $t$ considers all measurements at time $t$ to
make an estimate of the signal at time $t+2$.

## Smoothing
This quantity refers to the estimate of $f_t$, with $t < T$, having observed $y_{1:T}$.
Contrary to the filtering equation $(\text{F.1})$, which is *online*, the smoothing operation
waits until all measurements have been observed to make an estimate of the signal.
We define the smoothing operation as
{{< math >}}
$$
\tag{F.3}
    % f_{t | T} = \sum_{k=1}^T {\rm Cov}(f_t, \varepsilon_k)\,S_k^{-1}\,\varepsilon_k.
    f_{t | T} = \sum_{k=1}^T {\bf K}_{t,k}\,\varepsilon_k.
$$
{{< /math >}}

We exemplify smoothing at time $t$ in the table below
{{< math >}}
$$
\begin{array}{c|ccccc}
\text{BLUP} & & & & & & & f_{t|T} \\
\text{signal} & f_{t-3} & f_{t-2} & f_{t-1} & \color{#00B7EB}{f_{t}} & f_{t+1} & \ldots & f_T\\
\text{measurement} & \color{orange}{y_{t-3}} & \color{orange}{y_{t-2}} & \color{orange}{y_{t-1}}
                   & \color{orange}{y_{t}} & \color{orange}{y_{t+1}} & \ldots & \color{orange}{y_{T}}\\
\hline
\text{time} & t-3 & t-2 & t-1 & t & t+1 & \ldots & T
\end{array}
$$
{{< /math >}}
We observe that smoothing requires all information up to time $T$ to make an estimate of the signal at time $t$.
In this sense, smoothing is *offline*.


## Fixed-lag smoothing
This quantity is a middle ground between the filtering, which is *online*, and smoothing, which is *offline*.
The idea behind an $i$-step fixed-lag smoother is to estimate the signal $f_t$ after observing $y_{1:t+i}$.
In this sense, we only have to wait $i$ steps before making an estimate of the signal $f_t$.
{{< math >}}
$$
\tag{F.4}
    % f_{t | t + i} = \sum_{k=1}^{t+i} {\rm Cov}(f_t, \varepsilon_k)\,S_k^{-1}\,\varepsilon_k.
    f_{t | t + i} = \sum_{k=1}^{t+i} {\bf K}_{t,k}\,\varepsilon_k.
$$
{{< /math >}}

We exemplify a two-step fixed-lag smoothing below
{{< math >}}
$$
\begin{array}{c|ccccc}
\text{BLUP} & & & & & & f_{t|t+2} & \\
\text{signal} & f_{t-3} & f_{t-2} & f_{t-1} & \color{#00B7EB}{f_{t}} & f_{t+1} & f_{t+2} & f_{t+3}\\
\text{measurement} & \color{orange}{y_{t-3}} & \color{orange}{y_{t-2}} & \color{orange}{y_{t-1}} & \color{orange}{y_{t}}
                   & \color{orange}{y_{t+1}} & \color{orange}{y_{t+2}} & y_{t+3}\\
\hline
\text{time} & t-3 & t-2 & t-1 & t & t+1 & t+2 & t+3
\end{array}
$$
{{< /math >}}
We observe that we require information up to time $t+2$ to make a prediction of the signal at time $t$.

# Example: a data-driven BLUP
In this example, we provide a data-driven approach to estimate the BLUP at multiple timesteps.
We assume we have access to a *train phase* where the quantities
${\rm Cov}(f_t, \varepsilon_k)$, ${\rm Cov}(y_t, \varepsilon_k)$, and $S_k$ are found;
and a test-phase where BLUP estimates
$(\text{F.1 -- F.4})$
are obtained given an unseen run.