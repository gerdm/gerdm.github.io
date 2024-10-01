---
title: "Notes on the Kalman filter (I): signal plus noise models"
date: 2024-08-04
description: "Primer on the Kalman filter primer."
katex: true
draft: true
tags: [kalman-filter, summary]
tldr: "An introduction to signal-plus-noise models and their best linear unbiased estimates."
---

# Introduction

Throughout this post, we denote random variables in capital letters $X$ and
an in lower-case $x$ a sample of the random variable.
We defer proofs of the propositions to the appendix.

# Signal plus noise models
The story of the Kalman filter begins with signal plus noise models.
A signal plus noise model assumes that a random process
{{< math >}}$Y_{1:T} = (Y_1, \ldots, Y_T)${{< /math >}},
can be written as the sum of two components:
a predictable process
{{< math >}}$F_{1:T}${{< /math >}}
and an unpredictable process
{{< math >}}${E}_{1:T}${{< /math >}}.
We assume $(Y_t, F_t, E_t) \in {\mathbb R}^d$ for all
{{< math >}}$t \in {\cal T} = \{1, \ldots, T\}${{< /math >}}
and $d \geq 1$.

We write a signal-plus-noise process as
{{< math >}}
$$
\tag{S.1}
\underbrace{Y_{1:T}}_\text{measurement} =
    \underbrace{F_{1:T}}_{\text{signal}} + 
    \underbrace{{E}_{1:T}}_{\text{noise}}.
$$
{{< /math >}}
By _predictable_, we mean that the that the covariance between the measurement and the signal is (not necessarily) non-diagonal.
By _unpredictable_, we mean that the covariance between the measurement and the noise is diagonal.
This means

{{< math >}}
$$
\begin{aligned}
    {\rm Cov}(Y_t, F_j) &\neq 0, \\
    {\rm Cov}(Y_t, E_j) &= \mathbb{1}(t = j)\,{\bf R}_t,
\end{aligned}
$$
{{< /math >}}
for all {{< math >}}$t,j \in {\cal T}${{< /math >}}.
Here ${\bf R}_t$ is the covariance matrix of the noise at time $t$, which
by definition, is positive definite.
Finally, suppose
{{< math >}}$\mathbb{E}[F_t] = 0${{< /math >}} and
{{< math >}}$\mathbb{E}[{E}_t] = 0${{< /math >}}
for all $t \in {\cal T}$.

Throughout the post, we denote by $Y_{1:t}$ the random process and $y_{1:t}$ a sample of this process.

# The best-linear unbiased predictor (BLUP)
Suppose, $j,t\in{\cal T}$.
We seek to find the matrix ${\bf A} \in \reals^{d\times j}$ that maps the measurement process $Y_{1:j}$
to the signal process $F_t$ so that
{{< math >}}
$$
    F_t \approx {\bf A}\,Y_{1:j}.
$$
{{< /math >}}
That is, we want the matrix ${\bf A}$ that weights all observations up to time $j$.
Depending on the value of $j$, this estimate takes different names.
We come back to this point below.
Our notion of approximation is based on the matrix ${\bf A}$ that
minimises the expected L2 error
between the signal $F_t$ and the linear predictor
{{< math >}}${\bf A}\,Y_{1:j}${{< /math >}}.

We formalise this in the following proposition.
### Proposition 1
Let $Y_{1:j}$ be a random vector of measurements and $F_t$ signal random variable.
The linear mapping ${\bf A}$ that minimises the L2 error between the signal and the measurement takes the form
{{< math >}}
$$
    {\bf A}_\text{opt}
    = \argmin_{\bf A}\mathbb{E}\left[\|F_t - {\bf A}\,Y_{1:j}\|^2\right]
    = {\rm Cov}(F_t, Y_{1:j})\,{\rm Var}(Y_{1:j})^{-1}.
$$
{{< /math >}}
As a consequence, the best linear unbiased predictor (BLUP) of the signal at time $t$,
having access to a sample $y_{1:j}$, is
{{< math >}}
$$
\begin{aligned}
\tag{BLUP.1}
    f_{t|j}
    &= {\bf A}_\text{opt}\,y_{1:j} \\
    &= {\rm Cov}(F_t, Y_{1:j})\,{\rm Var}(Y_{1:j})^{-1}\,y_{1:j}.
\end{aligned}
$$
{{< /math >}}

Furthermore,
the error variance-covariance matrix of the BLUP is
{{< math >}}
$$
\tag{EVC.1}
    \Sigma_{t|j} = {\rm Var}(F_t - f_{t|j}) = {\rm Var}(F_t) - {\bf A}_\text{opt}\,{\rm Var}(F_{1:j})\,{\bf A}_\text{opt}^\intercal.
$$
{{< /math >}}



# Introduction of innovations
Computing ${\rm (BLUP.1)}$ requires a cubic amount of operations as a function of time $j$;
this is because of the term ${\rm Var}(Y_{1:j})^{-1}$,
which is a $j\times j$ positive definite matrix that we have to invert.

To go around this computational bottleneck, we introduce the concept of an innovation.
Here, we assume that $d \ll j$.
We denote by ${\cal E}_t$ an innovation random variable and $\varepsilon_t$ a sample of the random variable.
{{< math >}}
$$
\tag{I.1}
    \varepsilon_t =
    \begin{cases}
        y_1 & \text{for } t = 1,\\
        y_t - \sum_{k=1}^{t-1} {\rm Cov}(Y_t, {\cal E}_k)\,{S}_k^{-1}\,\varepsilon_k & \text{for } t \geq 2,
    \end{cases}
$$
{{< /math >}}
with $S_k = {\rm Var}({\cal E}_k)$.

One of the main advantages of the innovation process is that they are decorrelated, i.e.,
{{< math >}}
$$
    {\rm Cov}({\cal E}_t, {\cal E}_j)=
    \begin{cases}
    0 & \text{if } t\neq j,\\
    S_t & \text{if } t = j.
    \end{cases}
$$
{{< /math >}}
As we will see, this property of the innovations will allows us to compute $(\text{BLUP.1})$
in $O(j d^3)$ operations.
We formalise this in the following proposition

### Proposition 2
Let $Y_{1:j}$ be a signal-plus-noise random process and
{{< math >}}${\cal E}_{1:j}${{< /math >}} be the innovation process derived from $Y_{1:j}$.
Then,
{{< math >}}${\rm Cov}({\cal E}_t, {\cal E}_k) = 0${{< /math >}} for all $t \neq k$
and
{{< math >}}
$$
    {\rm Var}({\cal E}_{1:t}) = {\rm diag}(S_1, \ldots, S_t).
$$
{{< /math >}}

Furthermore, the innovation process and the measurement process satisfy
{{< math >}}
$$
\tag{I.2}
    Y_{1:j} = {\bf L}\,{\cal E}_{1:j},
$$
{{< /math >}}
with ${\bf L}$ a lower-triangular matrix with elements
{{< math >}}
$$
    {\bf L}_{t,j} =
    \begin{cases}
    {\rm Cov}(Y_t, {\cal E}_j)\,S_j^{-1} & \text{if } j < t, \\
    {\bf I} & \text{if } j = t, \\
    {\bf 0 } & \text{if } j > t.
    \end{cases}
$$
{{< /math >}}

# The BLUP under innovations

## Proposition 3
Suppose we have access to innovations $\varepsilon_{1:j}$ derived from measurements $y_{1:j}$ for some $j\in{\cal T}$.
Let
{{< math >}}
$$
\tag{G.1}
    {\bf K}_{t,k} = {\rm Cov}(F_t, {\cal E}_k)\,S_k^{-1}  
$$
{{< /math >}}
be the *gain* matrix for the signal $F_t$, given the innovation ${\cal E}_k$.

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


Equation $(\text{BLUP.2})$ highlights a key property when working with innovations in estimating the BLUP:
the number of computations to estimate $f_{t|j}$ becomes *linear* in time.

# Filtering, prediction, smoothing, and fixed-lag smoothing
Recall that our quantity of interest takes the form $(\text{BLUP.2})$.
Depending on our frame of reference, which depends on the choice of $j$,
and hence how much information we have to make an estimate of the signal process,
we can come up with different BLUP estimates for the unknown signal $f_t$.

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
Given a run of the experiment ran up to time $t$, filtering produces the best estimate of the signal,
given the measurements $y_{1:t}$.

We exemplify filtering in the table below.
The top row shows the point in time in which the estimate is computed;
the *signal* row shows in $\color{#00B7EB}\text{cyan}$ the target signal that we seek to obtain from the measurement;
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
There is a more computationally-efficient way to estimate ($F.3$),
which we will see in a later post.


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


---

# Appendix

### Proof of proposition 1
Let
{{< math >}}
$$
    {\cal L}({\bf A}) = \mathbb{E}\left[\|F_t - {\bf A}\,Y_{1:j}\|^2\right].
$$
{{< /math >}}
Then,
{{< math >}}
$$
\begin{aligned}
    \nabla_{\bf A}\,{\cal L}({\bf A})
    &= 2\,\mathbb{E}\left[(F_t - {\bf A}\,Y_{1:j})\,Y_{1:j}^\intercal\right]\\
    &= 2\,\left( \mathbb{E}\left[F_t\,Y_{1:j}^\intercal\right]-\mathbb{E}[{\bf A}\,Y_{1:j}\,Y_{1:j}^\intercal]\right)\\
    &= 2\,\left( \mathbb{E}\left[F_t\,Y_{1:j}^\intercal\right]-{\bf A}\,\mathbb{E}[Y_{1:j}\,Y_{1:j}^\intercal]\right)\\
    &= 2\,\left( {\rm Cov} (F_t, Y_{1:j}) - {\bf A}\,{\rm Var}(Y_{1:j})\right)
\end{aligned}
$$
{{< /math >}}
Setting this last equality to zero recovers ${\bf A}_\text{opt}$ above.
{{< math >}} $$ \ \tag*{$\blacksquare$} $$ {{< /math >}}

### Proof of proposition 2


### Proof of proposition 3
Using $(\text{BLUP.1})$ and $(\text{I.2})$, we see that
{{< math >}}
$$
\begin{aligned}
    {\bf A}_\text{opt}
    &= {\rm Cov}(F_t, Y_{1:j})\,{\rm Var}(y_{1:j})^{-1}\\
    &= {\rm Cov}(F_t, {\bf L}\,{\cal E}_{1:j})\,{\rm Var}({\bf L}\,{\cal E}_{1:j})^{-1}\\
    &= {\rm Cov}(F_t, {\cal E}_{1:j})\,{\bf L}^\intercal\,\{{\bf L}{\rm Var}({\cal E}_{1:j}){\bf L}^\intercal\}^{-1}\\
    &= {\rm Cov}(F_t, {\cal E}_{1:j})\,{\bf L}^\intercal\,{\bf L}^{-\intercal}\, {\rm Var}({\cal E}_{1:j})^{-1}{\bf L}^{-1}\\
    &= {\rm Cov}(F_t, {\cal E}_{1:j})\, {\rm Var}({\cal E}_{1:j})^{-1}{\bf L}^{-1}\\
    &= {\rm Cov}(F_t, {\cal E}_{1:j})\, {\rm Diag}(S_1, \ldots, S_j)^{-1}{\bf L}^{-1}.
\end{aligned}
$$
{{< /math >}}
Then, the BLUP of the signal $f_t$ given $y_{1:j}$ is
{{< math >}}
$$
\begin{aligned}
    f_{t|j}
    &= {\bf A}_\text{opt}\,y_{1:j}\\
    &= {\rm Cov}(F_t, Y_{1:j})\,{\rm Var}(Y_{1:j})^{-1}\,y_{1:j}\\
    &= {\rm Cov}(F_t, {\cal E}_{1:j})\, {\rm Diag}(S_1, \ldots, S_j)^{-1}{\bf L}^{-1}\,{\bf L}\,\varepsilon_{1:j}\\
    &= {\rm Cov}(F_t, {\cal E}_{1:j})\, {\rm Diag}(S_1, \ldots, S_j)^{-1}\,\varepsilon_{1:j}\\
    &= \sum_{k=1}^j {\rm Cov}(F_t, {\cal E}_k)\,S_k^{-1}\,\varepsilon_k\\
    &= \sum_{k=1}^j {\bf K}_{t,k}\,\varepsilon_k.
\end{aligned}
$$
{{< /math >}}

Furthermore, the error variance-covariance matrix of the BLUP takes the form
{{< math >}}
$$
\begin{aligned}
    &{\rm Var}(F_t - f_{t|j})\\
    &= {\rm Var}(F_t) - {\bf A}_\text{opt}\,{\rm Var}\,(Y_{1:j})\,{\bf A}_\text{opt}^\intercal\\
    &= {\rm Var}(F_t) -
    {\rm Cov}(F_t, {\cal E}_{1:j})\, {\rm Var}({\cal E}_{1:t})^{-1}{\bf L}^{-1}
    \{{\bf L}{\rm Var}({\cal E}_{1:j}){\bf L}^\intercal\}
    \left({\rm Cov}(F_t, {\cal E}_{1:j})\, {\rm Var}({\cal E}_{1:t})^{-1}{\bf L}^{-1}\right)^\intercal\\
    &= {\rm Var}(F_t) -
    {\rm Cov}(F_t, {\cal E}_{1:j})\, {\rm Var}({\cal E}_{1:t})^{-1}{\bf L}^{-1}
    {\bf L}{\rm Var}({\cal E}_{1:j}){\bf L}^\intercal
    \left({\rm Cov}(F_t, {\cal E}_{1:j})\, {\rm Var}({\cal E}_{1:t})^{-1}{\bf L}^{-1}\right)^\intercal\\
    &= {\rm Var}(F_t) -
    {\rm Cov}(F_t, {\cal E}_{1:j})\, {\rm Var}({\cal E}_{1:t})^{-1}{\bf L}^{-1}
    {\bf L}{\rm Var}({\cal E}_{1:j}){\bf L}^\intercal\,
    {\bf L}^{-\intercal}\,{\rm Var}({\cal E}_{1:t})^{-1}\,{\rm Cov}({\cal E}_{1:j}, F_t)\\
    &= {\rm Var}(F_t) -
    {\rm Cov}(F_t, {\cal E}_{1:j})\, {\rm Var}({\cal E}_{1:t})^{-1}{\rm Var}({\cal E}_{1:j})\,
    {\rm Var}({\cal E}_{1:t})^{-1}\,{\rm Cov}({\cal E}_{1:j}, F_t)\\
    &= {\rm Var}(F_t) -
    {\rm Cov}(F_t, {\cal E}_{1:j})\, {\rm diag}(S_1, \ldots, S_j)^{-1}
    {\rm diag}(S_1, \ldots, S_j)\,
    {\rm diag}(S_1, \ldots, S_j)^{-1}\,{\rm Cov}({\cal E}_{1:j}, F_t)\\
    &= {\rm Var}(F_t) - \sum_{k=1}^j{\rm Cov}(F_t, {\cal E}_k)\,S_k^{-1}S_k\,S_k^{-1}\,{\rm Cov}({\cal E}_k, F_t)\\
    &= {\rm Var}(F_t) - \sum_{k=1}^j \left({\rm Cov}(F_t, {\cal E}_k)\,S_k^{-1}\right)S_k\,\left({\rm Cov}(F_t, {\cal E}_k)\,S_k^{-1}\right)^\intercal\\
    &= {\rm Var}(F_t) - \sum_{k=1}^j {\bf K}_{t,k}\,S_k\,{\bf K}_{t,k}^\intercal.
\end{aligned}
$$
{{< /math >}}
{{< math >}} $$ \ \tag*{$\blacksquare$} $$ {{< /math >}}