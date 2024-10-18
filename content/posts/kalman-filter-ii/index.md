---
title: "Notes on the Kalman filter (II): state-space models"
date: 2024-10-15
description: "Primer on the Kalman filter primer Pt. 2"
katex: true
draft: true
tags: [kalman-filter, summary, kf-notes]
tldr: "The Kalman filter from a best linear unbiased predictor"
---

This is the second part of the series "Notes on the Kalman filter".
Here, we go beyond data-driven best linear unbiased predictors.

# A recap --- signal plus noise models and BLUPs
We consider signal plus noise models (SPMs)
{{< math >}}
$$
    Y_{1:T} = F_{1:T} + E_{1:T}.
$$
{{< /math >}}
where
$Y_{1:T} = (Y_1, \ldots Y_T)$ is the measurement process,
$F_{1:T}$ is the signal process, and
$E_{1:T}$ is the noise process.



## The best linear unbiased predictor (BLUP)
Let
{{< math >}}${\cal T} = \{1, \ldots, T\}${{< /math >}}.
We seek the best linear estimate for $F_t$ given $Y_{1:j}$,
for $i \in {\cal T}$, $j \in {\cal T}$.

Let
{{< math >}}
$$
\tag{PM.1}
    {\bf A}_\text{opt}
    = \argmin_{\bf A}\mathbb{E}\left[\|F_t - {\bf A}\,Y_{1:j}\|^2\right]
    = {\rm Cov}(F_t, Y_{1:j})\,{\rm Var}(Y_{1:j})^{-1}
$$
{{< /math >}}
be the best unbiased linear mapping from measurements $Y_{1:j}$ to the signal $F_t$.

Then, the best linear unbiased predictor (BLUP) for the signal $F_t$ given $Y_{1:j}$
is
{{< math >}}
$$
\begin{aligned}
\tag{BLUP.1}
    F_{t|j}
    &= {\bf A}_\text{opt}\,Y_{1:j} \\
    &= {\rm Cov}(F_t, Y_{1:j})\,{\rm Var}(Y_{1:j})^{-1}\,Y_{1:j}.
\end{aligned}
$$
{{< /math >}}
Furthermore, the error variance-covariance matrix of the BLUP is defined by
{{< math >}}
$$
\tag{EVC.1}
    \Sigma_{t|j} = {\rm Var}(F_t - F_{t|j}) =
    {\rm Var}(F_t) - {\bf A}_\text{opt}\,{\rm Var}(Y_{1:j})\,{\bf A}_\text{opt}^\intercal.
$$
{{< /math >}}

## Innovations
Let 
{{< math >}}
$$
\tag{I.1}
    {\cal E}_t =
    \begin{cases}
        Y_1 & \text{for } t = 1,\\
        Y_t - \sum_{k=1}^{t-1} {\rm Cov}(Y_t, {\cal E}_k)\,{S}_k^{-1}\,{\cal E}_k & \text{for } t \geq 2,
    \end{cases}
$$
{{< /math >}}
with $S_j = {\rm Var}({\cal E}_j)$ for all $j \in {\cal T}$.

Then,
{{< math >}}
$$
\tag{I.2}
    Y_{1:T} = {\bf L}\,{\cal E}_{1:T},
$$
{{< /math >}}
with
{{< math >}}
$$
\tag{I.3}
    {\bf L}_{t,j} =
    \begin{cases}
    {\rm Cov}(Y_t, {\cal E}_j)\,S_j^{-1} & \text{if } j < t, \\
    {\bf I} & \text{if } j = t, \\
    {\bf 0 } & \text{if } j > t.
    \end{cases}
$$
{{< /math >}}

Furthermore,
{{< math >}}
$$
\tag{I.4}
    {\rm Var}({\cal E}_{1:T}) = {\rm diag}(S_1, \ldots, S_T).
$$
{{< /math >}}


## BLUP under innovations
The BLUP of the signal $F_t$ given the measurement process
{{< math >}}$Y_{1:j}${{< /math >}}
can be written
as a linear combinations of gain matrices and innovations:
{{< math >}}
$$
    \tag{BLUP.2}
    F_{t|j} = \sum_{k=1}^j \hat{\bf K}_{t,k}\,{\cal E}_k.
$$
{{< /math >}}
Furthermore, the error-variance covariance matrix of the BLUP takes the form
{{< math >}}
$$
\begin{aligned}
    \tag{EVC.2}
    {\rm Var}(F_t - F_{t|j})
    = {\rm Var}(F_t) - \sum_{k=1}^j \hat{\bf K}_{t,k}\,S_k\,\hat{\bf K}_{t,k}^\intercal.
\end{aligned}
$$
{{< /math >}}
Here,
{{< math >}}
$$
\tag{G.1}
    \hat{\bf K}_{t,k} = {\rm Cov}(F_t, {\cal E}_k)\,S_k^{-1}  
$$
{{< /math >}}
is the *gain* matrix of the signal $F_t$ from the innovation ${\cal E}_k$.

---

# The state-space assumption
As we saw in the previous post, the BLUP estimates above work well when we have
multiple *simulations* of the SPN process.
However, in some cases, we have a single run of the process.
In some other cases, we have a notion of the evolution of the system.
This is the case, for instance, in physical systems such as weather forecasting models.

The state-space model (SSM) assumption is an inductive bias on the SPN model over the evolution
of the signal process.
For a SPM that follows an SSM,
we decompose the signal $F_t$ as the product of a projection matrix ${\bf H}_t \in {\mathbb R}^{d\times m}$
times a latent variable $X_t \in {\mathbb R}^m$, i.e.,
$F_t = {\bf H}_t\,X_t$.
The measurement gets written as
$$
    Y_t = {\bf H}_t\,X_t + E_t.
$$
The latent process $X_t$ is taken to be time-dependent and evolving according to the dynamics
{{< math >}}
$$
    X_t = {\bf F}_t\,{X}_{t-1} + U_t.
$$
Here
${\bf F}_t$ is an $m\times m$ matrix, and
$U_t$ is a zero-mean random variable with ${\rm Var}(U_t) = Q_t$.

{{< /math >}}
The SPM under the SSM assumption at time $t$ is written as
{{< math >}}
$$
\tag{SSM.1}
\begin{aligned}
    X_t &= {\bf F}_t\,{X}_{t-1} + U_t,\\
    Y_t &= {\bf H}_t\,X_t + E_t.
\end{aligned}
$$
{{< /math >}}
Furthermore, we have the following list of assumptions about the SPN with SSM representation

1. ${\rm Cov}(E_t, E_s) = 0$ for $t\neq s$,
2. ${\rm Var}(E_t) = R_t$ for $t \in {\cal T}$ --- known positive definite matrices,
3. ${\rm Cov}(U_s, U_t) = 0$ for $s \neq t$,
4. ${\rm Var}(U_t) = Q_t$ for $t \in {\cal T}$ --- known positive semi-definite matrices,
5. ${\rm Var}(X_0) = \Sigma_{0|0}$ --- known and positive definite,
6. ${\rm Cov}(E_t, U_s) = 0$,
7. ${\rm Cov}(E_t, X_0) = 0$,
8. ${\rm Cov}(U_t, X_0) = 0$,
9. ${\bf H}_t$ --- known, and
10. ${\bf F}_t$ --- known.

Conditions (6.-10.) hold for $t \in {\cal T}$. 

The algorithms and propositions we put forth below assume that (1-10) hold.

## The BLUP under the SSM
In some cases, we are interested in finding $X_{t|j}$.
### proposition 1
Consider the innovation process
{{< math >}}${\cal E}_{1:j}${{< /math >}}
derived from the measurement process $Y_{1:j}$
for some $j \in {\cal T}$.
Let
{{< math >}}
$$
\tag{G.2}
    {\bf K}_{t,k} = {\rm Cov}(X_t, {\cal E}_k)\,S_k^{-1}  
$$
{{< /math >}}
be the *gain* matrix of the latent state $X_t$ from the innovation ${\cal E}_k$.
The BLUP of the latent variable $X_t$ given the measurement process
{{< math >}}$Y_{1:j}${{< /math >}}
is the expressed as the following linear combinations of gain matrices and innovations:
{{< math >}}
$$
    \tag{BLUP.3}
    X_{t|j} = \sum_{k=1}^j {\bf K}_{t,k}\,{\cal E}_k.
$$
{{< /math >}}
Next, error-variance covariance matrix of the BLUP takes the form
{{< math >}}
$$
\begin{aligned}
    \tag{EVC.3}
    \Sigma_{t|j} := {\rm Var}(X_t - X_{t|j})
    = {\rm Var}(X_t) - \sum_{k=1}^j {\bf K}_{t,k}\,S_k\,{\bf K}_{t,k}^\intercal.
\end{aligned}
$$
{{< /math >}}

Finally, the BLUP and the error variance-covariance matrix of the signal $F_t$ given ${\cal E}_{1:j}$ can be written in terms of $X_t$
as
{{< math >}}
$$
    F_{t|j} = {\bf F}_t \sum_{k=1}^j {\bf K}_{t,k}{\cal E}_k = {\bf F}_t\,X_{t|j},
$$
{{< /math >}}
for the signal; and
{{< math >}}
$$
    {\rm Var}(F_t - F_{t|j}) =
    {\bf F}_t\,\Sigma_{t|j}\,{\bf F}_t^\intercal
$$
{{< /math >}}
for the error-variance covariance matrix.

See [proof 1]({{<ref "#proof-of-proposition-1" >}}) in the Appendix for a proof.

## Properties of the latent signal
* (F1) ${\rm Cov}({\cal E}_t, {\cal E}_j) = 0$ for all $t \neq s$ and ${\rm Cov}({\cal E}_t, E_j) = 0$ for $j > t$.
* (F2) ${\rm Cov}(E_t, X_s) = 0$ for all $s \in {\cal T}$.
* (F3) ${\rm Cov}(U_t, X_s) = {\rm Cov}(U_t, Y_s) = {\rm Cov}(U_t, {\cal E}_s) = 0$ for $s \leq t$.

## Results under a state-space model

### Proposition 2.1
{{< math >}}
$$
    {\cal E}_t = {\bf H}_t\,(X_t - X_{t|j}) + E_t
$$
{{< /math >}}

### Proposition 2.2
{{< math >}}
$$
    \Sigma_{t|t} = \Sigma_{t|t-1}\,\left({\bf I}  - {\bf H}_t^\intercal\, S_t^{-1}\,{\bf H}_t\,\Sigma_{t|t-1}\right)
$$
{{< /math >}}
### Proposition 2.3
Let $\Sigma_{1|0} = {\rm Var}(X_1)$. Then, for $t \in {\cal T}$,
{{< math >}}
$$
    \Sigma_{t|t-1} = {\bf F}_t\,\Sigma_{t|t-1}\,{\bf F}_t^\intercal + Q_{t-1}.
$$
{{< /math >}}

### Proposition 2.4
For $t \in {\cal T}$,
{{< math >}}
$$
    S_t = {\bf H}_t\,\Sigma_{t|t-1}\,{\bf H}_t^\intercal + W_t
$$
{{< /math >}}

## The error-variance covariance matrix
Propositions 2.1 - 2.4 are enough to determine the error-covariance matrix of the system at time $t$.

### Pseudocode
* {{< math >}}$\Sigma_{1|0} = {\bf F}_0\,\Sigma_{0|0}{\bf F}_0 + Q_0${{< /math >}}  `#` {{< math >}}${\rm Var}(X_1)${{< /math >}}
* {{< math >}}$S_1 = {bf H}_1\,\Sigma_{1|0}{\bf H}_t^\intercal + R_1${{< /math >}}  `#` {{< math >}}${\rm Var}(S)${{< /math >}}

{{< math >}}
$$
    
$$
{{< /math >}}

---

# Appendix
## Proof of proposition 1
The terms $(\text{BLUP.3})$ and $(\text{EVC.3})$ follow by direct result
of replacing $F_t$ for $X_t$ in $(\text{PM.1})$.
Next, 