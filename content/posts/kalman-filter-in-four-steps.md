---
title: "Deriving the Kalman Filter in four steps"
date: 2023-02-22T10:23:00+05:30
description: "Primer on the Kalman filter primer."
katex: true
draft: false
tags: [kalman-filter, summary, derivation]
---

# The Kalman filter

Suppose $\forall t=1,\ldots,T.{\bf x}_{t}\in\mathbb{R}^n, {\bf y}_t\in\mathbb{R}^m$,

{{< math >}}
$$
\begin{aligned}
p({\bf x}_t \vert {\bf x}_{t-1}) &= {\cal N}({\bf x}_t \vert {\bf A}_{t-1}{\bf x}_{t-1}, {\bf Q}_{t-1})\\
p({\bf y}_t \vert {\bf x}_t) &= {\cal N}({\bf y}_t \vert {\bf H}_t{\bf x}_t, {\bf R}_t)
\end{aligned}
$$
{{< /math >}}

then, the **Kalman filter equations** are given by

{{< math >}}
$$
\begin{aligned}
p({\bf x}_t \vert {\bf y}_{1:t-1}) &= {\cal N}({\bf x}_t \vert \bar{\bf m}_t,\bar{\bf P}_t) &\quad \text{(Predict)}\\
p({\bf x}_t \vert {\bf y}_{1:t}) &= {\cal N}({\bf x}_t \vert {\bf m}_t,{\bf P}_t) &\quad \text{(Update)}
\end{aligned}
$$
{{< /math >}}

where the prediction-step equations are given by

{{< math >}}
$$
\begin{aligned}
\bar{\bf m}_t &= {\bf A}_{t-1}{\bf m}_{t-1}\\
\bar{\bf P}_{t-1} &= {\bf A}_{t-1}{\bf P}_{t-1}{\bf A}_{t-1}^\intercal + {\bf Q}_{t-1}
\end{aligned}
$$
{{< /math >}}

and the update-step equations are given by

{{< math >}}
$$
\begin{aligned}
{\bf e}_t &= {\bf y}_t - {\bf H}_t\bar{\bf m}_t\\
{\bf S}_t &= {\bf H}_t\bar{\bf P}_t{\bf H}_{t}^\intercal + {\bf R}_t\\
{\bf K}_t &= \bar{\bf P}_t{\bf H}_{t}^\intercal{\bf S}_t^{-1}\\ \\
{\bf m}_t &= \bar{\bf m}_t + {\bf K}_t{\bf e}_t\\
{\bf P}_t &= \bar{\bf P}_t - {\bf K}_t{\bf S}_t {\bf K}_t^\intercal
\end{aligned}
$$
{{< /math >}}

## Sketch of Proof

The sketch of the proof follows 4 steps:

1. Estimate {{< math >}}$p({\bf x}_t, {\bf x}_{t-1} \vert {\bf y}_{1:t-1})${{< /math >}}— Join
2. Estimate {{< math >}}$p({\bf x}_t \vert {\bf  y}_{1:t-1}) = {\cal N}({\bf x}_{t} \vert \bar{\bf m}_t, \bar{\bf P}_t)${{< /math >}} — Marginalise
3. Estimate {{< math >}}$p({\bf x}_t, {\bf y}_t \vert {\bf y}_{1:t-1})${{< /math >}} using 2  — Join
4. Estimate {{< math >}}$p({\bf x}_t \vert {\bf y}_{1:t}) = {\cal N}({\bf x}_t \vert {\bf m}_t, {\bf P}_t)${{< /math >}} — Condition

Steps 1 through 3 follow from Lemma A1. Step 4 follows from Lemma A2.

## Proof

### Step 1 — {{< math >}}$p({\bf x}_t, {\bf x}_{t-1} \vert {\bf y}_{1:t-1})${{< /math >}}

Using Lemma A1, we write

{{< math >}}
$$
\begin{aligned}
p({\bf x}_t, {\bf x}_{t-1} \vert {\bf y}_{1:t-1}) &= p({\bf x}_{t-1} \vert {\bf y}_{1:t-1})p({\bf x}_t \vert {\bf x}_{t-1})\\
&={\cal N}({\bf x}_{t-1} \vert {\bf m}_{t-1}, {\bf P}_{t-1})
p({\bf x}_{t}\vert{\bf A}_{t-1}{\bf x}_{t-1}, {\bf Q}_t)\\
&=
{\cal N}\left(
\begin{bmatrix}
{\bf m}_{t-1}\\
{\bf A}_{t-1}{\bf m}_{t-1}
\end{bmatrix},
\begin{bmatrix}
{\bf P}_{t-1} & {\bf P}_{t-1}{\bf A}_{t-1}^\intercal\\
{\bf A}_{t-1}{\bf P}_{t-1} &
{\bf A}_{t-1}{\bf P}_{t-1}{\bf A}_{t-1}^\intercal + {\bf Q}_{t-1}
\end{bmatrix}
\right)
\end{aligned}
$$
{{< /math >}}

### Step 2 — {{< math >}}$p({\bf x}_t \vert {\bf  y}_{1:t-1})${{< /math >}}

Using Lemma A1, we integrate ${\bf x}_{t-1}$ to obtain


{{< math >}}
$$
\begin{aligned}
p({\bf x}_{t} \vert {\bf y}_{1:t-1}) &= {\cal N}({\bf A}_{t-1}{\bf m}_{t-1}, {\bf A}_{t-1}{\bf P}_{t-1}{\bf A}_{t-1}^\intercal + {\bf Q}_{t-1})\\
&= p(\bar{\bf m}_{t-1}, \bar{\bf P}_{t-1})
\end{aligned}
$$
{{< /math >}}

where

- {{< math >}}$\bar{\bf m}_{t-1} = {\bf A}_{t-1}{\bf m}_{t-1}${{< /math >}}
- {{< math >}}$\bar{\bf P}_{t-1} = {\bf A}_{t-1}{\bf P}_{t-1}{\bf A}_{t-1}^\intercal + {\bf Q}_{t-1}${{< /math >}}

### Step 3 — {{< math >}}$p({\bf x}_t, {\bf y}_t \vert {\bf y}_{1:t-1})${{< /math >}}

Having {{< math >}}$p({\bf x}_t \vert {\bf y}_{1:t-1})${{< /math >}} and using Lemma A1 we obtain

{{< math >}}
$$
\begin{aligned}
p({\bf y}_t, {\bf x}_{t} \vert {\bf y}_{1:t}) &= p({\bf x}_t| {\bf y}_{1:t-1})p({\bf y}_t \vert {\bf x}_t)\\
&= {\cal N}({\bf x}_t \vert \bar{\bf m}_t, \bar{\bf P}_t){\cal N}({\bf y}_t \vert {\bf H}_t{\bf y}_t, {\bf R}_t)\\
&= {\cal N}\left(
\begin{bmatrix}
{\bf x}_t\\
{\bf y}_t
\end{bmatrix}
{\huge\vert}
\begin{bmatrix}
\bar{\bf m}_t\\
{\bf H}_t {\bf y}_t
\end{bmatrix},
\begin{bmatrix}
\bar{\bf P}_t & \bar{\bf P}_t{\bf H}_t^\intercal\\
{\bf H}_t\bar{\bf P}_t &
{\bf H}_t\bar{\bf P}_t{\bf H}_t^\intercal + {\bf R}_t
\end{bmatrix}
\right)
\end{aligned}
$$
{{< /math >}}

### Step 4 — {{< math >}}$p({\bf x}_t \vert {\bf y}_{1:t})${{< /math >}}

Having  {{< math >}}$p({\bf x}_t, {\bf y}_t \vert {\bf y}_{1:t-1})${{< /math >}} and using Lemma A2, we obtain

{{< math >}}
$$
\begin{aligned}
p({\bf x}_t \vert {\bf y}_{1:t}) &= {\cal N}({\bf x}_t \vert \bar{\bf m}_t + {\bf K}_t[{\bf y}_t - {\bf H}_t\bar{\bf m}_t], \bar{\bf P}_t - {\bf K}_t{\bf S}_t{\bf K}_t^\intercal)\\
&= {\cal N}({\bf x}_t \vert {\bf m}_t, {\bf P}_t)
\end{aligned}
$$
{{< /math >}}

where

- {{< math >}}${\bf S}_t = {\bf H}_t\bar{\bf P}_t{\bf H}_t^\intercal + {\bf R}_t${{< /math >}}
- {{< math >}}${\bf K}_t = \bar{\bf P}_t{\bf H}_t^\intercal{\bf S}_t^{-1}${{< /math >}}
- {{< math >}}${\bf m}_t = {\bf x}_t \vert \bar{\bf m}_t + {\bf K}_t[{\bf y}_t - {\bf H}_t\bar{\bf m}_t]${{< /math >}}
- {{< math >}}${\bf P}_t = \bar{\bf P}_t - {\bf K}_t{\bf S}_t{\bf K}_t^\intercal${{< /math >}}

---

# Lemmas

## Lemma A1

Suppose ${\bf x}\in\mathbb{R}^n$ and ${\bf y}\in\mathbb{R}^m$ are random variables such that

{{< math >}}
$$
\begin{aligned}
{\bf x}&\sim{\cal N}({\bf m}, {\bf P})\\
{\bf y}\vert{\bf x} &\sim{\cal N}({\bf Hx} + {\bf u}, {\bf R})
\end{aligned}
$$
{{< /math >}}

then, the joint distribution for $({\bf x}, {\bf y})$ is given by

{{< math >}}
$$
\begin{pmatrix}
{\bf x}\\
{\bf y}
\end{pmatrix}
\sim
{\cal N}\left(
\begin{bmatrix}
{\bf m}\\
{\bf Hm} + {\bf u}
\end{bmatrix},
\begin{bmatrix}
{\bf P} & {\bf PH}^\intercal\\
{\bf HP} & {\bf HPH}^\intercal + {\bf R}
\end{bmatrix}
\right)
$$
{{< /math >}}

and the marginal distribution for ${\bf y}$ is given by

{{< math >}}
$$
{\bf y}\sim{\cal N}\left({\bf Hm} + {\bf u}, {\bf HPH}^\intercal + {\bf R}\right)
$$
{{< /math >}}

## Lemma A2

Suppose ${\bf x}\in\mathbb{R}^n$ and ${\bf y}\in\mathbb{R}^m$ have joint Gaussian distribution of the form

{{< math >}}
$$
\begin{pmatrix}
{\bf x}\\
{\bf y}
\end{pmatrix}
\sim
{\cal N}\left(
\begin{bmatrix}
{\bf a}\\
{\bf b}
\end{bmatrix},
\begin{bmatrix}
{\bf A} & {\bf C}^\intercal\\
{\bf C} & {\bf B}
\end{bmatrix}
\right)
$$
{{< /math >}}

then

{{< math >}}
$$
\begin{aligned}
{\bf x}&\sim{\cal N}({\bf a}, {\bf A})\\
{\bf y}&\sim{\cal N}({\bf b}, {\bf B})\\
{\bf x} | {\bf y} &\sim {\cal N}({\bf a} + {\bf CB}^{-1}({\bf y} - {\bf b}), {\bf A} - {\bf CB}^{-1}{\bf C}^\intercal)\\
{\bf y}\vert {\bf x} &\sim {\cal N}({\bf b} + {\bf C}^\intercal{\bf A}^{-1}({\bf x} - {\bf a}), {\bf B} - {\bf C}^\intercal{\bf A}{\bf C})
\end{aligned}
$$
{{< /math >}}

# References

1. Särkkä, S. (2013). *Bayesian Filtering and Smoothing* (Institute of Mathematical Statistics Textbooks). Cambridge: Cambridge University Press. doi:10.1017/CBO9781139344203