---
title: "Kalman filter primer"
date: 2024-08-04
description: "Primer on the Kalman filter primer."
katex: true
draft: true
tags: [kalman-filter, summary]
tldr: "A primer on the Kalman filter primer."
---

In the book "A Kalman Filter Primer" by R.L. Eubank,
the author provides a _no-frills_ introduction to the Kalman filter.

# Signal plus noise models
The story of the Kalman filter begins with signal plus noise models
{{< math >}}
$$
{\bf y}_{1:t} =
    \underbrace{{\bf f}_{1:t}}_{\text{signal}} + 
    \underbrace{{\bf e}_{1:t}}_{\text{noise}}.
$$
{{< /math >}}
By this, we mean that the observations {{< math >}}${\bf y}_{1:t}${{< /math >}}
can be written as the sum of two components:
a predictable component
{{< math >}}${\bf f}_{1:t}${{< /math >}}
and an unpredictable component
{{< math >}}${\bf e}_{1:t}${{< /math >}}.

By predictable, we mean that the that the covariance of the signal component is non-diagonal.

By unpredictable, we mean that the covariance of the noise component is diagonal.

{{< math >}}
$$
\text{cov}({\bf e}_t, {\bf e}_t) = \boldsymbol{1}(t = s)\,{\bf w}_t\,
$$
{{< /math >}}
for all $t, s = 1, \ldots, T$.

Both of these components have zero mean and are uncorrelated with each other.
Moreover, the noise component {{< math >}}${\bf e}_{1:t}${{< /math >}} is uncorrelated with itself, i.e,
{{< math >}}
$$
\text{cov}({\bf e}_t, {\bf e}_s) = 0, \quad \text{for } t \neq s.
$$
{{< /math >}}


{{< math >}}
$$
    {\bf A}_\text{opt} = \argmin_{\bf A}
$$
{{< /math >}}
