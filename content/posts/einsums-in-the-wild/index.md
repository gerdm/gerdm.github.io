---
title: "Einsums in the wild"
date: 2021-11-25
description: "Description of the Einsums and their usage in machine learning."
katex: true
draft: false
tags: [einsums, code, tutorial]
---

> Do you know what `inm,kij,jnm->knm` is all about?

For an interactive version of this post, see [this Colab notebook](https://colab.research.google.com/github/probml/probml-notebooks/blob/main/notebooks/einsums.ipynb).

# Introduction

Linear combinations are ubiquitous in machine learning and statistics. Many algorithms and models in the statistics and machine learning literature can be written (or approximated) as a matrix-vector multiplication. Einsums are a way of representing the linear interaction among vectors, matrices and higher-order dimensional arrays.

In this post, I lay out examples that make use of Einsums. I assume that the reader is familiar with the basics of einsums. However, I provide a quick introduction in the next section. For reference, see also [[1]](https://rockt.github.io/2018/04/30/einsum) and [[2]](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html). Throughout this post, we borrow from the [numpy literature](https://numpy.org/doc/stable/reference/arrays.ndarray.html) and denote the element ${\bf x} \in \mathbb{R}^{M_1 \times M_2 \times \ldots \times M_N}$ as an $N$-dimensional array.

In the next section, we present a brief summary of einsum expressions and its usage in numpy / jax.numpy.

## An quick introduction to einsums: from sums to indices

Let ${\bf a}\in\mathbb{R}^M$ a 1-dimensional array. We denote by $a_m$ the $m$-th element of $\bf a$. Suppose we want to express the sum over all elements in $\bf a$. This can be written as

$$
\sum_{m=1}^M a_m
$$

To introduce the einsum notation, we notice that the sum symbol ($\Sigma$) in this equation simply states that we should consider all elements of $\bf a$ and sum them. If we assume that 1) there is no ambiguity on the number of dimensions in $\bf a$ and 2) we sum over all of its elements, we define the einsum notation for the sum over all elements in the 1-dimensional array $\bf a$ as

$$
\sum_{m=1}^N a_m\stackrel{\text{einsum}}{\equiv}{\bf a}_m
$$

To keep our notation consistent,  we denote indices with parenthesis as *static* dimensions. Static dimensions allows us to expand the expressiveness power of einsums. That is, we denote all of the elements of $\bf a$ under the einsum notation as ${\bf a}_{(m)}$.

Since the name of the arrays are not necessarily meaningful to define these expressions, we define einsum expressions in numpy by focusing only on the indices. To represent which dimensions are *static* and which should be summed over, we introduce the `->` notation. Elements to the left of `->` define the set of indices of an array and elements to the right of `->` represent indices that we do **not** sum over. For example, the sum over all elements in $\bf a$ is written as

$$
{\bf a}_m \equiv \texttt{m->}
$$

and the selection of all elements of $\bf a$ is written as

$$
{\bf a}_{(m)} \equiv \texttt{m->m}
$$

In the following snippet, we show this notation in action.

```python
>>> a = np.array([1, 2, 3, 4])
>>> np.einsum("m->", a)
10
>>> np.einsum("m->m", a)
array([1, 2, 3, 4])
```

## Higher-dimensional arrays

Let ${\bf a}\in\mathbb{R}^M$ and ${\bf b}\in\mathbb{R}^M$ be two one-dimensional arrays. The dot product between $\bf a$ and $\bf b$ can be written as

$$
\begin{aligned}
{\bf a}^T {\bf b}
    &= a_1 b_1 + \ldots + a_M b_M\\\\
    &= \sum_{m=1}^M a_m b_m
\end{aligned}
$$

Following our previous notation, we see that the representation of this einsum expression in mathematical and numpy form is

$$
{\bf a}_m {\bf b}_m \equiv \texttt{m,m->}
$$

Furthermore, the einsum representation of the element-wise product between $\bf a$ and $\bf b$ is given by

$$
{\bf a}\_{(m)} {\bf b}\_{(m)} \equiv \texttt{m,m->m}
$$

As an example, consider the following 1-dimensional arrays `a` and `b`

```python
>>> a = np.array([1, 3, 1])
>>> b = np.array([-1, 2, -2])
>>> # dot product
>>> a @ b
3
>>> np.einsum("m,m->", a, b)
3
>>> # Element-wise product
>>> a * b
array([-1,  6, -2])
>>> np.einsum("m,m->m", a, b)
array([-1,  6, -2])
```

We can generalise the ideas previously presented. Consider the matrix-vector multiplication between ${\bf A}\in\mathbb{R}^{N\times M}$ and ${\bf x}\in\mathbb{R}^M$. We can write this in linear algebra form as

{{< math >}}
$$
\begin{aligned}
{\bf A x} &= 
\begin{bmatrix}
{\bf a}_1^T \\
\vdots \\
{\bf a}_N^T
\end{bmatrix} {\bf x}\\
&= \begin{bmatrix}
{\bf a}_1^T {\bf x} \\
\vdots \\
{\bf a}_N^T {\bf x}
\end{bmatrix}
\end{aligned}
$$
{{< /math >}}

Where we have denoted ${\bf a}_n^T$ as the $n$-th row of $\bf A$. From the equation above, we notice that the $n$-th entry of ${\bf A x} \in \mathbb{R}^N$ can be expressed as

{{< math >}}
$$
\begin{aligned}
({\bf Ax})_n &= \sum_{m=1}^M a_{n,m} x_m
\end{aligned}
$$
{{< /math >}}


We also observe that the first dimension of $\bf A$ for this expression is is static. The einsum representation in mathematical/numpy form becomes

$$
{\bf A}_{(n),m}{\bf x}_m \equiv \texttt{nm,m->n}
$$

Considering the result of the last example, we can easily express the resulting $i,j$-th entry of the  multiplication between two matrices. Let ${\bf A}\in\mathbb{R}^{N\times M}$, ${\bf B}\in\mathbb{R}^{M\times K}$, the product between $\bf A$ and $\bf B$ becomes 


{{< math >}}
$$
\begin{aligned}
{\bf A B} &=
\begin{pmatrix}
{\bf a}_1^T \\
\vdots \\
{\bf a}_N^T
\end{pmatrix}
\begin{bmatrix}
{\bf b}_1, \ldots, {\bf b}_M \\
\end{bmatrix} \\
&= \begin{pmatrix}
{\bf a}_1^T {\bf b}_1 & {\bf a}_1^T {\bf b}_2 & \ldots & {\bf a}_1^T {\bf b}_M \\
{\bf a}_2^T {\bf b}_1 & {\bf a}_2^T {\bf b}_2 & \ldots & {\bf a}_2^T {\bf b}_M\\
\vdots & \vdots & \ddots & \vdots \\
{\bf a}_N^T {\bf b}_1 & {\bf a}_N^T {\bf b}_2 & \ldots & {\bf a}_N^T {\bf b}_M
\end{pmatrix}
\end{aligned}
$$
{{< /math >}}

Then, the $(i,j)$-th entry of the matrix-matrix multiplication $\bf AB$ can be expressed as

{{< math >}}
$$
\begin{aligned}
{\bf AB}_{ij} &= {\bf a}_i^T {\bf b}_j \\
&= \sum_{m=1}^M a_{i,m} b_{m, j}
\end{aligned}
$$
{{< /math >}}



From the equation above, we see that the first dimension of $\bf A$ and the second dimension of $\bf B$ are static. We represent its einsum form as

{{< math >}}
$$
{\bf A}_{(i),m} {\bf B}_{m, (j)}\equiv \texttt{im,mj->ij}
$$
{{< /math >}}

```python
>>> A = np.array([[1, 2], [-2, 1]])
>>> B = np.array([[0, 1], [1, 0]])
>>> A @ B
array([[ 2,  1],
       [ 1, -2]])
>>> np.einsum("im,mj->ij", A, B)
array([[ 2,  1],
       [ 1, -2]])
```

## Even-higher-dimensional arrays

The advantage of using einsums in machine learning is their expressive power when working with higher-dimensional arrays. As we will see, knowing the einsum representation of a matrix-vector multiplication operation easily allows us to generalise it for multiple dimensions. This is because ensums can be thought of as expressions of linear transformations when *static* dimensions are present in the output.

To motivate the use of of expressing linear combinations as einsums expressions in machine learning, we consider the following example.

# Einsums in machine learning

Let ${\bf x}\in\mathbb{R}^M$ and ${\bf A}\in\mathbb{R}^{M\times M}$ be one-dimensional and two-dimensional arrays respectively. The squared Mahalanobis distance centred at zero with precision matrix $\bf A$ is defined as

$$
D_{\bf A}({\bf x}) = {\bf x}^T {\bf A} {\bf x}.
$$

Using the typical rules for matrix-vector multiplication, we evaluate $D_{\bf A}({\bf x})$ for any given ${\bf x}$ and a valid precision matrix $\bf A$. We readily can evaluate $D_{\bf A}({\bf x})$ as an einsum expression as `i,ij,j->`. This is because

{{< math >}}
$$
\begin{aligned}
{\bf x}^T {\bf A} {\bf x} &= \sum_{i,j} x_i A_{i,j} x_j \\
&\stackrel{\text{einsum}}{\equiv} {\bf x}_i {\bf A}_{i,j} {\bf x}_j
\end{aligned}
$$
{{< /math >}}

A more interesting scenario is to consider the case where we have $N$ observations stored in a 2-dimensional array ${\bf X} \in \mathbb{R}^{N\times M}$. If we denote by ${\bf x}_n \in \mathbb{R}^M$ the $n$-th observation in $\bf X$, to compute the squared Mahalanobis distance for each observation means to obtain

{{< math >}}
$$
{\bf x}_n^T {\bf A} {\bf x}_n \ \forall n \in \{1, \ldots, N\}.
$$
{{< /math >}}

One such a way to obtain a collection of squared Mahalanobis distances evaluated at each of the $N$ elements in $\bf X$ is to compute

$$
\text{Diag}({\bf X}^T{\bf A}{\bf X})
$$

where {{< math >}}$\text{Diag}({\bf M})_i = {\bf M}_{i,i}${{< /math >}}. To see why, note that

{{< math >}}
$$
\begin{aligned}
({\bf X}{\bf A}{\bf X}^T) &= \begin{bmatrix}{\bf x}_1^T \\ \vdots \\{\bf x}_N^T\end{bmatrix} {\bf A}
\begin{bmatrix}{\bf x_1} & \ldots & {\bf x}_N\end{bmatrix}\\
&= \begin{bmatrix}{\bf x}_1^T {\bf A}\\ \vdots \\{\bf x}_N^T {\bf A}\end{bmatrix} \begin{bmatrix}{\bf x_1} & \ldots & {\bf x}_N\end{bmatrix}\\
&= \begin{bmatrix}
{\bf x}_1^T {\bf A} {\bf x}_1 & {\bf x}_1^T {\bf A} {\bf x}_2 & \ldots & {\bf x}_1^T {\bf A} {\bf x}_N \\
{\bf x}_2^T {\bf A} {\bf x}_1 & {\bf x}_2^T {\bf A} {\bf x}_2 & \ldots & {\bf x}_2^T {\bf A} {\bf x}_N \\
\vdots & \vdots & \ddots & \vdots \\
{\bf x}_N^T {\bf A} {\bf x}_1 & {\bf x}_N^T {\bf A} {\bf x}_2 & \ldots & {\bf x}_N^T {\bf A} {\bf x}_N
\end{bmatrix}
\end{aligned} 
$$
{{< /math >}}

So that

$$
\text{Diag}({\bf X}{\bf A}{\bf X}^T)_n = {\bf x}_n^T {\bf A} {\bf x}_n.
$$

The computation of the above expression is inefficient since we need to compute $N^2$ terms to obtain our desired 1-dimensional array of size $N$. A much efficient way to compute and express the set of squared Mahalanobis distances is to make use of einsums. As we’ve seen, $D_{\bf A}({\bf x})$ is written in einsum form as

{{< math >}}
$$
{\bf x}_{i}{\bf A}_{i,j}{\bf x}_j
$$
{{< /math >}}

The extension of the latter expression to a set of $N$ elements is straightforward by noting that we only need to specify that the first dimension of $\bf X$ is static. We obtain

{{< math >}}
$$
{\bf X}_{(n), i}{\bf A}_{i,j}{\bf X}_{(n),j} \equiv \texttt{ni,ij,nj->n}
$$
{{< /math >}}

In this particular example, using einsums helps us avoid computing the terms not in the diagonal, which increases the speed at which we can compute this expression compared to the traditional method

```python
In [1]: %%timeit -n 10 -r 10
   ...: np.diag(X @ A @ X.T)
1.61 s ± 97.3 ms per loop (mean ± std. dev. of 10 runs, 10 loops each)

In [2]: %%timeit -n 10 -r 10
   ...: np.einsum("ni,ij,nj->n", X, A, X, optimize="optimal")
160 ms ± 6.18 ms per loop (mean ± std. dev. of 10 runs, 10 loops each)
```

To generalise this result, consider the 3-dimensional array ${\bf X} \in \mathbb{R}^{N_1 \times N_2 \times M}$. The algebraic representation of the squared Mahalanobis distance evaluated over the last dimension of $\bf X$ is not possible using the basic tools of linear algebra (we will see a use case of this when we show how to plot the log-density of a 2-dimensional Gaussian distribution). From our previous result, we see that to expand this computation for $\bf X$, we only need to introduce an additional *static* index to obtain:

{{< math >}}
$$
{\bf X}_{(n), (m), i}{\bf A}_{i,j}{\bf X}_{(n),(m), j} \equiv \texttt{nmi,ij,nmj->nm}
$$
{{< /math >}}


What these last expressions show is that  einsums can be of great help in scenarios when we have to compute a known linear transformation over unused indices.

If we continue with the process of increasing the dimension of $\bf X$, we obtain the following results

1. `i,ij,j->` for scalar output,
2. `ni,ij,nj->n` for a 1-dimensional array output,
3. `nmi,ij,nmj->nm` for a 2-dimensional array (grid) output,
4. `nmli,ij,nmlj->nml` for a 3-dimensional array output, and
5. `...i,ij,...j->...` for a $d$-dimensional array output.

Furthermore, einsums expressions are commutative over block of indices. This means that the result of the einsum expression is independent of the order in which arrays are positioned. For our previous example, the following three expressions are equivalent:

```python
ni,ij,nj->n
ni,nj,ij->n
ni,nj,ij->n
```

## Log-density of a Gaussian

Let  ${\bf x}\sim\mathcal{N}(\boldsymbol\mu, \boldsymbol\Sigma)$, the log-density of $\bf x$ is given by

$$
\log p({\bf x} \vert \boldsymbol\mu, \boldsymbol\Sigma) = -\frac{1}{2}({\bf x} - \boldsymbol\mu)^T\boldsymbol\Sigma^{-1}({\bf x} - \boldsymbol\mu) + \text{cst.}
$$

Suppose we want to plot the log-density of a bivariate Gaussian distribution up to a normalisation constant over a region $\mathcal{X} \subseteq \mathbb{R}^2$. As we have previously seen, the expression ${\bf x}^T {\bf A} {\bf x}$ can be represented in einsum form as `i,ij,j->`. By introducing *static* dimensions `n` and `m`, we  compute the log-density over $\mathcal X$ by adding the `n` and `m` indices in the einsum expression and specifying them as the final result. We obtain `nmi,ij,nmj->nm`. A common way to obtain the grid $\mathcal X$ in python is through `jnp.mgrid`. We present an example of this below.

```python
mean_vec = jnp.array([1, 0]) 
cov_matrix = jnp.array([[4, -2], [-2, 4]])
prec_matrix = jnp.linalg.inv(cov_matrix)

step = 0.1
xmin, xmax = -8, 8
ymin, ymax = -10, 10
X_grid = jnp.mgrid[xmin:xmax:step, ymin:ymax:step] + mean_vec[:, None, None]

diff_grid = (X_grid - mean_vec[:, None, None])
log_prob_grid = -jnp.einsum("inm,ij,jnm->nm", diff_grid, prec_matrix, diff_grid) / 2
plt.contour(*X_grid, log_prob_grid, 30)
```

![gaussian.png](gaussian.png)

We expand the previous idea to the case of a set of multivariate Gaussians with constant mean and multiple covariance matrices.

Recall that the einsum expression to compute the log-density of a bivariate normal over a region $\mathcal X \subseteq \mathbb{R}^2$ is given by `inm,ij,jnm->nm`. Assuming that we have a set of $K$ Gaussian distributions. For each index $k$, we have a precision matrix ${\bf S}_k$ and constant mean $\boldsymbol\mu \in \mathbb{R}^M$. To compute the density over each of the regions we simply modify our previous expression to take account of a new *static* dimension `k`. We obtain `inm,kij,jnm->knm`. 

As an example, consider the collection of four covariance matrices `C1`,`C2`,`C3`,`C4`. We show that einsums can be used to compute the log-density over the multiple Gaussians

```python
C1 = jnp.array([
    [4, -2],
    [-2, 4]
])

C2 = jnp.array([
    [4, 0],
    [0, 4]
])

C3 = jnp.array([
    [4, 2],
    [2, 4]
])

C4 = jnp.array([
    [1, -2],
    [2, 4]
])

C = jnp.stack([C1, C2, C3, C4], axis=0)
S = jnp.linalg.inv(C) # inversion over the fist dimension

log_prob_grid_multiple = -jnp.einsum("inm,kij,jnm->knm", diff_grid, S, diff_grid) / 2

fig, ax = plt.subplots(2, 2)
ax = ax.ravel()
for axi, log_prob_grid in zip(ax, log_prob_grid_multiple):
    axi.contour(*X_grid, log_prob_grid, 30)
```

![k-gaussians.png](k-gaussians.png)

To recap, the einsum expression for the Mahalanobis distance distance evaluated at $\bf x$ is given by 

- `i,ij,ij->` for a single array (a vector),
- `ni,ij,nj->n` for a collection of $M$-dimensional arrays (a matrix of observations),
- `nmi,ij,nmj->nm` for a grid of of $M$-dimensional arrays,
- `nmi,kij,nmj->knm`for a grid of $M$ dimensional arrays evaluated over different precision matrices.

## Predictive surface of a Bayesian logistic regression model

As long as the inner-most operation we want to compute consists of a linear combination of elements we can make use of einsums. As a next example, consider the estimation of the predictive surface of a logistic regression with Gaussian prior. That is, we want to compute

{{< math >}}
$$
\begin{aligned}
p(\hat y = 1 \vert {\bf x}) &= \int_{\mathbb{R}^2} \sigma({\bf w}^T {\bf x}) p({\bf w} \vert \mathcal{D}) d{\bf w}\\
&= \mathbb{E}{{\bf w} \vert \mathcal{D}}\big[\sigma({\bf w}^T {\bf x})\big]\\
&\approx \frac{1}{S} \sum_{s=1}^S \sigma\left({\bf w^{(s)}}^T {\bf x}\right)
\end{aligned}
$$
{{< /math >}}

Suppose we estimated the posterior parameters $\hat{\bf w}, \hat\Sigma$. Since the posterior predictive distribution is analytically intractable, we turn to a Monte Carlo approximation of the posterior predictive surface. As with the previous two examples, we want to compute $p(\hat y = 1 \vert {\bf x})$ over a surface $\mathcal X \subseteq \mathbb{R}^2$. In this scenario, we have $S$ samples of weights sampled from the posterior ${\bf w} \vert \mathcal D$ which we wish to evaluate over all points in the grid $\mathcal X$. Recalling that the dot product between two vectors is written in einsum form as `m,m->`, to obtain a 3-dimensional array comprising of $S$simulations evaluated at each point in $\mathcal X$, we simply expand the dot product expression to contain the *static* indices `s` for the simulation and `i,j` for the position in the grid. We obtain `sm,mij->sij`. After obtaining the grid `sij`, we can compute the approximated predictive distribution by applying the logistic function over each element and averaging over `s`. The following code shows how to achieve this.

```python
# Estimated posterior mean and precision matrix
w_hat = jnp.array([4.29669692, 1.6520908])
S_hat = jnp.array([[2.74809243, 0.76832627],
                   [0.76832627, 0.88442754]])
C_hat = jnp.linalg.inv(S_hat)

n_samples = 1_000
boundary, step = 8, 0.1
X_grid = jnp.mgrid[-boundary:boundary:step, -boundary:boundary:step]

w_samples = jax.random.multivariate_normal(key, w_hat, S_hat, shape=(n_samples,))

logit_grid = jnp.einsum("sm,mij->sij", w_samples, X_grid)
P_grid = jax.nn.sigmoid(logit_grid).mean(axis=0)
plt.contour(*X_grid, P_grid, 30)
plt.title(r"$p(\hat y = 1 \vert x)$", fontsize=15);
```

![bayesian-logistic-reg.png](bayesian-logistic-reg.png)

## Image compression: Singular value decomposition

A typical example that one encounters while learning about singular value decomposition (SVD) is the use of SVD to decompress an image. An image is usually represented as a 3-dimensional array ${\bf R}\in\mathbb{R}^{N\times M \times 3}$, where the last dimension of $\bf R$ represents the red, blue, and green (RBG) values of the image. To work with SVD we first need to transform $\bf R$ into a two-dimensional grayscale $\bf P$. According to [Wikipedia](https://en.wikipedia.org/wiki/Grayscale#cite_note-5), we can do this by taking a weighted combination of the RGB values, using weights ${\bf c}=[0.2989, 0.5870, 0.1140]$. This can be implemented as an einsum using

{{< math >}}
$$
{\bf P} \stackrel{\text{einsum}}{\equiv} {\bf c}_{k}{\bf L}_{(n),(m),k} \in\mathbb{R}^{N\times M} \equiv \texttt{c,ijc->ij}
$$
{{< /math >}}

It’s a classical result of linear algebra that our matrix $\bf P$ can be factorised as

$$
{\bf P} = {\bf U}\boldsymbol\Sigma {\bf V}^T,
$$

where ${\bf U}\in\mathbb{R}^{M\times M}$, ${\bf V}\in\mathbb{R}^{N\times N}$, and $\boldsymbol\Sigma \in\mathbb{R}^{M\times N}$ is a matrix with diagonal terms $\{\sigma_1, \sigma_2, \ldots, \sigma_{\min(n,m)}\}$ and zero everywhere else. In scipy, the SVD decomposition of $\bf P$ is conviniently factorised (in einsum form) as

{{< math >}}
$$
{\bf P} \stackrel{\text{einsum}}{\equiv}\hat{\bf U}_{(n),k}\hat{\boldsymbol\sigma}_{k} \hat{\bf V}_{k, (m)}
$$
{{< /math >}}

where $\hat{\bf U}\in\mathbb{R}^{M\times R}$, $\hat{\bf U}\in\mathbb{R}^{M\times R}$, $\boldsymbol\sigma=\{\sigma_1 \ldots, \sigma_R\}$, and  $R = \min(M,N)$.

As a pedagogical example, suppose we wish to approximate the matrix $\bf P$ using the first $K < R$ singular components. First, we observe that the ($n,m$)-th entry of $\bf P$ is given by

{{< math >}}
$$
{\bf P}_{n,m} = \sum_{k=1}^R \hat{\bf U}_{n,k}\hat{\boldsymbol\sigma}_{k} \hat{\bf V}_{k, m}
$$
{{< /math >}}

If we wish to consider the first $K$ components of ${\bf P}_{n,m}$, we only need to modify the limit term in the sum to obtain.

{{< math >}}
$$
\sum_{k=1}^K \hat{\bf U}_{n,k}\hat{\boldsymbol\sigma}_{k} \hat{\bf V}_{k, m}
$$
{{< /math >}}

However, this last expression cannot be represented in einsum notation. As me mentioned at the beginning, every einsum expression assumes that the sum is over **all** chosen indices. To get around this constraint, we simply introduce the 1-dimensional vector ${\bf 1}_{\cdot \leq K}$ of size $R$ that has value $1$ for the first $K$ entries and $0$ for the rest $R-K$ elements. Hence, we write the approximation of the matrix $\bf P$ using the first $K$ singular components as

{{< math >}}
$$
\sum_{k=1}^R \hat{\bf U}_{n,k}\hat{\boldsymbol\sigma}_{k} \hat{\bf V}_{k, m} ({\bf 1}_{\cdot \leq K})_{k}
$$
{{< /math >}}

We observe that this is easily written in einsum form as

{{< math >}}
$$
\hat{\bf U}_{(n),k}\hat{\boldsymbol\sigma}_{k} \hat{\bf V}_{k, (m)}({\bf 1}_{\cdot \leq K})_{k} \equiv \texttt{nk,k,km,k->nm}
$$
{{< /math >}}

We could also consider multiple values of $K$. To do this, we define the 2-dimensional array

{{< math >}}
$$
{\bf I} = \begin{bmatrix}
{\bf 1}_{\cdot \leq K}\\
{\bf 1}_{\cdot \leq K_2}\\
\vdots\\
{\bf 1}_{\cdot \leq K_C}\\
\end{bmatrix}
$$
{{< /math >}}

Next, we simply modify our previous expression to take into account the additional static dimension of the matrix $\bf I$. We obtain 

{{< math >}}
$$
\begin{aligned}
\hat{\bf U}_{(n),k}\hat{\boldsymbol\sigma}_{k} \hat{\bf V}_{k, (m)}{\bf I}_{(c), k}  &\equiv \texttt{nk,k,km,ck->nmc}\\
 &\equiv \texttt{nk,k,km,ck->cnm}
\end{aligned}
$$
{{< /math >}}

We provide an example of this idea in the next code: first, we load an image living in a 3-dimensional array. Next, we transform it `img` to obtain a 2-dimensional array `img_gray`. We perform SVD over `img_gray` and define a matrix `indexer` containing our different thresholds. Finally, we make use of our previously-defined expression to compute the SVD approximation of the image at the different values defined in `indexer`.

```python
FILEPATH = os.environ["FILEPATH"]
img = plt.imread(FILEPATH)

c_weights = jnp.array([0.2989, 0.5870, 0.1140])
img_gray = jnp.einsum("c,ijc->ij", c_weights, img)
U, s, Vh = jax.scipy.linalg.svd(img_gray, full_matrices=False)

indexer = s[:, None] > jnp.array([10, 100, 1_000, 5_000])
img_svd_collection = jnp.einsum("nk,k,km,ck->cnm", U, s, Vh, indexer)

fig, ax = plt.subplots(2, 2, figsize=(5, 6))
ax = ax.ravel()
for axi, img_svd in zip(ax, img_svd_collection):
    axi.imshow(img_svd, cmap="bone")
    axi.axis("off")
plt.tight_layout(w_pad=-1)
```

![bimba-svd.png](bimba-svd.png)

---

## Misc examples

- Computing the state-value and action-value function in a tabular space

For an example, see [this notebook](https://github.com/sapienzaio/reinforcement-learning/blob/main/ch04/gridworld-policy-evaluation.ipynb).

```python
single_reward = (rewards[None, :] + vk[:, None])
vk_update = np.einsum("ijkl,ij->k", p_gridworld, single_reward) / num_actions
```

- Diagonal extended Kalman filter (dEKF)

For an example, see [this script](https://github.com/probml/JSL/blob/main/jsl/nlds/diagonal_extended_kalman_filter.py).

```bash
A = jnp.linalg.inv(Rt + jnp.einsum("id,jd,d->ij", Ht, Ht, Vt))
mu_t = mu_t_cond + jnp.einsum("s,is,ij,j->s", Vt, Ht, A, xi)
Vt = Vt - jnp.einsum("s,is,ij,is,s->s", Vt, Ht, A, Ht, Vt) + self.Q(mu_t, t)
```

## Acknowledgements

I would like to thank Kevin Murphy, a.k.a, [sirBayes](https://twitter.com/sirbayes) for his suggestions and comments.