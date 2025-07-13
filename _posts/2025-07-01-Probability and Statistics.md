---
layout: post
title: "Probability"
description: "Probability"
date: 2025-01-01
feature_image: https://online.stanford.edu/sites/default/files/styles/embedded_large/public/2018-04/theory-of-probability_stats116.jpg?itok=oy7zHUnx
tags: [probability and statistics]
---

# Chapter 2: Probability: Univariate Models


## 2.1 Introduction to Probability

Probability theory is foundational to probabilistic machine learning, famously described by Pierre Laplace in 1812 as "common sense reduced to calculation". <!--more--> The chapter introduces two primary interpretations of probability:

*   **Frequentist Interpretation**: In this view, probabilities represent **long-run frequencies of events that can happen multiple times**. For instance, stating that a fair coin has a 50% chance of landing heads implies that if you flip it many times, you expect it to land heads approximately half the time.
*   **Bayesian Interpretation**: While not deeply elaborated in this introductory section, the Bayesian perspective, which interprets probability as a degree of belief, is central to the broader context of Bayesian inference discussed later in Section 2.3.

## 2.2 Random Variables

Random variables are fundamental to probabilistic modeling. This section distinguishes between discrete and continuous types and introduces key concepts for their description.

### Discrete Random Variables

A random variable (RV) $X$ is **discrete** if its sample space $\mathcal{X}$ is finite or countably infinite.
*   The **probability mass function (PMF)**, denoted $p(x)$, computes the probability of $X$ taking on a specific value $x$, i.e., $p(x) \triangleq \text{Pr}(X=x)$.
*   PMFs must satisfy two properties: $0 \le p(x) \le 1$ for all $x$, and $\sum_{x \in \mathcal{X}} p(x) = 1$.
*   For a finite number of values $K$, a PMF can be represented as a list of $K$ numbers or visualized as a histogram.
*   Examples include a **uniform distribution** where each outcome has equal probability (e.g., $p(x)=1/4$ for $x \in \{1,2,3,4\}$), and a **degenerate distribution** (or delta function) that places all its mass on a single value (e.g., $p(x)=I(x=1)$, where $I()$ is the binary indicator function).

### Continuous Random Variables

While not detailed under "Continuous random variables" directly in Section 2.2, the concept is central to the **univariate Gaussian (normal) distribution** discussed in Section 2.6. For continuous variables, probabilities are defined over intervals using a **probability density function (PDF)**. A key insight is that the PDF $p(y)$ at a specific point can be greater than 1, as it represents density, not a probability.

### Sets of Related Random Variables

When dealing with multiple random variables, their relationships are described by:
*   **Joint Distribution**: For two variables $X$ and $Y$, the joint distribution $p(x,y) = \text{Pr}(X=x, Y=y)$ specifies the probability of observing specific values for both simultaneously. For finite variables, this can be a 2D table where all entries sum to one.
*   **Independent Variables**: If $X$ and $Y$ are independent, their joint distribution is the product of their individual (marginal) distributions.
*   **Marginal Distribution**: The marginal distribution of one variable (e.g., $p(X=x)$) is obtained by summing (or integrating) the joint distribution over all possible values of the other variable(s). This is also known as the **sum rule** or **rule of total probability**.

### Independence and Conditional Independence

*   **Conditional Independence ($X \perp Y | Z$)**: This crucial concept means that $X$ and $Y$ are independent given the value of $Z$. Formally, $p(X,Y|Z) = p(X|Z)p(Y|Z)$. This implies that any dependencies between $X$ and $Y$ are mediated through $Z$. This forms the basis of **graphical models**, which use graph structures to encode complex conditional independence assumptions in joint distributions (further discussed in Section 3.6).

### Moments of a Distribution

Moments provide summary statistics of a distribution:
*   **Mode**: The value with the highest probability mass (for discrete) or probability density (for continuous). It is defined as $x^* = \text{argmax}_x p(x)$. A distribution can be **multimodal** (having multiple modes), and even a unique mode may not always be a good summary of the entire distribution.
*   **Conditional Moments**: These describe moments of one variable given the knowledge of another. The **law of iterated expectations** (or law of total expectation) states that $E[X] = E_Y[E[X|Y]]$. This means the expected value of $X$ can be found by first taking the expectation of $X$ conditional on $Y$, and then taking the expectation of that result over $Y$. An illustrative example is calculating the average lifetime of lightbulbs from different factories, considering each factory's output proportion.

### Limitations of Summary Statistics

A significant insight is that relying solely on low-order summary statistics (like mean, variance) can be misleading. The **Datasaurus Dozen** graphically demonstrates this: multiple datasets can have identical means, standard deviations, and correlations, yet their visual plots reveal vastly different underlying structures. This highlights the importance of visualizing data beyond just simple numerical summaries.

## 2.3 Bayes’ Rule

Bayes' rule is a cornerstone of probabilistic machine learning, particularly in **Bayesian inference**, which focuses on "the act of passing from sample data to generalizations, usually with calculated degrees of certainty". Sir Harold Jeffreys famously compared it to Pythagoras's theorem in geometry, emphasizing its fundamental importance.

The process of Bayesian inference involves:
1.  **Prior Distribution ($p(H)$)**: This represents initial beliefs or knowledge about an unknown hidden state $H$ (e.g., a disease, model parameters) before observing any data. If $H$ has $K$ possible values, the prior is a vector of $K$ probabilities summing to 1.
2.  **Observation Distribution ($p(Y|H=h)$)**: This describes the likelihood of observing outcomes $Y$ given a specific hidden state $h$.
3.  **Likelihood Function ($p(Y=y|H=h)$)**: When the observation distribution is evaluated at the actual observed data $y$, it becomes the likelihood function, which is a function of $h$ but not a probability distribution over $y$.
4.  **Unnormalized Joint Distribution**: Multiplying the prior $p(H=h)$ by the likelihood $p(Y=y|H=h)$ gives the unnormalized joint distribution $p(H=h, Y=y)$.
5.  **Marginal Likelihood ($p(Y=y)$)**: This is the normalizing constant obtained by summing (or integrating) the unnormalized joint distribution over all possible hidden states $h'$. It's also known as the "evidence" for the observed data.
6.  **Posterior Distribution ($p(h|y)$)**: The core of Bayesian inference. It is calculated as:
    $$p(h|y) = \frac{p(H=h)p(Y=y|H=h)}{p(Y=y)} = \frac{\text{Prior} \times \text{Likelihood}}{\text{Marginal Likelihood}} \text{}$$
    The posterior provides an updated distribution over the possible states of the world after accounting for the observed data.

Bayes' rule is particularly powerful for **inverse problems**, where the goal is to infer unobserved causes (hidden states) from observed effects (data). This requires defining a "forwards model" $p(y|h)$ and a prior $p(h)$ to constrain the plausible hidden states.

## 2.4 Bernoulli and Binomial Distributions

These distributions are fundamental for modeling binary or count data.

### Bernoulli Distribution

The **Bernoulli distribution** is the simplest probability distribution, used to model **binary events** (outcomes that are either 0 or 1, like a coin toss).
*   It is parameterized by $\theta$, the probability of success (event $Y=1$).
*   The **PMF** is given by:
    $$\text{Ber}(y|\theta) = \begin{cases} 1-\theta & \text{if } y=0 \\ \theta & \text{if } y=1 \end{cases} \text{}$$
    This can be written concisely as $\text{Ber}(y|\theta) \triangleq \theta^y (1-\theta)^{1-y}$.
*   In machine learning, Bernoulli models are often used conditionally, for example, $p(y|x,\theta) = \text{Ber}(y|\sigma(f(x;\theta)))$, where $f$ is an unconstrained function and $\sigma()$ is the **sigmoid (or logistic) function**, defined as $\sigma(a) \triangleq \frac{1}{1 + e^{-a}}$. The sigmoid function is S-shaped.

### Binomial Distribution

The **Binomial distribution** models the number of successes $y$ in a fixed number $N$ of independent Bernoulli trials, each with success probability $\theta$. Figures 2.9(a) and 2.9(b) illustrate its PMF for different $\theta$ values with $N=10$.

### Binary Logistic Regression

This is a widely used classification model that leverages the Bernoulli distribution with a linear predictor.
*   It assumes a linear relationship between inputs and the log-odds of the positive class: $f(x;\theta) = w^T x + b$.
*   The model for the probability of $y=1$ given $x$ is:
    $$p(y=1|x;\theta) = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}} \text{}$$
*   The **decision boundary** is where $p(y=1|x;\theta) = 0.5$, meaning $w^T x + b = 0$. For instance, using the Iris dataset (classifying "Virginica" vs "not Virginica" based on petal width), the decision boundary is approximately at petal length $x^* \approx 1.7$. The model's confidence increases as inputs move further from this boundary.
*   A key insight here is why **linear regression is inappropriate for binary classification**: its predictions can extend beyond the $$ probability range, which is not sensible for probabilities.

## 2.5 Categorical and Multinomial Distributions

These generalize the Bernoulli and Binomial distributions for outcomes with more than two categories.

### Categorical Distribution

The **Categorical distribution** extends the Bernoulli to a $C$-sided event (e.g., rolling a $C$-sided die).
*   Its PMF is $\text{Cat}(y|\theta) \triangleq \prod_{c=1}^C \theta_c^{y_c}$, where $y$ is a one-hot encoding indicating the chosen category, and $\theta_c$ is the probability of category $c$.
*   The Categorical distribution is a special case of the **Multinomial distribution** when the number of trials $N=1$.

### Multinomial Distribution

The **Multinomial distribution** models the counts of observations for each of $C$ categories over $N$ trials.

### Softmax Function

To ensure that an unconstrained function's output can be interpreted as a probability vector for categorical outcomes, the **softmax function** is used:
*   $\text{softmax}(a_c) \triangleq \frac{e^{a_c}}{\sum_{c'=1}^C e^{a_{c'}}}$.
*   The input $a_c$ (often called a "logit" or "pre-activation") can be any real number, and the softmax transforms it into a probability between 0 and 1, with all probabilities summing to 1.
*   A **temperature parameter $T$** can be applied as $\text{softmax}(a/T)$. A high temperature leads to a more uniform distribution, while a low temperature makes the distribution "spiky," concentrating probability mass on the largest elements. This concept is related to the **Boltzmann distribution** in statistical physics.

### Multiclass Logistic Regression

This model is used for classification problems with more than two classes.
*   It employs a linear predictor $f(x;\theta) = Wx + b$, where $W$ is a $C \times D$ weight matrix and $b$ is a $C$-dimensional bias vector.
*   The class probabilities are given by:
    $$p(y=c|x;\theta) = \frac{e^{a_c}}{\sum_{c'=1}^C e^{a_{c'}}} \text{}$$
    where $a = Wx+b$ is the vector of logits.
*   For $C=2$ classes, this model reduces to binary logistic regression.
*   The decision boundaries generated by multiclass logistic regression are **linear**. Nonlinear boundaries can be achieved by transforming the input features (e.g., using polynomial basis functions).

### Log-Sum-Exp Trick

This is a numerical stability technique used when computing expressions involving sums of exponentials, such as in the denominator of the softmax function. It prevents underflow/overflow issues with very small or very large exponents, especially when calculating the cross-entropy loss.

## 2.6 Univariate Gaussian (Normal) Distribution

The **Gaussian distribution**, also known as the **normal distribution**, is arguably the most ubiquitous distribution for real-valued random variables.

### Definition and Properties

*   Its **probability density function (PDF)** is:
    $$N(y|\mu, \sigma^2) \triangleq \frac{1}{\sqrt{2\pi\sigma^2}} \exp \left[-\frac{1}{2\sigma^2}(y-\mu)^2\right] \text{}$$
    where $\mu$ is the **mean** and $\sigma^2$ is the **variance**. The term $\frac{1}{\sqrt{2\pi\sigma^2}}$ is the normalization constant ensuring the density integrates to 1.
*   A Gaussian with $\mu=0$ and $\sigma=1$ is called the **standard normal**.
*   The probability of a continuous variable falling within an interval $(a, b]$ is given by integrating its PDF: $\text{Pr}(a < Y \le b) = \int_a^b p(y)dy$.
*   A key property relating moments is $E[Y^2] = \sigma^2 + \mu^2$.
*   The **standard deviation** $\text{std}[Y] \triangleq \sqrt{V[Y]} = \sigma$ is often preferred over variance as it shares the same units as the variable itself, making it more interpretable.

### Regression with Gaussian Outputs

Gaussian distributions are often used as output models in regression.
*   A **conditional density model** takes the form $p(y|x;\theta) = N(y|f_\mu(x;\theta), f_\sigma(x;\theta)^2)$, where the mean $f_\mu(x;\theta)$ and variance $f_\sigma(x;\theta)^2$ are functions of the input $x$.
*   If the variance is constant, it's called **homoskedastic** regression (fixed $\sigma^2$). If the variance depends on the input, it's **heteroscedastic** (input-dependent $\sigma(x)^2$).

### Why is the Gaussian Distribution So Widely Used?

The sources highlight several reasons for the Gaussian's prevalence:
*   **Interpretable Parameters**: It has two easily interpretable parameters: mean and variance, which capture basic properties of a distribution.
*   **Central Limit Theorem**: As discussed in Section 2.8.6, the sum of a large number of independent random variables, regardless of their original distributions, tends towards a Gaussian distribution. This makes it an excellent choice for modeling **residual errors or "noise"** in many systems.
*   **Maximum Entropy**: The Gaussian distribution makes the **least number of assumptions** (i.e., has maximum entropy) subject to having a specified mean and variance (detailed in Section 3.4.4). This makes it a good default choice when little is known about the underlying distribution beyond its mean and variance.
*   **Mathematical Tractability**: Its simple mathematical form allows for easy implementation and leads to effective methods in various machine learning algorithms.

## 2.7 Some Other Common Univariate Distributions

Beyond the Gaussian, several other univariate distributions are important:

*   **Student t Distribution**: A heavy-tailed distribution, robust to outliers.
*   **Cauchy Distribution**: A special case of the Student t, it notably does not have a well-defined mean or variance.
*   **Laplace Distribution**: Another heavy-tailed distribution, used in robust linear regression due to its connection to L1-regularization (minimizing absolute errors). Its PDF is $\text{Laplace}(y|\mu, b) \propto \exp(-\frac{|y-\mu|}{b})$.
*   **Beta Distribution**: Defined on the interval $$, parameterized by $a$ and $b$. Its PDF is $\text{Beta}(x|a, b) = \frac{1}{B(a, b)} x^{a-1}(1-x)^{b-1}$, where $B(a,b)$ is the Beta function. It is commonly used as a prior distribution for probabilities, for example, in the beta-binomial model. Formulas for its mean, mode, and variance are provided, with conditions for the mode's existence.
*   **Gamma Distribution**: A flexible distribution for positive real-valued random variables ($x>0$). It has shape $a$ and rate $b$ parameters, with PDF $\text{Ga}(x|\text{shape}=a, \text{rate}=b) \triangleq \frac{b^a}{\Gamma(a)} x^{a-1}e^{-xb}$. Its mean and variance exist under certain conditions on $a$.
*   **Empirical Distribution**: This is a non-parametric approximation of a distribution given a set of $N$ samples $D=\{x^{(1)}, \ldots, x^{(N)}\}$. It is represented as a sum of delta functions (spikes) centered at each sample: $p_N(x) = \frac{1}{N} \sum_{n=1}^N \delta_{x^{(n)}}(x)$. The corresponding **empirical CDF** is a staircase function. This concept is crucial for techniques like Monte Carlo approximations.

## 2.8 Transformations of Random Variables

This section addresses how to compute the probability distribution of a new random variable $Y$ when $Y$ is a deterministic function $f(X)$ of another random variable $X$.

### Discrete Case

If $X$ is discrete, the PMF of $Y$ is found by summing the probabilities of all $X$ values that map to the same $Y$ value: $p_Y(y) = \sum_{x:f(x)=y} p_X(x)$.

### Continuous Case

For continuous variables, one typically works with CDFs.
*   **Invertible Transformations (Bijections)**: For a monotonic (and thus invertible) function $f$, the PDF of $Y$ can be found using the **change of variables formula**:
    $$p_Y(y) = p_X(g(y)) \left|\frac{d}{dy} g(y)\right| \text{}$$
    where $g$ is the inverse of $f$. This formula is derived from the principle of probability mass conservation ($p(x)dx = p(y)dy$). This extends to multivariate cases using the Jacobian determinant of the inverse function.
*   **Moments of a Linear Transformation**: If $Y = AX + B$, then $E[Y] = AE[X] + B$. For scalar $y=a^Tx+b$, the expected value is $E[y] = a^T\mu + b$.

### The Convolution Theorem

This theorem describes the distribution of a sum of independent random variables.
*   For discrete independent RVs $x_1$ and $x_2$, the PMF of their sum $y=x_1+x_2$ is $p(y=j) = \sum_k p(x_1=k)p(x_2=j-k)$. An example is the sum of two dice rolls, which tends to look like a Gaussian distribution.
*   For continuous independent Gaussian RVs, if $x_1 \sim N(\mu_1, \sigma_1^2)$ and $x_2 \sim N(\mu_2, \sigma_2^2)$, their sum $y=x_1+x_2$ is also Gaussian: $p(y) = N(y|\mu_1+\mu_2, \sigma_1^2+\sigma_2^2)$.

### Central Limit Theorem (CLT)

The CLT is a profound result stating that the sum (or average) of a large number of independent and identically distributed (IID) random variables, regardless of their individual distributions, will tend to follow an **approximately Gaussian distribution**. Figure 2.23 illustrates this by showing how the distribution of sample means from a Beta distribution approaches a Gaussian as the sample size increases.

### Monte Carlo Approximation

When analytical derivation of a transformed variable's distribution is difficult, **Monte Carlo approximation** offers a powerful numerical approach.
*   It involves drawing a large number of samples from the known distribution $p(x)$, applying the transformation $f$ to each sample to get samples of $y$, and then forming an **empirical distribution** of these $y$ samples. This empirical distribution is an "equally weighted sum of spikes" centered on the samples: $p_S(y) \triangleq \frac{1}{N_s} \sum_{s=1}^{N_s} \delta(y-y_s)$.
*   This method is widely used in statistics and machine learning, with its origins tracing back to the development of the atomic bomb and gambling simulations in Monaco (hence the name "Monte Carlo").

## 2.9 Exercises

The chapter concludes with exercises designed to reinforce understanding of key concepts, including:
*   Conditional independence.
*   Deriving the variance of a sum of random variables ($V[X+Y] = V[X]+V[Y]+2\text{Cov}[X,Y]$).
*   Calculating moments (mean, mode, variance) for the Beta distribution.
*   Applying Bayes' rule to a practical scenario like medical diagnosis, specifically calculating the probability of having a disease given a positive test result and disease prevalence.

***