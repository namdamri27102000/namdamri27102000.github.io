---
layout: post
title: "Linear Regression"
description: "Linear Regression is the fundamental statistical method for predicting a continuous outcome based on the linear combination of input features."
date: 2025-06-29
feature_image: https://phantichnghiepvu.com/wp-content/uploads/2022/09/reg5-1024x577.png
tags: [machine learning]
---

Linear regression is a fundamental algorithm in Machine Learning, but no AI Expert can ignore it. Fundamentally, even complex Deep Neural Networks (DNNs) comprises linear nodes and these non-linear activation layers, so deep understanding of linear regression is always first step of AI Career, we can use it as a baseline to compare with more advanced algorithms. 

<!--more-->

### The Goal of Linear Regression

The primary goal of linear regression is to find the **best-fitting straight line** (or hyperplane in higher dimensions) that minimizes the distance between the observed data points and the line. This line is often referred to as the **regression line**. Once this line is determined, it can be used to:

* **Predict** the value of the dependent variable for new, unseen data.
* **Understand** the strength and direction of the relationship between variables.
* **Identify** which independent variables are significant predictors.

### Types of Linear Regression

There are two main types of linear regression:

1.  **Simple Linear Regression:** Involves one independent variable and one dependent variable. The relationship can be represented by the equation:

    $$Y = \beta_0 + \beta_1 X + \epsilon$$

    Where:
    * $Y$ is the dependent variable.
    * $X$ is the independent variable.
    * $\beta_0$ is the **y-intercept** (the value of Y when X is 0).
    * $\beta_1$ is the **slope** of the regression line (the change in Y for a one-unit change in X).
    * $\epsilon$ is the **error term** (representing random variability or unmeasured factors).

2.  **Multiple Linear Regression:** Involves two or more independent variables and one dependent variable. The equation extends to:

    $$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_n X_n + \epsilon$$

    Where:
    * $X_1, X_2, ..., X_n$ are the multiple independent variables.
    * $\beta_0, \beta_1, ..., \beta_n$ are their respective coefficients.

### How it Works: Ordinary Least Squares (OLS)

The most common method for fitting a linear regression model is **Ordinary Least Squares (OLS)**. OLS works by minimizing the sum of the squared differences between the actual observed values ($Y_i$) and the predicted values ($\hat{Y}_i$) from the regression line. This sum is known as the **Residual Sum of Squares (RSS)** or **Sum of Squared Errors (SSE)**.

$$RSS = \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2$$

By minimizing this sum, OLS finds the unique line that best represents the trend in the data.

### Assumptions of Linear Regression

For the results of a linear regression model to be reliable and interpretable, several assumptions should ideally be met:

* **Linearity:** The relationship between the independent and dependent variables is linear.
* **Independence of Errors:** The residuals (errors) are independent of each other.
* **Homoscedasticity:** The variance of the residuals is constant across all levels of the independent variables.
* **Normality of Errors:** The residuals are normally distributed.
* **No or Little Multicollinearity:** In multiple linear regression, independent variables should not be highly correlated with each other.

### Applications

Linear regression is widely used in various fields, including:

* **Economics:** Predicting economic growth, inflation, or stock prices.
* **Finance:** Assessing risk and return of investments.
* **Healthcare:** Predicting disease progression or treatment effectiveness.
* **Marketing:** Analyzing the impact of advertising on sales.
* **Engineering:** Modeling system behavior or predicting material properties.

Understanding linear regression provides a fundamental building block for more advanced statistical modeling and machine learning techniques.