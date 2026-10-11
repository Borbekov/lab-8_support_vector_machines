# Lab 8: Support Vector Machines

Practical exercises from ISLR (Introduction to Statistical Learning) Chapter 9, implemented in Python using sklearn and the ISLP library.

## Overview

This lab covers maximal margin classifiers, support vector classifiers, and support vector machines with non-linear kernels — applied to simulated two- and three-class data, gene expression data, car fuel efficiency, and orange juice brand purchases.

## Datasets

- **Khan** — gene expression measurements (2,308 genes) for four types of small round blue cell tumors (63 training / 20 test observations)
- **Auto** — car specifications and gas mileage (392 observations)
- **OJ** — Citrus Hill vs. Minute Maid orange juice purchases (1,070 observations)
- Simulated two-dimensional datasets: linearly separable, barely separable, non-linear (radial/quadratic/cubic boundaries), and a three-class variant

## Topics Covered

### Support Vector Classifier

- `sklearn.svm.SVC(kernel='linear')` and the role of the cost parameter `C` (small `C` → wide margin, many support vectors; large `C` → narrow margin, few support vectors)
- Visualizing the decision function and support vectors with `ISLP.svm.plot`
- Extracting the linear decision boundary coefficients via `coef_`
- Tuning `C` with `GridSearchCV` + `KFold` and inspecting `cv_results_`
- Linearly separable data: maximal margin hyperplane with very large `C` vs. a more stable, wider-margin fit with small `C`

### Support Vector Machine

- Non-linear kernels: polynomial (`kernel='poly'`, `degree`) and radial (`kernel='rbf'`, `gamma`)
- Effect of large `C` on training errors and boundary irregularity (overfitting risk)
- Joint cross-validation over `C` and `gamma` for the radial kernel

### ROC Curves

- Using SVM decision function values as scores for ROC analysis via `RocCurveDisplay.from_estimator`
- Training vs. test ROC curves for a tuned model and a highly flexible model ($\gamma = 50$)

### SVM with Multiple Classes

- Multi-class classification with `decision_function_shape='ovo'` (one-versus-one) and `'ovr'` (one-versus-rest)

### Application to Gene Expression Data

- Linear-kernel SVC on the high-dimensional **Khan** data ($p \gg n$), where additional kernel flexibility is unnecessary
- Perfect separation on training data and evaluation on the held-out test set

## Model Evaluation

- `confusion_table` / `confusion_matrix` for classification results
- Training, cross-validation, and test error rates compared across values of `C`, `gamma`, and `degree`
- Number of support vectors (`n_support_`) as an indicator of margin width
- ROC curves on training vs. test data

## Pipeline

- `GridSearchCV` with `param_grid` over `C`, `gamma`, and `degree`
- `KFold` cross-validation with shuffling
- `StandardScaler` for standardizing predictors before fitting SVMs
- `DecisionBoundaryDisplay` for plotting decision regions

## Exercises

- Simulated non-linear two-class data: linear SVC vs. polynomial and radial SVMs, comparing training and test error rates with decision boundary plots
- Quadratic decision boundary: logistic regression with linear vs. non-linear features ($X_1^2$, $X_2^2$, $X_1 X_2$), compared with a linear SVC and a radial SVM
- Barely linearly separable data: training, cross-validation, and test errors across a range of `C`, testing the claim that a small `C` can generalize better than a huge one
- **Auto**: predicting high vs. low gas mileage with linear, radial, and polynomial SVMs tuned by cross-validation
- **OJ**: predicting `Purchase` with a linear SVC — number of support points, training/test error rates, and selecting optimal `C` via cross-validation on standardized predictors

## Technologies

- Python 3.12
- scikit-learn
- ISLP
- pandas, numpy
- matplotlib
