## Melgani part

## CAP 1

## CAP 2

Explain non-parametric estimation, what non-parametric means and why we use it, explain both KNN and Parzen windows, do they tend to the optimal solution? Explain ML estimation, develop the first derivative for a gaussian and explain the meaning of the result. Explain box classifier

What is the formula of Maximum Likelihood, why we can use the separate products of the probabilities and what the Likelihood describe. How we can describe the goodness of an estimation and Is Maximum Likelihood Estimation a good estimator? Is it better or not with respect to the Bayesian Estimation? Why?  
Then he asked me to apply Maximum Likelihood estimation on an exercise.  
Last question was: In which case ML and Bayesian are the same?

Parzen Window Estimation: context, properties etc. Which is the most common window function (write the formula)? Is Parzen a good estimator (bias/variance)? Graphical representation of how the counting/interpolation works with  a couple of samples.

Parzen Windows: general context (what we want to do, why they are non parametric estimation), gaussian kernel with spherical symmetry with formula and some drawing to represent it, How the choice of the width impacts our estimate? Is it a good estimate? (Bias, variance and consideration about the characteristics of the width to have a consistent estimate)

K-NN estimation, how does it works, we draw a graph for example and asked how is the distribution, what does it mean overfit and oversmooth. Then he asked how the K-NN classifier works and made an example. Last question what are the difference with parzen window

What is an EM algorithm and how does it work? Explain formula and both steps. Then Multi Gaussian example for EM algorithm. KNN and KNN example.

## CAP 3

Feature reduction, Huges effect and Huges graph. Fast geometrical explanation of the curse of dimensionality. 

Asked me about feature selection. While talking about different separability measures he wanted to know in depth about Jeffrey-Matusita, the formula and its graphical meaning, how to compute it when we have 3 classes (average) and how it can be expressed as the Bhattacharyya distance (he only wanted the final result, not the full demonstration). Then we talked about search strategies (both SFS and SBS), and if the result of SFS and SBS can be the same. 

Meanings of divergence measure and feature selection: He asked about feature selection and difference between extraction then we focused on divergence measure and Matusita distance and formulas. Then We talked about SFS and divergence matrix

Asked me to explain PCA. When I started explaining he asked about covariance matrix, how it is created, what is cov matrix and etc. Then also some detailed questions about characteristic equation, eigenvalues/vectors. Lastly, main differences between sfs and pca. 

LDA: what is it? Is it a supervised method or unsupervised method? Is PCA supervised or unsupervised? Define the principal formulas. Specify if the means in the formula are projections of means of classes or means of the projection of points on the chosen direction: are these two objects the same? Make clear the dependence of the formula on projection direction, also by defining scatter matrices (and what characteristics of the data do these represent?). How changes the framework with multiple classes and what do the C-1 eigenvectors found represent? Write the solution for the binary classification case. Finally, he asked me to make a graphical example in a 2D feature space where PCA and LDA produced a much equivalent result, and one where the result was much different (similar to the one on slides).

LDA: What is it? Definition of between-class distance and within-class distance. Definition of best case for two distributions, following the two criteria (Compact classes (as impulses) distant one from the other). Formula for Fisher Linear Discriminant, both with distances and Scatter Matrices. Which is the formula for the w maximizing the Generalized Rayleigh ratio? Citation of LDA in multiclass case.

When are the results of LDA and PCA the same?


## CAP 4

NOT SURE
Design phase of an ML system, difference between supervised and unsupervised classification, some examples of loss functions (Mean Squared Error (MSE), Mean Absolute Error (MAE) ) (for regression), cross-validation (k-fold, leave-one-out), confusion matrix (two-class/multi-class, accuracy, precision, recall). What can we do when the model has low accuracy?


Generalization Error (and then Empirical error): I told him the contest of the gen. error like the matrix with the costs of each action and stuff. Regarding formulas, he asked me only the gen.error formula without the demonstration and he asked me to explain it. That was pretty much it.

How does the Minimum risk criterium work, comparison with MAP and ML classification. Which is the best one (minimum risk). He also asked me the formula of the Gaussian curve in a multi dimensional space (with mean vector and covariance matrix).

MAP criterion: comparison and main differences between ML. Example of how priors change the threshold compared to ML criterion. Proof of MAP as error minimization. At the end of the day is better MAP or ML and why? Why MAP could not be the best one? (Linked to approximation error)

Generalization error (Approximation error - Estimation error) - (Trained machine) - Target space/ hypothesis space - Empirical risk

Loss function (correlation coefficient - MSE - RMSE-absolute error) 

Cross validation (exhaustive / non-exhaustive(k-fold) /nested(kxl))

 matrix (binary and multi-class classifiers)  + examples + common derivations (accuracy/precision/sensitivity/f-one-score) + graphical  explanation

ROC + graphical explanation - examples - AUC

### All chapters

Block scheme/Design phases of an ML system, describe all the blocks and phases with examples (in particular at the beginning of data acquisition and feature extraction, which is the best feature in terms of discrimination, cit. of LDA). Then speaking about the selection (cit. of overfitting), the training and the evaluation of the model (cit. of k-fold), we reach the probabilities regarding the binary classifier  (cit. of MAP). At the end, what can we do when the model has low accuracy or in general if a final user wants to reduce the probability of false alarm and the probability of missed alarm?