# Smoothed InfoNCE: Breaking the log N Curse without Overshooting

## Background

We revisit the $\log N$ bound of InfoNCE, which sets an upper limit on the estimator thereby often causing the estimator to return an under-estimate of the mutual information. We show that the existing solution of excluding data samples from the reference set causes an equally debilitating problem, namely, it causes the estimator to overshoot, often with no sign of convergence, thereby leading to an over-estimate of the mutual information.

We illustrate the undershooting problem and overshooting problem of InfoNCE in infonce.ipynb.

## Proposed Model: Smoothed InfoNCE

We conduct the experiments on estimating mutual information for high-dimensional Gaussian data. The results show that our model can successfully break the log N curse of InfoNCE without overshooting to infinity.