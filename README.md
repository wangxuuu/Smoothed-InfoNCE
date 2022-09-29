# Smoothed InfoNCE: Breaking the log N Curse without Overshooting

## Background

We revisit the $\log N$ bound of InfoNCE, which sets an upper limit on the estimator thereby often causing the estimator to return an under-estimate of the mutual information. We show that the existing solution of excluding data samples from the reference set causes an equally debilitating problem, namely, it causes the estimator to overshoot, often with no sign of convergence, thereby leading to an over-estimate of the mutual information.

We illustrate the undershooting problem and overshooting problem of InfoNCE in [infonce.ipynb](https://github.com/wangxuuu/Smoothed-InfoNCE/blob/main/infonce.ipynb).

## Proposed Model: Smoothed InfoNCE

We conduct the experiments on estimating mutual information for high-dimensional Gaussian data. The results show that our model can successfully break the log N curse of InfoNCE without overshooting to infinity. The code can be accessed at [smoothed_infonce.ipynb](https://github.com/wangxuuu/Smoothed-InfoNCE/blob/main/smoothed_infonce.ipynb)

## Citation

If you find this code and the paper helpful, please considering cite our work:

> @inproceedings{wang2022smoothed,
  title={Smoothed InfoNCE: Breaking the log N Curse without Overshooting}, 
  author={Wang, Xu and Al-Bashabsheh, Ali and Zhao, Chao and Chan, Chung},
  booktitle={2022 IEEE International Symposium on Information Theory (ISIT)},
  pages={724--729},
  year={2022},
  organization={IEEE}
}