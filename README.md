# Recalibration of Aleatoric and Epistemic Regression Uncertainty in Medical Imaging

Max-Heinrich Laves, Sontje Ihler, Jacob F. Fast, Lüder A. Kahrs, Tobias Ortmaier

Code for our MIDL2019 paper *Well-Calibrated Regression Uncertainty in Medical Imaging with Deep Learning* and our subsequent MELBA submission *Recalibration of Aleatoric and Epistemic Regression Uncertainty in Medical Imaging*.

## Abstract

The consideration of predictive uncertainty in medical imaging with deep learning is of utmost importance.
We apply estimation of predictive uncertainty by variational Bayesian inference with Monte Carlo dropout to regression tasks and show why predictive uncertainty is systematically underestimated.
We suggest to use *sigma scaling* with a single scalar value; a simple, yet effective calibration method for both aleatoric and epistemic uncertainty.
The performance of our approach is evaluated on a variety of common medical regression data sets using different state-of-the-art convolutional network architectures.
In all experiments, sigma scaling is able to reliably recalibrate predictive uncertainty, surpassing more complex calibration methods.
It is easy to implement and maintains the accuracy.
Well-calibrated uncertainty in regression allows robust rejection of unreliable predictions or detection of out-of-distribution samples.

## BibTeX

MELBA2021

```
under review
```

MIDL2020

```
@inproceedings{laves2020well,
  title={Well-calibrated regression uncertainty in medical imaging with deep learning},
  author={Laves, Max-Heinrich and Ihler, Sontje and Fast, Jacob F and Kahrs, L{\"u}der A and Ortmaier, Tobias},
  booktitle={Medical Imaging with Deep Learning},
  pages={393--412},
  year={2020},
  organization={PMLR}
}
```

## How to run the code

### Requirements

* Python 3
* pytorch
* fire
* numpy
* tqdm
* seaborn

We provide bash scripts for all experiments in the paper.
You can simply run them by e.g. `./train_breastpathq_gaussian.sh`, which trains all models on the BreastPathQ data set with dropout using the full negative log-likelihood as loss function.
However, you have to provide the data sets and adjust the paths in the code.

All evaluations were done using Jupyter notebooks.
E.g. `test_boneage_gaussian_ood_rejection.ipynb` performs the OoD and rejection experiments for calibrated BNNs.
The notebooks that contain "levi" in the file name perform the method of Levi et al. (2019).

## Contact

Max-Heinrich Laves  
[laves@imes.uni-hannover.de](mailto:laves@imes.uni-hannover.de)  
[@MaxLaves](https://twitter.com/MaxLaves)

Institute of Mechatronic Systems  
Leibniz Universität Hannover  
An der Universität 1, 30823 Garbsen, Germany
