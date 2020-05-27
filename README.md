# Well-Calibrated Regression Uncertainty in Medical Imaging with Deep Learning

Max-Heinrich Laves, Sontje Ihler, Jacob F. Fast, Lüder A. Kahrs, Tobias Ortmaier

## Abstract

The consideration of predictive uncertainty in medical imaging with deep learning is of utmost importance.
We apply estimation of predictive uncertainty by variational Bayesian inference with Monte Carlo dropout to regression tasks and show why predictive uncertainty is systematically underestimated.
We suggest to use *sigma scaling* with a single scalar value; a simple, yet effective calibration method for both aleatoric and epistemic uncertainty.
The performance of our approach is evaluated on a variety of common medical regression data sets using different state-of-the-art convolutional network architectures.
In all experiments, sigma scaling is able to reliably recalibrate predictive uncertainty, surpassing more complex calibration methods.
It is easy to implement and maintains the accuracy.
Well-calibrated uncertainty in regression allows robust rejection of unreliable predictions or detection of out-of-distribution samples.

## Contact

Max-Heinrich Laves  
[laves@imes.uni-hannover.de](mailto:laves@imes.uni-hannover.de)  
[@MaxLaves](https://twitter.com/MaxLaves)

Institute of Mechatronic Systems  
Leibniz Universität Hannover  
Appelstr. 11A, 30167 Hannover, Germany
