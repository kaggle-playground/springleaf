# Restricted Boltzmann Machine (RBM) for Missing Values

In principle, generative models (e.g. RBM) can handle noisy and missing values data very well. Because they can model the distribution of the input P(X), whereas discrimitive models (e.g. SVMs) only model the distribution of P(y|X). 

## Tools

[RBM in Sklearn](http://scikit-learn.org/stable/auto_examples/neural_networks/plot_rbm_logistic_classification.html)

## Related Papers & References

https://www.quora.com/What-are-some-efficient-implementations-of-RBM-that-accept-gaussian-visible-units

http://www.machinelearning.org/proceedings/icml2007/papers/407.pdf

https://github.com/echen/restricted-boltzmann-machines

As for the ICML 2007 paper, we don't have to pay too much attention to the theoretical details. We just need to use RBM to re-implement the algorithms and apply it to our data at hand.
