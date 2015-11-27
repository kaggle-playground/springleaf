# Kaggle Springleaf
https://www.kaggle.com/c/springleaf-marketing-response

Our starting point code base is: https://www.kaggle.com/mpearmain/springleaf-marketing-response/keras-starter-code/code

## Key Components
1. MLP with DBN (deep belief networks)
2. Ensemble based on `sklearn`
3. Random hyperparameter tuning based on `sklearn`

We will be heavily using `nolearn`, https://github.com/dnouri/nolearn. It's a wrapper for `lasagne`, https://github.com/Lasagne/Lasagne

## Procedure
1. Bootstrap the training data with `sklearn` to get subsets.
2. Train MLP with DNB on each subset, with the help of random hyperparameter tuner from `sklearn`, http://scikit-learn.org/stable/auto_examples/model_selection/randomized_search.html
3. Do ensemble learning using `sklearn`, http://scikit-learn.org/stable/modules/ensemble.html#bagging-meta-estimator





