#How to integrate sklearn with nolearn.Lasagne

nolearn.lasagne will use NeuralNet to control all layer information and parameter information

To use random search to tune hyperparameter for nolearn.lasagne, we will pass the net to random search as input like following:
```
random_search = RandomizedSearchCV(net1, {'hidden_num_units': sp_randint(50, 200)})
random_search.fit(X, y)
print random_search.grid_scores_
```
RandomizedSearchCV has several inputs, the model, and the parameters that you want to tune.

Then use to conduct the training
```
random_search.fit(X, y)
```
whole example file is individual_experiment/nolearn.lasagne_try.py
