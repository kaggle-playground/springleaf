
# Data Cleaning with Random Forest and Denoising Autoencoder

## Principles & General Ideas

It seems that the key to this competition is about data cleaning. 

Importantly, for category data the naive way is to use `one-hot` encoding, which will result in slow training. Instead, we must use `hashing trick`. For example, to reduce dimensionality N:

naive: 
```
input[key_to_idx(key)] = 1  # one-hot encoding
```

hash:
```
input[hash_fun(key) % M] += 1  #   M << N
```

## Based on Orange in Python

http://docs.orange.biolab.si/reference/rst/Orange.feature.imputation.html

We can modify the `class ImputeLearner`. 


## Using R

### Amelia
In this post: http://fastml.com/impute-missing-values-with-amelia/, they confirmed that imputing missing values is crucial to their success in Kaggle competitions. 

`amelia` seems faster, shoudl we try it? http://gking.harvard.edu/amelia/

### MICE
There is a book on using MICE: http://www.stefvanbuuren.nl/mi/FIMD.html

### Fully conditional specification (FCS)
There is a book and associated toolbox: http://missingdata.lshtm.ac.uk/index.php?option=com_content&view=article&id=217&Itemid=139


## From Scrach

We could do this by ourselves in Python.

### Feature completion with random forest

Since there are lots of missing heterogeneous features (e.g. years, category), we cannot apply matrix completion algorithms here. Instead, we think of it as a series of classification tasks. 

For example, we have 10 columns in total. In column 1, there are lots of 99999999 in it. We convert them into a special value, say `NaN`, the treat them as our testing labels. We use the existing features as our training labels, and all the other 9 feature columns as our training input features. 

In our fist round, we are trying to complete the missing features in the column. Next round, we consider column 2, and so on. 

Due to the robustness of random forest, we choose it as our main tool here. 

### Further clean data with denoising autoencoder

An example is here: http://stackoverflow.com/questions/32407621/impute-multiple-missing-values-in-a-feature-vector

Autoencoder is an unsupervised learning algorithm and can be used to further clean data. Here we can use `Keras`, which has built-in denoising autoencoder module.
