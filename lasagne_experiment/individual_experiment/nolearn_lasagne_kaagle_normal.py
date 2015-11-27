import numpy as np
import pandas as pd


from nolearn.lasagne import NeuralNet

import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum

from sklearn.grid_search import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from scipy.stats import randint as sp_randint

#DATA_Path = '/home/yumeng/Documents/project/kaggle_springleaf/Lasagne_experiment/source/'
DATA_Path= '~/docker_folder/kaggle_springleaf/source/'
TRAIN_DATA_FILENAME = 'train.csv'
TEST_DATA_FILENAME = 'test.csv'


def load_train_data():
    print("Loading training data")
    df = pd.read_csv(DATA_Path+TRAIN_DATA_FILENAME)

    # Remove line below to run locally - Be careful you need more than 8GB RAM
    df = df.sample(n=800)

    labels = df.target

    df = df.drop('target',1)
    df = df.drop('ID',1)

    # Junk cols - Some feature engineering needed here
    df = df.ix[:, 520:660].fillna(-1)

    X = df.values.copy()

    np.random.shuffle(X)

    X = X.astype(np.float32)
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels).astype(np.int32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, encoder, scaler

def load_test_data( scaler):
    print("Loading Test Data")
    df = pd.read_csv(DATA_Path+TEST_DATA_FILENAME)
    ids = df.ID.astype(str)

    df = df.drop('ID',1)

    # Junk cols - Some feature engineering needed here
    df = df.ix[:, 520:660].fillna(-1)
    X = df.values.copy()

    X, = X.astype(np.float32),
    X = scaler.transform(X)
    return X, ids






if __name__ == "__main__":
    X, y, encoder, scaler = load_train_data()
    '''Convert class vector to binary class matrix, for use with categorical_crossentropy'''

    #X_test, ids = load_test_data( scaler)
    print('Number of classes:', len(encoder.classes_))

    num_classes = len(encoder.classes_)
    num_features = X.shape[1]


    net1 = NeuralNet(
        layers=[  # three layers: one hidden layer
            ('input', layers.InputLayer),
            ('hidden', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        # layer parameters:
        input_shape=(None, num_features),
        hidden_num_units=200,  # number of units in hidden layer
        output_nonlinearity=lasagne.nonlinearities.softmax,  # output layer
        output_num_units=num_classes, # 10 target values

        # optimization method:
        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.9,

        regression=False,  # flag to indicate we're dealing with regression problem
        max_epochs=1,  # we want to train this many epochs
        verbose=0,
    )

 #   net1.fit(X,y)

    random_search = RandomizedSearchCV(net1, {'hidden_num_units': sp_randint(50, 200)})
    random_search.fit(X, y)
    print random_search.grid_scores_

#    preds = random_search.predict_proba(X_test, verbose=0)[:, 1]
#    submission = pd.DataFrame(preds, index=ids, columns=['target'])
#    submission.to_csv('Keras_BTB.csv')
