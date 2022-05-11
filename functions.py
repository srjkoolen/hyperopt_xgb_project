import pandas as pd
import numpy as np
from hyperopt import hp
from itertools import islice
import xgboost as xgb

    
    
def model_param_xgb():
    """
    This function keep in dictionaries the models that we want to run and all the parameters
    of the models that we want to test.

    hp.choice(label, options) — Returns one of the options, which should be a list or tuple.
    hp.randint(label, upper) — Returns a random integer between the range [0, upper).
    hp.uniform(label, low, high) — Returns a value uniformly between low and high.
    hp.normal(label, mean, std) — Returns a real value that’s normally-distributed with mean and standard deviation sigma.

    Arguments:
    Returns:
        parameters {dictionary} -- parameters of the model
    """   
    
    params = {
        "alpha": hp.quniform("alpha", 0, 0.8, 0.001),
        "lambda": hp.quniform("lambda", 0.4, 0.8, 0.001),
        "subsample": hp.quniform("subsample", 0.1, 1, 0.001),
        "n_estimators": hp.uniformint("n_estimators", 10, 500),
        "colsample_bytree": hp.quniform("colsample_bytree", 0.4, 0.8, 0.001),
        "learning_rate": hp.quniform("learning_rate", 0.1, 0.5, 0.001),
        "base_score": hp.quniform("base_score", 0.1, 0.7, 0.001),
        "max_depth": hp.uniformint("max_depth", 2,30),
        "max_leaves": hp.uniformint("max_leaves", 0,10),
        "max_bin": hp.uniformint("max_bin", 2,400),
        "random_state": 1234,
        "tree_method": "hist",
        "single_precision_histogram": False,
        "use_label_encoder": False,
        'algorithm': xgb.XGBClassifier,        
    }
    return params


def get_model_parameters(values):
    
    values_model = values.copy()
    [values_model.pop(param) for param in ['algorithm']]
    return values_model


def train_model_xgb(X_train, y_train, params):
    """
    This function extracts from the final model the final variable importance
    Arguments and give the probabilities for each block
    Arguments:
        importances {pandas.DataFrame} -- variable importance
        X_train {pandas.DataFrame} -- Dataframe without the cavity column
        y_train {pandas.DataFrame} -- Dataframe containing the cavity column
    Returns:
        model_fit {sklearn.ensemble.model} -- fitted model
    """
    
    model_parameters = get_model_parameters(params)
    # Random seed
    # Apply the xgb boost
    model_fit = params["algorithm"](**model_parameters)
    # fit
    model_fit.fit(X_train, y_train)
    return model_fit