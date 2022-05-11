import pandas as pd
import numpy as np
from functools import partial
from sklearn.datasets import load_iris
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from hyperopt import space_eval

from functions import (
    train_model_xgb,
    model_param_xgb
)
# set the random state
global SEED
SEED = 1234


# Lead the iris dataset which will be used as example dataset
data = load_iris()

# create the raw_data DataFrame
raw_data = pd.DataFrame(data=data.data, columns=data.feature_names)
raw_data['target'] = data.target

# start the hyperopt Trial
trials = Trials()

# create the starting point with the best score of 0
best_score = 0

def get_score(data, model_param):

    global score_list
    score_list = []
    
    # split the data in train ad test data
    X_train, X_test, y_train, y_test = train_test_split(
        raw_data.drop(['target'], axis=1), raw_data[['target']], 
        test_size=0.33, random_state=SEED)
    
    # train the xgboost model on train data using the model_param (choosen by hyperopt) 
    model = train_model_xgb(X_train, y_train, model_param)

    # use model and test data to predict
    y_pred = model.predict(X_test)

    # comapre the y_test with y_pred and create score
    score = -f1_score(y_test['target'], 
        y_pred, average='macro')

    print("score {}".format(score))
    print("model params", str(model_param))
    
    global best_score, best_params
    
    # use new score to deide if global score must be updated
    if score < best_score:
        best_score = score
        best_params = model_param
    return {"loss": score, 
            "status": STATUS_OK}


# include params for fmin
get_score = partial(get_score, raw_data)

# minimize loss by executing different hyperparams scenarios
fmin(fn=get_score, space=model_param_xgb(), algo=tpe.suggest, max_evals=100, trials=trials, rstate=np.random.default_rng(SEED))

print('best socre', best_score)
print('=================================', )
print('best parameters')
print(best_params)


# create DataFrame of all trials for evaluation
results = pd.DataFrame(columns=['iteration'] + list(model_param_xgb()) + ['loss'])

for idx, trial in enumerate(trials.trials):
    row = [idx]
    translated_eval = space_eval(model_param_xgb(), {k: v[0] for k, v in trial['misc']['vals'].items()})
    for k in list(model_param_xgb()):
        row.append(translated_eval[k])
    row.append(trial['result']['loss'])
    results.loc[idx] = row

results.to_csv('results.csv')

