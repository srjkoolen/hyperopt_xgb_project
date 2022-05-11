import pandas as pd
import numpy as np
from functools import partial
from sklearn.datasets import load_iris
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score

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

del data



trials = Trials()


best_score = 0
iteration_report_list = []
def get_score(data, model_param):

    global score_list
    score_list = []
    
    X_train, X_test, y_train, y_test = train_test_split(
        raw_data.drop(['target'], axis=1), raw_data[['target']], 
        test_size=0.33, random_state=SEED)
    
    model = train_model_xgb(X_train, y_train, model_param)

    
    y_pred = model.predict(X_test)

    score = -f1_score(y_test['target'], 
        y_pred, average='macro')

    print("score {}".format(score))
    print("model params", str(model_param))
    
    global best_score, best_params
    
    if score < best_score:
        best_score = score
        best_params = model_param
    return {"loss": score, 
            "status": STATUS_OK}


# include params for fmin
get_score = partial(get_score, raw_data)

# minimize loss by executing different hyperparams scenarios
fmin(fn=get_score, space=model_param_xgb(), algo=tpe.suggest, max_evals=10, trials=trials, rstate=np.random.default_rng(SEED))

print('best socre', best_score)
print('=================================', )
print('best parameters')
print(best_params)
