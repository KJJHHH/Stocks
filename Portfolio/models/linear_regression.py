import copy
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression 

def linear_regression(
        param,
        fnl_df,                  # data
        tune = False):
    
    (X_train, y_train, X_test, y_test) = fnl_df

    model = LinearRegression(**param)
    model.fit(np.array(X_train), y_train)
    pred = pd.DataFrame(model.predict(np.array(X_test)), columns = ["prediction"]).set_index(y_test.index)
    y_train_hat = pd.DataFrame(model.predict(np.array(X_train)), columns = ["train_pred"]).set_index(y_train.index)
    loss = "na"

    return pred, y_train_hat, loss
    
