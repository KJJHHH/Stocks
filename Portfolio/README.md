# Build Stock Portfolio 
<!---🦁🙉😹🧑💗🦁🏋🐱🖼️📝--->


Build stock portfolio by utilising different machine learning algorithm including linear regressions, SVM, tree methods, and deep learning and train the algorithms with rolling prediction for each month. The portfolios are based on automotive, semi conductor, and TFT-LCD industry. 

## Data and Preprocessing
- The data for training algorithms are downloaded from TEJ database, including fundimental, chip, and Betas data. Besides, we compute the technical analysis values include RSI, SMA, EMA, MACD, bband, KD, beta, willr, and bias with talib packages
    - 🗣️ [Tutorial for talib](https://medium.com/ai%E8%82%A1%E4%BB%94/%E7%94%A8-python-%E5%BF%AB%E9%80%9F%E8%A8%88%E7%AE%97-158-%E7%A8%AE%E6%8A%80%E8%A1%93%E6%8C%87%E6%A8%99-26f9579b8f3a)
    - 🗣️ [talib packages](https://github.com/TA-Lib/ta-lib-python?tab=readme-ov-file#indicator-groups)
    - 🗣️ [Betas](https://api.tej.com.tw/columndoc.html?subId=51)

- The date to update monthly portfolio for each month: The last revenue announce date for all companies. If any announcement date is later than 12th, delete the company for the month

- For each prediction in rolling prediction, use five-year training data to predict the adjacent month



## Algorithm
### Tuning
- For each algorithms, we utilise grid search or random search
- [Machine learning tuning with **sklearn**](https://scikit-learn.org/stable/modules/grid_search.html)
- [Deep learning tuning with **optuna**](https://github.com/optuna/optuna)

### Models
- Multiple linear regression
    - Tuning Parameters: None
- Elastic Net
    - Tuning Parameters
        ```python
            params = {
                    "l1_ratio" : np.arange(0., 1., .1),
                    "alpha" : [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
                }
        ```
- Decision tree
    - Tuning Parameters
        ```python
            params = {
                    "criterion": ['friedman_mse', 'absolute_error', 'poisson', 'squared_error'],  
                    "max_depth": [None, 5, 10], 
                    "min_samples_split": [5, 10],
                }
        ```
- Random forest
    - Tuning Parameters
        ```python
            params = {
                'n_estimators': [20, 50, 100], 
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2']
            }
        ```
- Xgboost
    - Tuning Parameters
        ```python
            params = {
                "learning_rate": [0.01, 0.1, 0.001],
                "n_estimators": [5, 10, 20, 30], 
                "max_depth": [None, 3, 10, 5],
                "min_child_weight": [1, 2, 3] 
            }
        ```
- SVM
    - Tuning Parameters
        ```python
            params = {
                "C": [0.1, 1, 10],
                "kernel": ["rbf"],
                "gamma": ["scale", "auto", 0.1, 1]
            }        
        ```
- Neural Network
    - Tuning Parameters
        ```python
            # suggest_int: random search int
            # suggest_catogorical: random pick a choice given
            # seggest_float: random search float 
            params = {
                "layers": trial.suggest_int("n_layers", 1, 10),
                "activation function": trial.suggest_categorical("active", ["relu", None]),
                "hidden dim": trial.suggest_int("hidden_nodes{}".format(i), 4, 512),
                "Optimisor": trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
                "epochs": trial.suggest_int("epochs", 150, 300), 
                "regularisation": trial.suggest_float("reg_coef", 1e-5, 1e-1, log=True),
                "learning rate": trial.suggest_float("lr", 1e-5, 1e-1, log=True),
                "batch size": 25,

            }
        ```



## Result
### Automotive Industry
- Annualised mean, volitility, and Sharpe ratio of returns

    | Model      | Simple Lienar | Elastic Net | Decisoin Tree | Random Forest | Xgboost | SVM   | Deep Learning | Ensemble Voting |
    | -----      | ------------- | ------------| ------------- | --------------| --------| ----  | --------------| ----------------|
    | Mean       | 0.23          | 0.42        | -0.22         | 0.32          | 0.37    | -0.04 | 0.01          |      0.30       |
    |Volitility  | 0.11          | 0.24        | 0.22          | 0.20          | 0.28    | 0.17  | 0.16          |       0.13      |
    |Sharpe ratio| 1.95          | 1.69        | 0.99          | 1.65          | 2.06    | -0.22 | 0.08          |       2.33      |

- Deep Learning tuning trials 
    
    | Trials | Mean | Volitility | Sharpe Ratio | Running Time     |
    | ------ | ---- | ---------- | ------------ | ---------------- |
    | 1      | -0.05 | 0.12      | -0.4         | 60 min           |
    | 5      | 0.09 | 0.16       | 0.53         | 238 min          |
    | 20     | 0.08 | 0.15       | 0.55         | 600 min          |


> NOTE
- With linaer regression and all rolling data (automotive), try decide the bounadry to long or short by train data or test data, and add previous boundary to decide the very period's boundary.
- Use [ Test/No previous]

    | Return Mean / Vol / Sharpe | Decide with Train | Decide with Test |
    | ---------------------------| ----------------- |------------------|
    | With previous              | .26 / .19 / 1.35  | .27 / .22 / 1.22 |
    | No previous                |  .23 / .14 / 1.61 | .23 / .12 / 1.95 | 


