# Build Stock Portfolio 
<!---🦁🙉😹🧑💗🦁🏋🐱🖼️📝--->


Build stock portfolio by utilising multiple machine learning algorithm with rolling prediction for each month.

## Data and Preprocessing
- The data for training algorithms are downloaded from TEJ database, including **fundamental**, **chip**, and **Betas data**. Besides, we compute the **technical analysis** values include RSI, SMA, EMA, MACD, bband, KD, beta, willr, and bias with talib packages
- The date to update monthly portfolio for each month: The last revenue announce date for all companies. If any announcement date is later than 12th, delete the company for the month
- For each **rolling prediction**, use five-year training data to predict the adjacent month

## Models
- Multiple linear regression
- Elastic Net
- Decision tree
- Random forest
- Xgboost
- SVM
- Neural Network

## Result
 The portfolios are based on automotive, semi conductor, and TFT-LCD industry. 
- [x] automotive
- [ ] semi conductor
- [ ] TFT-LCD industry
 
### Automotive Industry
- Annualised returns mean, volitility, and Sharpe ratio 

    | Model      | Simple Lienar | Elastic Net | Decisoin Tree | Random Forest | Xgboost | SVM   | Deep Learning | Ensemble Voting |
    | -----      | ------------- | ------------| ------------- | --------------| --------| ----  | --------------| ----------------|
    | Mean       | 0.23          | 0.42        | -0.22         | 0.32          | 0.37    | -0.04 | 0.01          |      0.30       |
    |Volitility  | 0.11          | 0.24        | 0.22          | 0.20          | 0.28    | 0.17  | 0.16          |       0.13      |
    |Sharpe ratio| 1.95          | 1.69        | 0.99          | 1.65          | 2.06    | -0.22 | 0.08          |       2.33      |

- Tuning Deep Learning with Optuna
    
    | Trials | Mean | Volitility | Sharpe Ratio | Running Time     |
    | ------ | ---- | ---------- | ------------ | ---------------- |
    | 1      | -0.05 | 0.12      | -0.4         | 60 min           |
    | 5      | 0.09 | 0.16       | 0.53         | 238 min          |
    | 20     | 0.08 | 0.15       | 0.55         | 600 min          |



