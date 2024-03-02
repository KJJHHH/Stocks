# Transformer Models to Predict Stock 
### Data Preprocess
1. Download data with Open, Close, High, Low, Volume
2. Transform to the percentile change for each date as do, dc, ...
3. Normalise with train set
4. Select do, dc, dh, dl, dvm and Close
5. Use window 400, i.e. predict with last 400 dates’ data, as X value
6. Predict the next dates’ (Open - Close)/Close

### Models
- Transformer Decoder-Only
- Transoformer EncodDecoder

### Experiments
- Strategy\
Buy if: (predicted next day’s Close - Open) > today’s Close * 0.004
- Accuracy: The sign accuracy of predicted and true values

|             | Buy and Hold | Decoder-Only | Transformer |
| ----------- | ------------ | ------------ | ----------- |
| Accuracy    |              |    0.441     |     0.445   |
| Final Asset |     0.89     |    1.08      |     1.13    |
- Plot Results
    - Decoder-Only
    ![alt text](https://github.com/KJJHHH/Stocks/blob/main/Stock_Individual_Transformer/Model-Decoder/Model_Result/Transformer-Decoder-Only_class2_5871_backtest.png)
    - Transformer
    ![alt text](https://github.com/KJJHHH/Stocks/blob/main/Stock_Individual_Transformer/Model-Transformer/Model_Result/Transformer-Encoder-Decoder_class2_5871_backtest.png)

