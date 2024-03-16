# Computer Vision Methods to Predict Stock 
## Experiments
Asset using transformer model comparing to buy-and-hold strategy
- Conformer: [Asset](https://github.com/KJJHHH/Stocks/blob/main/Stock_Individual_CV/Models/Model_Result/Conformer-CNN_class2_5871_backtest.png) | 中租，5871
- Resnet: [Asset](https://github.com/KJJHHH/Stocks/blob/main/Stock_Individual_CV/Models/Model_Result/ResNet_class2_5871_backtest.png) | 中租，5871
- Conformer Resnet: [Asset](https://github.com/KJJHHH/Stocks/blob/main/Stock_Individual_CV/Models/Model_Result/Conformer-Resnet_class2_5871_backtest.png) | 中租，5871
- VisionTransformer: [Asset](https://github.com/KJJHHH/Stocks/blob/main/Stock_Individual_CV/Models/Model_Result/Vision-Transformer_class2_5871_backtest.png) | 中租，5871
### Models
- [x] ResNet
- [x] Conformer
- [x] Conformer + ResNet (ConRes)
- [x] VisionTransformer
- [ ] Pretrained VisionTransformer
### Data Preprocess
1. Download data with Open, Close, High, Low, Volume
2. Transform to the percentile change for each date
3. Use window 100, i.e. predict with last 100 dates’ data, as X value
4. Expand X shape (5, 100) to (5, 100, 100) and cosine values
5. Predict the next dates’ Open change and Close change (percentile)



