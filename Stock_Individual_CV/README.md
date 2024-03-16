# Computer Vision Methods to Predict Stock 
## Experiments
Asset using transformer model comparing to buy-and-hold strategy
- Conformer: [Asset]() | 中租，5871
- Resnet: [Asset]() | 中租，5871
### Data Preprocess
1. Download data with Open, Close, High, Low, Volume
2. Transform to the percentile change for each date
3. Use window 100, i.e. predict with last 100 dates’ data, as X value
4. Expand X shape (5, 100) to (5, 100, 100) and cosine values
5. Predict the next dates’ Open change and Close change (percentile)

### Models
- [x] ResNet
- [x] Conformer
- [x] Conformer + ResNet (ConRes)
- [x] VisionTransformer
- [ ] Pretrained VisionTransformer


