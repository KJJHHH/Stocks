# Transformer Models to Predict Stock 
# Data Preprocess

1. Download data with Open, Close, High, Low, Volume
2. Transform to the percentile change for each date as do, dc, …
3. Normalise with train set. Set another column ‘ Close_origin’ as the Close before normalise
4. Select do, dc, dh, dl, dv, and Close (Normalise)
5. Use window 100, i.e. predict with last 100 dates’ data, as X value
6. Predict the next dates’ do, dc6. 
### In Transformer Encoder Decoder
1. Src: whole batch x_train
2. tgt: batched x_train

# Models and Training
Two models
- [ ] Decoder-only 
- [x] Transformer-Encoder-Decoder
```python
L: Total length
P: Patch numbers = L - S + 1
S: Sequence length for each patch
D: Input dim
B: Batch size
```
### Decoder-Only
- Positional Encoding
- Transformer Decoder
    - Input (B, S, D) Output (B, S, D)
- Fully Connected Layer
### Transformer
- Positional Encoding
- Use Convolution as encoder to map
    - 💡 3 x 3 Convolution * 3 with residual connection last convolution encoding output   
    - tgt from (B, S, D) to (B, 1, D)  
    - src from (T, S, D) to (B, T, D)      
- Transformer
    - Transformer Encoder: Src → Memory (B, T, D) 
    - Transformer Decoder: Memory, Tgt → output (B, 1, D)
- Linear map (B, D) to (B, 2), as do and dc

# Experiments
- Strategy

    Buy if: (predicted next day’s Close - Open) > today’s Close * 0.004
    |             | Buy and Hold | Decoder-Only | Transformer |
    | ----------- | ------------ | ------------ | ----------- |
    | Final Asset |     0.89     |    1.08      |     1.13    |
    
- Plot Results: Asset
    - Decoder-Only
    ![alt text](https://github.com/KJJHHH/Stocks/blob/main/Stock_Individual_Transformer/Model-Decoder/Model_Result/Transformer-Decoder-Only_class2_5871_backtest.png)
    - Transformer
    ![alt text](https://github.com/KJJHHH/Stocks/blob/main/Stock_Individual_Transformer/Model-Transformer/Model_Result/Transformer-Encoder-Decoder_class2_5871_backtest.png)
