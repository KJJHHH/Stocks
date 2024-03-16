# Transformer Models to Predict Stock 
# Data Preprocess

1. Download data with Open, Close, High, Low, Volume
2. Transform to the percentile change for each date as do, dc, …
3. Normalise with train set.
4. Select do, dc, dh, dl, dv, and Close (Normalise)
5. Use window 100, i.e. predict with last 100 dates’ data, as X value
6. Predict the next dates’ do, dc6. 

# Models and Training
## Two models 
- [x] Transformer-Encoder-Decoder
- [ ] Decoder-only
```python
L: Total length 
P: Patch numbers = L - S + 1
S: Sequence length for each patch
D: Input dim
B: Batch size
```
### Transformer Model Structure
```python
# Input: src shape (B, )
```
- Positional Encoding   
- Transformer
    - Transformer Encoder: Src → Memory (B, T, D) 
    - Transformer Decoder: Memory, Tgt → output (B, 1, D)
- Linear transformation (B, D) to (B, 2), as do and dc

# Experiments
### Final Asset and Loss
- 5871: 
[Asset](https://github.com/KJJHHH/Stocks/blob/main/Stock_Individual_Transformer/Model-Transformer/Model_Result/TransEnDecoder-Window10-EL1-DL1-Hid128-NHead1_class2_5871_backtest.png)
[Loss](https://github.com/KJJHHH/Stocks/blob/main/Stock_Individual_Transformer/Model-Transformer/Model_Result/TransEnDecoder-Window10-EL1-DL1-Hid128-NHead1_class2_2884_loss.png)
- 2454: 
[Asset](https://github.com/KJJHHH/Stocks/blob/main/Stock_Individual_Transformer/Model-Transformer/Model_Result/TransEnDecoder-Window10-EL1-DL1-Hid128-NHead1_class2_2454_backtest.png)
[Loss](https://github.com/KJJHHH/Stocks/blob/main/Stock_Individual_Transformer/Model-Transformer/Model_Result/TransEnDecoder-Window10-EL1-DL1-Hid128-NHead1_class2_2454_loss.png)
- 2884: 
[Asset](https://github.com/KJJHHH/Stocks/blob/main/Stock_Individual_Transformer/Model-Transformer/Model_Result/TransEnDecoder-Window10-EL1-DL1-Hid128-NHead1_class2_2884_backtest.png)
[Loss](https://github.com/KJJHHH/Stocks/blob/main/Stock_Individual_Transformer/Model-Transformer/Model_Result/TransEnDecoder-Window10-EL1-DL1-Hid128-NHead1_class2_5871_loss.png)