# SENTIMENT ANALYSIS AND STOCK AGENT LM
## 1. Goal
- Build sentiment analysis to trade
- Build personal stock agent with LLM
## 2. Data
### 2.1. News
- 經濟日報(UDN)
    - Text and Title
### 2.2. Price
- Yahoo finance
### Preprocessing
- Translate Chinese data to English data: finbert model needs english input.


## 3. Sentiment Analysis
<div style="border-left: 4px solid #2196F3; background-color: #E3F2FD; padding: 10px; margin-bottom: 10px;">
  <strong>finBert: </strong> <a href="https://huggingface.co/yiyanghkust/finbert-tone">yiyanghkust/finbert-tone</a>
</div>

### 3.2. TW Stock - 0050
<details> 
<summary>Backtest Result</summary>

- Data size: 
- From pretrained model
![alt text](finbert-backtest-result/0050-pretrain.png)
- Finetune with 0050 related news data (news keyword: ETF)
![alt text](finbert-backtest-result/0050-finetune.png)
</details>


### 3.3. TW Stock - 2409
<details> 
<summary>Backtest Result</summary>

- Data size: 
- From pretrained model
![alt text](finbert-backtest-result/2409-pretrain.png)
- Finetune with 2409 related news data
![alt text](finbert-backtest-result/2409-finetune.png)
- Finetune with 2049 related and ETF news data
![alt text](finbert-backtest-result/2409-finetune-multidata.png)
</details>



## 4. Personal stock agent
### 4.1. Model Pretrained
- Taide: [taide/Llama3-TAIDE-LX-8B-Chat-Alpha1](https://huggingface.co/taide/Llama3-TAIDE-LX-8B-Chat-Alpha1)
- Chinese Mediatek Model
    - Pretrained:
    - Finetune
        - Text to text
        - Text to sentiment
NOTE:
- Summarise model: llama3

## 5. Possible Improvement
- In minutes
