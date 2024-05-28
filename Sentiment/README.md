# SENTIMENT ANALYSIS AND STOCK AGENT LM
## Goal
- Build sentiment analysis to trade
- Build personal stock agent with LLM
## Data
### News
- 經濟日報(UDN)
    - Text and Title
### Price
- Yahoo finance
### Preprocessing
- Translate Chinese data to English data: finbert model needs english input.


# Sentiment Analysis
### Model Pretrained
- finBert: [yiyanghkust/finbert-tone](https://huggingface.co/yiyanghkust/finbert-tone)

### TW Stock - 0050
<details> 
<summary>Backtest Result</summary>

- Data size: 
- From pretrained model
![alt text](finbert-backtest-result/0050-pretrain.png)
- Finetune with 0050 related news data (news keyword: ETF)
![alt text](finbert-backtest-result/0050-finetune.png)
</details>


### TW Stock - 2409
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



# Personal stock agent
### Model Pretrained
- Taide: [taide/Llama3-TAIDE-LX-8B-Chat-Alpha1](https://huggingface.co/taide/Llama3-TAIDE-LX-8B-Chat-Alpha1)

- Chinese Mediatek Model
    - Pretrained:
    - Finetune
        - Text to text
        - Text to sentiment
NOTE:
- Summarise model: llama3

## Improvement
- In minutes
