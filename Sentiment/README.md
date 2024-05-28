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
### 2.3. Preprocessing
- Translate Chinese data to English data: finbert model needs english input.


## 3. Sentiment Analysis
<div style="border-left: 4px solid #2196F3; background-color: #E3F2FD; padding: 10px; margin-bottom: 10px;">
  <strong>Model pretrained: finBert-</strong> <a href="https://huggingface.co/yiyanghkust/finbert-tone">yiyanghkust/finbert-tone</a>
</div>

|           | Pretrain   | Finetune                                               |
| --------- | ---------- | ------------------------------------------------------ |
| data size |            |                                                        |
| Asset     |![alt text](finbert-backtest/0050-pt.png)| ![alt text](finbert-backtest/0050-ft.png) |

### 3.2. TW Stock - 0050
- Data size: 
- Backtest result
    - [From pretrained model](https://github.com/KJJHHH/Stocks/blob/main/Sentiment/finbert-backtest-result/0050-pretrain.png)
    - [Finetune with 0050 related news data (news keyword: ETF)](https://github.com/KJJHHH/Stocks/blob/main/Sentiment/finbert-backtest-result/0050-finetune.png)

### 3.3. TW Stock - 2409
- Data size: 
- Backtest result
    - [From pretrained model](https://github.com/KJJHHH/Stocks/blob/main/Sentiment/finbert-backtest-result/2409-pretrain.png)
    - [Finetune with 2409 related news data](https://github.com/KJJHHH/Stocks/blob/main/Sentiment/finbert-backtest-result/2409-finetune.png)
    - [Finetune with 2049 related and ETF news data](https://github.com/KJJHHH/Stocks/blob/main/Sentiment/finbert-backtest-result/2409-finetune-multidata.png)


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
