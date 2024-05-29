# Sentiment Analysis Performance Comparison using FinBERT

## 1. Introduction

In this analysis, we evaluate the performance of sentiment analysis using the FinBERT model on two Taiwanese stocks: TW Stock 0050 and 2409. The study involves comparing the outcomes of pretraining and finetuning phases, accompanied by backtesting strategies to gauge asset performance.

Additionally, we aim to leverage the insights gained from sentiment analysis to build a personal investment advisor. This advisor will summarize news articles and provide simple insights to assist investors in making informed decisions.


## 2. Data Overview

- **TW Stock 0050**
  - **Pretraining Data**: Not applicable (N/A)
  - **Finetuning Data**: 6623 news articles
  
- **TW Stock 2409**
  - **Pretraining Data**: N/A
  - **Finetuning Data**:
    - Related Data Only: 1301 news articles
    - Multiple Data (Related and Unrelated): 1301 + 6623 news articles

## 3. Sentiment Analysis Performance
- <span style="color:   #4169E1;">Blue line</span>: trade with sentiment analysis.
- <span style="color: orange;">Orange line</span>: buy and hold.

```
### NOTE
- **Finetuning Strategies**:
  - **Related Data Only**: Finetune with news articles related to the stock.
  - **Multiple Data**: Finetune with both related and unrelated news articles.
- Some of the backtest results below shows flat line at the last few day since I have not update the news yet.
```

### 3.1. TW Stock 0050
- **Pretraining** vs. **Finetuning**

|       | Pretrain   | Finetune                                               |
|:-----:|:----------:|:------------------------------------------------------:|
| Data  |       -    |         6623                                           |
| Asset |![Pretraining Performance](finbert-backtest/0050-pt.png)| ![Finetuning Performance](finbert-backtest/0050-ft.png) |

### 3.2. TW Stock 2409

|       | Pretrain   |   Finetune - Related Data Only | Finetune - Multiple Data  |
|:-----:|:----------:|:------------------------------:|:-------------------------:|
| Data  |     -      |           1301                 |    1301 + 6623            |
| Asset | ![Pretraining Performance](finbert-backtest/2409-pt.png) | ![Related Data Only](finbert-backtest/2409-ft.png) | ![Multiple Data](finbert-backtest/2409-ft-m.png) |

### 3.3 TW Stock 2330

|       | Pretrain   |   Finetune - Related Data Only | Finetune - Multiple Data  |
|:-----:|:----------:|:------------------------------:|:-------------------------:|
| Data  |     -      |           1301                 |    1301 + 6623            |
| Asset | ![Pretraining Performance](finbert-backtest/2330-pt.png) | ![Related Data Only](finbert-backtest/2330-ft.png) | ![Multiple Data](finbert-backtest/2330-ft-m.png) |


### 3.4 TW Stock 2454

|       | Pretrain   |   Finetune - Related Data Only | Finetune - Multiple Data  |
|:-----:|:----------:|:------------------------------:|:-------------------------:|
| Data  |     -      |           1301                 |    1301 + 6623            |
| Asset | ![Pretraining Performance](finbert-backtest/2454-pt.png) | ![Related Data Only](finbert-backtest/2454-ft.png) | ![Multiple Data](finbert-backtest/2454-ft-m.png) |


### 3.5 TW Stock 5871

|       | Pretrain   |   Finetune - Related Data Only | Finetune - Multiple Data  |
|:-----:|:----------:|:------------------------------:|:-------------------------:|
| Data  |     -      |           1301                 |    1301 + 6623            |
| Asset | ![Pretraining Performance](finbert-backtest/5871-pt.png) | ![Related Data Only](finbert-backtest/5871-ft.png) | ![Multiple Data](finbert-backtest/5871-ft-m.png) |


## 4. Personal stock agent
### 4.1. Model Pretrained
- Taide: [taide/TAIDE-LX-7B-Chat](https://huggingface.co/taide/TAIDE-LX-7B-Chat)


## 5. Future Directions
- Add more news data: including the keyword for industry.
