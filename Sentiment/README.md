# SENTIMENT ANALYSIS WITH ETF NEWS
## Data
- Date/Minutes data
### News
- 經濟日報(UDN)
- Process
    1. Scrape the news' time, title, and content
    2. Translate to English for finBert model
    3. Summarise the content. Summarise model
        - EN: "facebook/bart-large-cnn"
        - CH: ? 
### Price
- Yahoo finance
## Model
### finBert: [yiyanghkust/finbert-tone](https://huggingface.co/yiyanghkust/finbert-tone)
Finance finetuned model
- [x] Pretrained model
- [x] Finetuned - Text to Score 
### Taide: [taide/Llama3-TAIDE-LX-8B-Chat-Alpha1](https://huggingface.co/taide/Llama3-TAIDE-LX-8B-Chat-Alpha1)
General pretrained model
- [x] Pretrained model
- [x] Finetune - Text to Text - LoRA
- [ ] Finetune - Text to Score - Full
### Finetune data
- CH
    - Text to text finetuning: [CFGPT](https://github.com/TongjiFinLab/CFGPT?tab=readme-ov-file) -> to traditional Chinese
    - Text to score finetuning
- EN
    - Text to score finetuning
## Method
- FinBert
    - Pretrained:
        1. Score the sentiment for each article
        2. Calculate the mean score for articles in the same date
    - Finetune:
        <aside>
        💡 NOTE: summarise a lot to fit the max input limit for summarise and finBert model
        </aside>
        1. Summarise articles
        2. Concatenate all articles in the same date
        3. Summarise again...
- Chinese Mediatek Model
    - Pretrained:
    - Finetune
        - Text to text
        - Text to sentiment
NOTE:
- Summarise model: llama3
