# SENTIMENT ANALYSIS WITH ETF NEWS
- [ ] finBert: Pretrained model
- [ ] 
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
### Finetune data
- CH: [CFGPT](https://github.com/TongjiFinLab/CFGPT?tab=readme-ov-file)
- EN: ?
## Model
### finBert: [yiyanghkust/finbert-tone](https://huggingface.co/yiyanghkust/finbert-tone)
- [ ] Pretrained model
### Taide: [taide/Llama3-TAIDE-LX-8B-Chat-Alpha1](https://huggingface.co/taide/Llama3-TAIDE-LX-8B-Chat-Alpha1)
- [ ] Pretrained model
- [ ] Finetune model