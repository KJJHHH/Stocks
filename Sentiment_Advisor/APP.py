import datetime, os
import requests, json, re
import torch
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
from datetime import datetime, time, timedelta
from googletrans import Translator

def translate(text):
    translator = Translator()
    translation = translator.translate(text, dest='en')
    return translation.text

def get_news_urls(keyword):
    yesterday = datetime.today().date() - timedelta(days=1)
    start = datetime.combine(yesterday, time.min)
    page, date_time = 1, None
    print(f'Retrieving news links published after {start} in page (https://money.udn.com/search/result/1001/{keyword}/[page])')
        
    links_all = {}
    while True:
        url = f'https://money.udn.com/search/result/1001/{keyword}/{page}'
        htmls = requests.get(url).content
        soup = BeautifulSoup(htmls, 'html.parser')            
        urls_class = soup.find_all('div', {'class': 'story__content'})
        if urls_class == []:
            print('No Content')
            break
        
        for div in urls_class:
            # Check if time < start
            date_time = datetime.strptime(div.find_all('time')[0].text, '%Y-%m-%d %H:%M')
            if date_time < start:
                return links_all
            
            # article_url
            link = div.find_all('a', href=True)[0]['href']
            links_all[link] = str(date_time)
            # print(links_all)            
        page += 1
    
    return links_all

def get_news_text(keyword ="友達", task = 'sa'):
    news_ch = {}
    news_en = {}
        
    # Get and store article url in 'filename_url' 
    links_all = get_news_urls(keyword)
    
    for link, time in tqdm(links_all.items()):
        """
        Format for store article: {'link': [time, title, content]}
        """        
        # Get the htmls
        htmls = requests.get(link).content
        soup = BeautifulSoup(htmls, 'html.parser')
        
        # Get title, content
        try:
            'Something went wrong here for little news, do not know why'
            title = soup.find_all('h1', {"article-head__title"})[0].text
            content = soup.find_all('section', {"article-body__editor"})
        except:
            news_ch[link] = [time, None, None]
            news_en[link] = [time, None, None]
            continue
        
        # Concat and translate
        content_article = ''
        
        for con in content:
            con_ = con.text
            if '延伸閱讀' in con_:
                continue
            content_article += con_
            
        if content_article == '':
            continue
        
        # Store the data
        news_ch[link] = [time, title, content_article]
        
        # Sentiment analysis need english news
        # To find what news got translate error just find what news in ch but not in en
        if task == 'st':
            try:
                news_en[link] = [time, translate(title), translate(content_article)]
            except:
                print('Translate error')

    print('Finish Scraping')
    return news_ch, news_en

def labelling(price):
    if price > .004:
        return 1
    elif price < -.004:
        return 2
    else:
        return 0

def senti(news):
    from transformers import BertTokenizer, BertForSequenceClassification
    
    # Get the concated titles for yesterday's news
    title_concat = ''
    for link, infos in tqdm(news.items()):
        time, title, content = infos
        title_concat += title + '\n' 
    
    # Load model from checkpoints
    ckpt_dir = 'finbert-checkpoints'
    ckpts = []
    for ckpt in os.listdir(ckpt_dir):
        if (ckpt.startswith("checkpoint-")):
            ckpts.append(ckpt)

    ckpts = sorted(ckpts, key = lambda ckpt: int(ckpt.split("-")[-1]))        
    id_of_ckpt_to_use = -1 
    ckpt_name = os.path.join(ckpt_dir, ckpts[id_of_ckpt_to_use])
    model_name = ckpt_name # Pretrained: 'yiyanghkust/finbert-tone'
    cache_dir = './cache'
    tokenizer = BertTokenizer.from_pretrained(
        'yiyanghkust/finbert-tone',
        cache_dir=cache_dir,
    )

    model = BertForSequenceClassification.from_pretrained(
        model_name,
        device_map={'': 0},  # 設定使用的設備，此處指定為 GPU 0
        cache_dir=cache_dir
    )

        # Tokenize
    
    # Inference
    inputs = tokenizer(
        title_concat,
        return_tensors='pt',  # return PyTorch tensors
        padding=True,
        truncation=True,
        max_length=512  # adjust according to your needs
        )
    inputs = {k: v.cuda() for k, v in inputs.items()}
    logits = model(**inputs)[0]
    
    return torch.argmax(logits, dim=-1)

def evaluate(model, tokenizer, instruction, generation_config, max_len, input="", verbose=True):
    """
    (1) Goal:
        - This function is used to get the model's output given input strings

    (2) Arguments:
        - instruction: str, description of what you want model to do
        - generation_config: transformers.GenerationConfig object, to specify decoding parameters relating to model inference
        - max_len: int, max length of model's output
        - input: str, input string the model needs to solve the instruction, default is "" (no input)
        - verbose: bool, whether to print the mode's output, default is True

    (3) Returns:
        - output: str, the mode's response according to the instruction and the input

    (3) Example:
        - If you the instruction is "ABC" and the input is "DEF" and you want model to give an answer under 128 tokens, you can use the function like this:
            evaluate(instruction="ABC", generation_config=generation_config, max_len=128, input="DEF")

    """
    # construct full input prompt
    prompt = f"""\
[INST] <<SYS>>
You are a helpful assistant and good at writing Tang poem. 你是一個樂於助人的助手且擅長寫唐詩。
<</SYS>>

{instruction}
{input}
[/INST]"""
    # 將提示文本轉換為模型所需的數字表示形式
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    # 使用模型進行生成回覆
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=max_len,
    )
    # 將生成的回覆解碼並印出
    for s in generation_output.sequences:
        output = tokenizer.decode(s)
        output = output.split("[/INST]")[1].replace("</s>", "").replace("<s>", "").replace("Assistant:", "").replace("Assistant", "").strip()
        if (verbose):
            print(output)

    return output

def importance():
    # delete not important news
    """
    Need to finetune another model
    """
    pass

def agent(company_name, news):
    import numpy as np
    from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
    from transformers import GenerationConfig
    from peft import (
        prepare_model_for_int8_training,
        LoraConfig,
        get_peft_model,
        get_peft_model_state_dict,
        prepare_model_for_kbit_training,
        PeftModel
    )

    # Load model from checkpoints
    ckpt_dir = 'taide-checkpoints-exp1'
    ckpts = []
    for ckpt in os.listdir(ckpt_dir):
        if (ckpt.startswith("checkpoint-")):
            ckpts.append(ckpt)

    ckpts = sorted(ckpts, key = lambda ckpt: int(ckpt.split("-")[-1]))
    id_of_ckpt_to_use = -1
    ckpt_name = os.path.join(ckpt_dir, ckpts[id_of_ckpt_to_use])
    model_name = './TAIDE-LX-7B-Chat'
    cache_dir = './cache'
    
    max_len = 128*(4)   # 生成回復的最大長度
    temperature = 0.1  # 設定生成回覆的隨機度，值越小生成的回覆越穩定
    top_p = 0.3  # Top-p (nucleus) 抽樣的機率閾值，用於控制生成回覆的多樣性
    top_k = 5 # 調整Top-k值，以增加生成回覆的多樣性和避免生成重複的詞彙 

    cache_dir = "./cache"  # 設定快取目錄路徑
    no_repeat_ngram_size = 3  # 設定禁止重複 Ngram 的大小，用於避免生成重複片段

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # 使用 tokenizer 將模型名稱轉換成模型可讀的數字表示形式
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        quantization_config=nf4_config
    )

    # 從預訓練模型載入模型並設定為 8 位整數 (INT8) 模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=nf4_config,
        device_map={'': 0},  # 設定使用的設備，此處指定為 GPU 0
        cache_dir=cache_dir
    )

    # 從指定的 checkpoint 載入模型權重
    model = PeftModel.from_pretrained(model, ckpt_name, device_map={'': 0})

    # 設定生成配置，包括隨機度、束搜索等相關參數
    generation_config = GenerationConfig(
        do_sample=True,
        temperature=temperature,
        num_beams=1,
        top_p=top_p,
        top_k=top_k,
        no_repeat_ngram_size=no_repeat_ngram_size,
        pad_token_id=2
    )    
    
    instruct = [
        f"你是一位資深金融投資者，請根據新聞內容提到公司{company_name}可能存在的發展與風險，請給出：\
        1. 一新聞總結，\
        2. 投資獲利能力評級(投資獲利能立：1到10分，10分最高，1分最低)，\
        3. 風險評級(風險：1到10分，10分最高風險，1分最低風險)。新聞內容：",  
        ]
    
    
    # Loop Method (How to use loop to replace a recurrence) 
    """
    Max number of news is max_news_accepted**2   
    """
    output_all = ''
    output_tmp = ''
    max_news_accepted = int(2048/max_len)
       
    keys = list(news.keys())[:max_news_accepted**2] 
    news = {k: news[k] for k in keys}
    for count, (link, infos) in enumerate(tqdm(news.items())):
        time, title, content = infos
    
        input = [content]
        if count % max_news_accepted == 0:
            output_all += evaluate(model, tokenizer, instruct[0], generation_config, max_len, output_tmp, verbose = False) + '\n'
            output_tmp = ''
        output_tmp += evaluate(model, tokenizer, instruct[0], generation_config, max_len, input[0], verbose = False) + '\n'
        
    print(f'Output: {evaluate(model, tokenizer, instruct[0], generation_config, max_len, output_all, verbose = False)}')
    
    
    """
    # Recurrence method
    def summarise(content_list):
        '''
        - input: news (list)
        - ouptut: concatenated summarised news 
        '''
        max_news_accepted = int(2048/max_len)
        
        if len(content_list) > max_news_accepted:
            content_summary = ''
            for i in range(int(np.ceil(len(content_list)//max_news_accepted))):
                summaries = summarise(content_list[i*max_news_accepted:(i+1)*max_news_accepted]) + '\n'
                content_summary += evaluate(model, tokenizer, instruct[0], generation_config, max_len, summaries, verbose = False) + '\n'
                torch.cuda.empty_cache()
            
        else:
            content_summaries = ''
            for content in content_list:
                content_summaries += evaluate(model, tokenizer, instruct[0], generation_config, max_len, content, verbose = False) + '\n'
                torch.cuda.empty_cache()
            return content_summaries
        return content_summary
    
    content_all = []
    for link, infos in news.items():
        time, title, content = infos
        content_all.append(content)
    
    output = summarise(content_all)
    print(f'Output: {output}')
    """
        
    


if __name__ == "__main__":
    task = input('Input "sa" for Stock Agent or "st" for Sentiment Analysis)?') or 'sa'
    keyword = input('Enter the company name:') or '友達'
    company_name = '友達'
    print(f'Keyword: {keyword}, Company name: {company_name}')
    
    # Get news
    news_ch, news_en = get_news_text(keyword)
    
    if task == "sa":
        # Model-Taide: Finetuned finance model, text to text,  with chinese llm
        agent(company_name, news_ch)
        
    elif task == "st":
        # Model-finBert: Finetuned finance model, text classification, with english llm
        senti_label = senti(news_en)
        senti_label = 'Positive' if senti_label == 1 else "Negative" if senti_label == 2 else "Neutral"
        print(f"Sentiment analysis for today's news: {senti_label}")
    
    