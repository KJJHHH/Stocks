import datetime, os
import requests, json, re
import torch
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
from datetime import datetime, time, timedelta
from googletrans import Translator
from transformers import BertTokenizer, BertForSequenceClassification

def translate(text):
    translator = Translator()
    translation = translator.translate(text, dest='en')
    return translation.text

def get_news_urls(KEYWORD):
    yesterday = datetime.today().date() - timedelta(days=1)
    start = datetime.combine(yesterday, time.min)
        
    links_all = {}
    page, date_time = 1, None
        
    print('-----------------')
    print(f'Scrape news after {start}')
    print('Start getting article links: ', f'https://money.udn.com/search/result/1001/{KEYWORD}/{page}')
    
    while True:
        url = f'https://money.udn.com/search/result/1001/{KEYWORD}/{page}'
        htmls = requests.get(url).content
        soup = BeautifulSoup(htmls, 'html.parser')            
        urls_class = soup.find_all('div', {'class': 'story__content'})
        if urls_class == []:                                               
            break
        
        for div in urls_class:
            # Check if time < start
            date_time = datetime.strptime(div.find_all('time')[0].text, '%Y-%m-%d %H:%M')
            if date_time < start:
                return links_all
            
            # article_url
            link = div.find_all('a', href=True)[0]['href']
            if link in links_all:
                continue
            
            links_all[link] = str(date_time)
            # print(links_all)            
        page += 1
        
    return links_all

def get_news_text(KEYWORD ="友達"):
    news_ch = {}
    news_en = {}
        
    # Get and store article url in 'filename_url' 
    links_all = get_news_urls(KEYWORD)
    
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
            try:
                content_article += con_
            except:
                print('translate error')
        if content_article == '':
            continue
        
        # Store the data
        news_ch[link] = [time, title, content_article]
        try:
            # To find translate error just find what news in ch but not in en
            news_en[link] = [time, translate(title), translate(content_article)]
        except:
            print('Translate error')
            
    return news_ch, news_en

def labelling(price):
    if price > .004:
        return 1
    elif price < -.004:
        return 2
    else:
        return 0

def senti(news):
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

def agent(news):
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
    
    # Get the concated titles for yesterday's news
    title_concat = ''
    for link, infos in tqdm(news.items()):
        time, title, content = infos
        title_concat += title + '\n'

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
    
    max_len = 128   # 生成回復的最大長度
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
        # top_k=top_k,
        no_repeat_ngram_size=no_repeat_ngram_size,
        pad_token_id=2
    )
    
    
    instruct = [
        "你是一位資深金融投資者，請根據新聞內容提到的可能存在的投資獲利能力與風險進行分析，給出一個新聞結論，\
        並給出適合該公司的投資獲利能力評級(投資獲利能立：1到10分，10分最高，1分最低)，\
        以及風險評級(風險：1到10分，10分最高風險，1分最低風險)。研報內容：",  
        ]
    input = [title_concat]
    output = evaluate(model, tokenizer, instruct[0], generation_config, max_len, input[0], verbose = False)
    print(f'Output: {output}')
        




if __name__ == "__main__":
    task = input('Do you want "Stock Agent" (enter "sa") or "Sentiment Analysis" (enter "st")?')
    
    news_ch, news_en = get_news_text('友達')
    if task == "sa":
        agent(news_en)
        
    elif task == "st":
        senti_label = senti(news_en)
        senti_label = 'Positive' if senti_label == 1 else "Negative" if senti_label == 2 else "Neutral"
        print(f"Sentiment analysis for today's news: {senti_label}")
    
    