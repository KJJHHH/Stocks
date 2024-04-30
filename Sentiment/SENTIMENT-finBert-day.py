import datetime
from datetime import timedelta
import json
import pickle
import json
import logging
import warnings
import os
from tqdm import tqdm
from data.utils import *
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
warnings.filterwarnings('ignore')


def sentiment():
    # --------
    # Data
    assert os.path.exists(filename_news) and os.path.exists(filename_price), \
        f"Error: data file does not exist"
    with open(filename_news, 'r', encoding='utf-8') as f:
        data_news = json.load(f)
    with open(filename_price, 'rb') as f:
        data_price = pickle.load(f)
    
    data_price = data_price.pct_change().shift(-1).dropna()
    data_price.index = data_price.index.date
    data_price = data_price.to_dict()
    
    # ---------
    # Load scored data
    """
    - link_score = {link: [date, price, score_title, score_text]}
    """
    if os.path.exists(filename_result):
        print('Load the precious scored news ')
        with open(f'results/{MEDIA}_{KEYWORD}.json', 'r', encoding='utf-8') as f:
            link_score = json.load(f)
    else:
        link_score = {}

    # --------
    # Models
    # 1. 'yiyanghkust/finbert-tone'
    # 2. 'ProsusAI/finbert'
    finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
    tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
    nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)   
    def scoring(nlp, text):
        return nlp(text)
    
    # --------
    # Scoring
    sign = {'Positive': 1, 'Negative': -1, 'Neutral': 0.3}

    for link, info in tqdm(data_news.items()):
        if link in link_score or link in link_score:
            continue
        time, title, text = info[0], info[1], info[2]
        
        # The news' date: if not in price date, add 1 day until in
        date_news = datetime.datetime.strptime(time[:10], '%Y-%m-%d').date()
        if date_news not in data_price:
            if date_news > max(data_price.keys()):
                continue
            else:
                while date_news not in data_price:
                    date_news += timedelta(days=1)          
                    
        # The date's price
        date_price = data_price[date_news]
        
        # =======
        # Compute score for each day            
        """
        1. Summary for text
        2. Compute score for each article title and text
        3. link_score = {link: [date, price, score_title, score_text]}
        NOTE:
        - If none scores is added to score, the score == None
        """        
        print('-------')
        try: # Text
            summarise_text = summarise(text)
            result = scoring(nlp, summarise_text)
            score_text = result[0]['score'] * sign[result[0]['label']]
            print(result[0]['label'])
        except Exception as e:
            score_text = None
            logging.error(f'Error occur with text {link} | Error message: {e}')   
        try: # Title
            result = scoring(nlp, title)
            score_title = result[0]['score'] * sign[result[0]['label']]
            print(result[0]['label'])
        except Exception as e:
            score_title = None
            logging.error(f'Error occur with title {link} | Error message: {e}')
        
        link_score[link] = [str(date_news), date_price, score_title, score_text]
        with open(filename_result, 'w') as f:
            json.dump(link_score, f)
    
if __name__ == '__main__':
    # Set
    MEDIA = 'UDN'
    KEYWORD = 'ETF'
    STOCK = '0050.TW'
    DAY_RANGE = '1d'
    filename_news = f'data/TW/NEWS_{MEDIA}_{KEYWORD}.json'
    filename_price = f'data/TW/PRICE_{STOCK}_{DAY_RANGE}.pickle'
    filename_result = f'results/{MEDIA}_{KEYWORD}.json'
    filename_error = f'logs/ERROR_{MEDIA}_{KEYWORD}'
    
    # Logging
    if os.path.exists(filename_error):
        os.remove(filename_error)
    logging.basicConfig(filename=filename_error, level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run
    sentiment()

