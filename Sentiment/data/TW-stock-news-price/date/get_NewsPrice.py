from bs4 import BeautifulSoup
import torch
import requests
import datetime
from datetime import time
from time import sleep
import json
import os
import sys
sys.path.append('../../')
from utils import *

# -------------
# PRICE
# NEWS
def get_price(TICKERS = '00940.TW', INTERVAL = '1d'):
    import yfinance as yf
    import pickle
    
    now = datetime.datetime.now()
    start = datetime.datetime.strptime('2015-01-01', "%Y-%m-%d")
    price = yf.download(TICKERS,start=start, end=now, interval=INTERVAL)['Adj Close']
    with open(f'PRICE_{TICKERS}_{INTERVAL}.pickle', 'wb') as f:
        pickle.dump(price, f)

def get_news_anue(NOW, KEYWORD ="大盤", start = None):
    media = 'ANUE'
    start = datetime.datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
    page = 1
    filename = f'NEWS_{media}_{KEYWORD}.json'
    data = load_json(filename) if os.path.exists(filename) else {}
    
    while True:
        json_data = requests.get(
            f'https://ess.api.cnyes.com/ess/api/v1/news/keyword?q={KEYWORD}&page={page}').json()
        print(f'https://ess.api.cnyes.com/ess/api/v1/news/keyword?q={KEYWORD}&page={page}')
        items=json_data['data']['items']
        for item in items:
            # Article url: https://news.cnyes.com/news/id/5537360
            # Check if news already stored
            if f'{item["newsId"]}' in data:
                if NOW:
                    print("No latest news")
                    return 
                else: 
                    continue
                
            # Ids, title, and publish time
            news_id = item["newsId"]
            title = item["title"]
            publish_at = item["publishAt"]
            
            # UTC time format
            utc_time = datetime.datetime.utcfromtimestamp(publish_at)
            formatted_date = utc_time.strftime('%Y-%m-%d-%H:%M:%S')
            
            # Return if news is before the earliest date news
            if utc_time < start:
                return data
            
            # Get the contents
            url = requests.get(f'https://news.cnyes.com/'
                                f'news/id/{news_id}').content
            soup = BeautifulSoup(url, 'html.parser')
            p_elements=soup.find_all('p')
            
            # Paragraph
            p=''
            for paragraph in p_elements:
                paragraph_ = paragraph.get_text()
                remove_l = ['FB 粉絲團', 'LINE', 'Line', 'line', 'YT', '準時接受盤中快訊', '無不當之財務利益關係',
                            '請收看我的', '歡迎大家轉貼本文', '產業趨勢報告', 'https://reurl', 
                            '客服專線', 'https://lin', 'https://youtu', '更完整分析', '下一篇',
                            '上一篇', '近5日籌碼']
                if len(paragraph_) < 20:
                    continue
                if any(element in paragraph_ for element in remove_l): 
                    continue
                
                try:    
                    # paragraph_ = summarise(paragraph_)
                    p+=paragraph_
                    p+='\n'
                    torch.cuda.empty_cache()
                except:
                    pass
            
            print('-----')
            print(item["newsId"])
            print(p)
            data[news_id] = [KEYWORD, formatted_date ,title, p]
            # Store data
            store_json(filename, data)
            
        # update page
        page += 1

def get_news_udn(KEYWORD ="大盤",  UPDATE_NEWS = False, start = None):
    """
    - uls: [links]
    - Data: {link: [date, title, content]}
    """
    from tqdm import tqdm
    media = 'UDN'
    start = datetime.datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
    
    """
    filename_ch: chinese data
    filename_en: english data
    filename_url: article url
    """
    filename_ch = f'NEWS_{media}_{KEYWORD}_CH.json'
    filename_en = f'NEWS_{media}_{KEYWORD}_EN.json'
    filename_url = f'NEWS_{media}_{KEYWORD}_URL.json'
    
    links_all = load_json(filename_url) if os.path.exists(filename_url) else {}
    news_ch = load_json(filename_ch) if os.path.exists(filename_ch) else {}
    news_en = load_json(filename_en) if os.path.exists(filename_en) else {}
    
    def get_urls():
        """
        Format for store article urls: {'link': time}, {'link': time}
        """
        # Get udn urls       
        # 1. Load the stored data (past data) if data exist
        # 2. Get the latest data links 
        page, time = 1, None
        print(f'Scrape news after {start}')
        print('-----------------')
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
                time = datetime.datetime.strptime(div.find_all('time')[0].text, '%Y-%m-%d %H:%M')
                if time < start:
                    return 
                
                # article_url
                link = div.find_all('a', href=True)[0]['href']
                if link in links_all:
                    continue
                
                links_all[link] = str(time)
                store_json(filename_url, links_all)
                
            page += 1
            print(f'\r---> article links at {time}, page {page}', end='', flush=True)
        
    # Get and store article url in 'filename_url' 
    get_urls() if UPDATE_NEWS else None 
    
    # Print Process
    print(f'---> Get article links donee')
    print('Start getting article content: ')
    
    # All link, data structure: {link: time}
    links_all = load_json(filename_url)
    
    # New link: get the links that have not scraped and store in 'filename_ch' yet
    new_link_to_scrape = links_all.keys() - news_ch.keys()
    links_new = {k: v for k, v in links_all.items() if k in new_link_to_scrape}
    for link, time in tqdm(links_new.items()):
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
            store_json(filename_ch, news_ch)
            store_json(filename_en, news_en)
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
        store_json(filename_ch, news_ch)
        try:
            # To find translate error just find what news in ch but not in en
            news_en[link] = [time, translate(title), translate(content_article)]
            store_json(filename_en, news_en)
        except:
            print('Translate error')
            
# ------------
# MAIN
def main(UPDATE_NEWS:bool, TICKERS:str, KEYWORD:str, INTERVAL:str, MEDIA: str, start:str):
    """
    Get current date data.
    1. Trigger every morning at 8 am to check if new news 
    2. If have new news, add the news to past datas
    """
    # Get past data
    if MEDIA == 'UDN':
        get_news_udn(KEYWORD=KEYWORD, UPDATE_NEWS=UPDATE_NEWS, start=start)
    elif MEDIA == 'other meida name':
        pass
    # Get past price
    get_price(INTERVAL=INTERVAL, TICKERS=TICKERS)
        
if __name__ == '__main__':
    '''
    # Here still confusion about how to do. Just like this temporary.
    - UPDATE_LATEST_NEWS: get latest news
    - UPDATE_NEWS: get new links (always set to True)
    - TICKERS: Tickers to get price
    - KEYWORD: Keyword to search news
    - INTERVAL: ['1m', '1d']
    - MEDIA: ['UDN', 'ANUE']
    - start: Start time to get news 
    '''
    TICKERS = '2357.TW'
    KEYWORD = '華碩'
    UPDATE_LATEST_NEWS = False
    UPDATE_NEWS = True
    INTERVAL = '1d' 
    MEDIA = 'UDN'    
    start = datetime.datetime.now().date().strftime('%Y-%m-%d %H:%M:%S') if UPDATE_LATEST_NEWS else '2019-02-28 00:00:00'      
    print(f'NOW: {UPDATE_NEWS}, Tickers: {TICKERS}, Keyword: {KEYWORD}, INTERVAL:{INTERVAL}, MEDIA: {MEDIA}')        
    main(UPDATE_NEWS, TICKERS, KEYWORD, INTERVAL, MEDIA, start)

"""
NOTE:
Here scrape the yahoo finance price. But do not need these codes since it seems can get it from API.
----------
def get_yahoo_finance_url(symbol):
    # Not working at the web: "https://finance.yahoo.com/quote/"
    base_url = 'https://tw.stock.yahoo.com/quote/' 
    url = base_url + symbol
    return url

def scrape_get_current_price(TICKERS = '00940.TW'):
    yahoo_finance_url = get_yahoo_finance_url(TICKERS)
    response = requests.get(yahoo_finance_url)    
    
    if response.status_code == 200:
        '''
        price html: <span class="Fz(32px) Fw(b) Lh(1) Mend(16px) D(f) Ai(c) C($c-trend-up)">9.52</span>
        '''
        filename = 'price.json'
        data = load_json(filename) if os.path.exists(filename) else {}
        soup = BeautifulSoup(response.text, 'html.parser')
        price = soup.find("span", {"class": "Fz(32px)"}).text
        data[str(datetime.datetime.now())] = [TICKERS, price]
        print(f"Current price: {price}")
        store_json('price.json', data)
    
    else:
        print("Failed to retrieve data.")
        return None

"""