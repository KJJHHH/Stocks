from bs4 import BeautifulSoup
import torch
import requests
import datetime
from datetime import time
from time import sleep
import json
import os
import sys
sys.path.append('../')
from utils import *

# ------------
# STORE and LOAD
encoding = 'cp950'
def store_data(file_path, data):
    with open(file_path, "w", encoding=encoding) as json_file:
        json.dump(data, json_file, indent=4) 
        
def load_data(file_path):
    with open(file_path, "r", encoding=encoding) as json_file:
        data = json.load(json_file)
    return data

# -------------
# PRICE
# NEWS
def get_price(NOW, TICKERS = '00940.TW', INTERVAL = '1m', start=None):
    import yfinance as yf
    import pickle
    
    now = datetime.datetime.now()
    if NOW:
        # Get the now price
        # 1. Load the stored data (past data) if data exist
        # 2. Get the latest data
        # 3. Add the latest data to the stored data
        pass
        """
        now = datetime.datetime.now()
        price = yf.download(TICKERS, start=now, end=now, interval='1m')
        price['Close']
        """
    else:
        # Get the past price
        start = datetime.datetime.strptime(start, "%Y-%m-%d %H:%M:%S").date()
        price = yf.download(TICKERS,start=start, end=now, interval=INTERVAL)['Close']
        with open(f'PRICE_{TICKERS}_{INTERVAL}.pickle', 'wb') as f:
            pickle.dump(price, f)

def get_news_anue(NOW, KEYWORD ="大盤", start = None):
    media = 'ANUE'
    start = datetime.datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
    page = 1
    filename = f'NEWS_{media}_{KEYWORD}.json'
    data = load_data(filename) if os.path.exists(filename) else {}
    
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
                    paragraph_ = translate(paragraph_)
                    paragraph_ = summarise(paragraph_)
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
            store_data(filename, data)
            
        # update page
        page += 1

def get_news_udn(NOW, KEYWORD ="大盤", start = None):
    from tqdm import tqdm
    media = 'UDN'
    start = datetime.datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
    filename = f'NEWS_{media}_{KEYWORD}.json'
    filename_url = f'NEWS_{media}_{KEYWORD}_URL.json'
    data = load_data(filename) if os.path.exists(filename) else {}    
    
    def get_urls():
        """
        Format for store article urls:
        urls_all = {'time': url}
        """
        # Get udn urls        
        page = 1
        urls_all = load_data(filename_url) if os.path.exists(filename_url) else {}
        print(f'Scrape news after {start}')
        print('-----------------')
        print('Start getting article links: ', f'https://money.udn.com/search/result/1001/{KEYWORD}/{page}')
        
        while True:
            url = f'https://money.udn.com/search/result/1001/{KEYWORD}/{page}'
            htmls = requests.get(url).content
            soup = BeautifulSoup(htmls, 'html.parser')
            
            urls_class = soup.find_all('div', {'class': 'story__content'})
            for div in urls_class:
                # article_url
                time = datetime.datetime.strptime(div.find_all('time')[0].text, '%Y-%m-%d %H:%M')
                # Check if already get the url
                if time in urls_all:
                    continue
                if time < start:
                    return 
                urls_all[str(time)] = div.find_all('a', href=True)[0]['href']
                store_data(filename_url, urls_all)
            page += 1
            print(f'\r---> article links at {time}', end='', flush=True)
        
    
    # Store article url and store in 'filename_url' 
    get_urls() 
    print(f'\r---> Get article links doneeeeeeeeeeeeeeee')
    urls_all = load_data(filename_url)
    data = load_data(filename) if os.path.exists(filename) else {}
    print('-----------------')
    print('Start getting article content: ')
    for time, link in tqdm(urls_all.items()):
        """
        Format for store article:
        data = {'link': [time, title, content]}
        """
        # Get the link and check if already get the article
        if link in data:
            continue
        
        # Get the htmls
        htmls = requests.get(link).content
        soup = BeautifulSoup(htmls, 'html.parser')
        
        # Get title, content
        try:
            'Something went wrong here for little news, do not know why'
            title = soup.find_all('h1', {"article-head__title"})[0].text
            content = soup.find_all('section', {"article-body__editor"})
        except:
            continue
        
        # Concat and translate
        content_article = ''
        for con in content:
            con_ = con.text
            if '延伸閱讀' in con_:
                continue
            try:
                title = translate(title)
                con_ = translate(con_)
                content_article += con_
            except:
                print('translate error')
        if content_article == '':
            continue
        
        # Store the data
        data[link] = [time, title, content_article]
        store_data(filename, data)
            
     

# ------------
# MAIN
def main(NOW:bool, TICKERS:str, KEYWORD:str, INTERVAL:str, MEDIA: str, start:str):
    if NOW:
        """
        Get current min data.
        Trigger to get news and price every minutes.
        """
        start_time = time(9, 0)
        end_time = time(13, 30)
        while True:
            print('=====> Get news')
            get_news_anue(KEYWORD)
            
            print('=====> Get price')
            get_price(TICKERS)
            
            print('=====> Sleep 60s')
            now = datetime.datetime.now().time()
            if now < start_time or now > end_time:
                return 
            else:
                sleep(60)
    else:
        # Get past data
        if MEDIA == 'UDN':
            get_news_udn(NOW=NOW, KEYWORD=KEYWORD, start=start)
        elif MEDIA == 'ANUE':
            get_news_anue(NOW=NOW, KEYWORD=KEYWORD, start=start)
        # Get past price
        get_price(NOW=NOW, INTERVAL=INTERVAL, TICKERS=TICKERS, start=start)
        
if __name__ == '__main__':
    '''
    NOW: now -> True / 0 -> False
    TICKERS: Tickers to get price
    KEYWORD: Keyword to search news
    INTERVAL: ['1m', '1d']
    MEDIA: ['UDN', 'ANUE']
    ---
    start: Start time to get news
    '''
    NOW = True if sys.argv[1] == 'now' else False
    TICKERS = sys.argv[2] if len(sys.argv) > 2 else '0050.TW'
    KEYWORD = sys.argv[3] if len(sys.argv) > 3 else '大盤'
    INTERVAL = sys.argv[4] if len(sys.argv) > 4 else '1d'
    MEDIA = sys.argv[5] if len(sys.argv) > 5 else 'UDN'
    
    start = '2023-01-01 00:00:00' if not NOW else None
    print(f'NOW: {NOW}, Tickers: {TICKERS}, Keyword: {KEYWORD}, INTERVAL:{INTERVAL}, MEDIA: {MEDIA}')        
        
    main(NOW, TICKERS, KEYWORD, INTERVAL, MEDIA, start)




"""
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
        data = load_data(filename) if os.path.exists(filename) else {}
        soup = BeautifulSoup(response.text, 'html.parser')
        price = soup.find("span", {"class": "Fz(32px)"}).text
        data[str(datetime.datetime.now())] = [TICKERS, price]
        print(f"Current price: {price}")
        store_data('price.json', data)
    
    else:
        print("Failed to retrieve data.")
        return None

"""