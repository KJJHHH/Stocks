fish
# --- 
cd CSMLQuantEcon/Quant/Stocks/Sentiment/data/TW
conda activate quant 
#                    NOW    TICKER      KEYWORD   INTERVAL    MEDIA
python news_price.py  0    '0050.TW'    '大盤'    '1d'        'UDN'
python news_price.py  0    '0050.TW'    'ETF'     '1d'        'UDN'