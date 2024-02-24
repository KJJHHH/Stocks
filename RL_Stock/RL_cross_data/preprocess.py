import yfinance as yf
import pandas as pd

def price_change_dropna(group_price_data, selected_coid):        
    stocks = []
    for i in selected_coid:
        stock_price_data = group_price_data[i]
        stock_price_data['do'] = stock_price_data.Open.pct_change()
        stock_price_data['dh'] = stock_price_data.High.pct_change()
        stock_price_data['dl'] = stock_price_data.Low.pct_change()
        stock_price_data['dc'] = stock_price_data.Close.pct_change()
        stock_price_data['dv'] = stock_price_data.Volume_1000_Shares.pct_change()
        stock_price_data = stock_price_data.fillna('ffill').dropna()
        stocks.append(stock_price_data)
                
    stock_price_data = stocks
    return stock_price_data

def select_industry(data_cross_section, industry):
    return data_cross_section.set_index('Industry_Eng').loc['M1600 Electrical and Cable']

def group_by_coid(data_cross_section):
    """
    """
    grouped = data_cross_section.groupby('coid')
    all_groups = {}
    for name, group in grouped:
        all_groups[name] = group        
    return all_groups

def get_adj_close(data):
    # adj Close
    data["Adj Close"] = data["Close"] * data["Adjust_Factor"]
    return data

def set_index(data):
    """
    """
    datas = []
    for d in data:
        datas.append(d.reset_index().drop('Industry_Eng',axis=1).set_index(['coid', 'mdate']))
    return datas

def get_reward_for_holding_stock(data_list):
    """
    data_list: list of data frame
    return: list of reward for each day if hold stock in each day
    """
    rewards_comp_list = []
    for data in data_list:
        # return of each day if hold
        data['rewards'] = (data['Close'].shift(-1) - data['Open'].shift(-1))/data['Open'].shift(-1)
        rewards_comp_list.append(data)
    return rewards_comp_list


