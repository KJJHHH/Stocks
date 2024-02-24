import torch
import yfinance as yf
import pandas as pd
import numpy as np
from tqdm import tqdm
from talib import abstract
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def fetch_price(C, start_date, end_date = None):
    if end_date is None:
        data = C.history(start=start_date)
    else:
        data = C.history(start=start_date, end=end_date)
    return data

def pct_change(stock_price_data):
    stock_price_data['do'] = stock_price_data['Open'].pct_change()
    stock_price_data['dh'] = stock_price_data['High'].pct_change()
    stock_price_data['dl'] = stock_price_data['Low'].pct_change()
    stock_price_data['dc'] = stock_price_data['Close'].pct_change()
    stock_price_data['dv'] = stock_price_data['Volume'].pct_change()
    
    stock_price_data['o-c-nextday'] = \
        (stock_price_data['Close'].shift(-1) - stock_price_data['Open'].shift(-1))\
        /(stock_price_data['Open'].shift(-1))
    
    stock_price_data.dropna(inplace=True)
    return stock_price_data

def clean(prc):
    # Replace infinite values with NaN
    prc.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Drop rows with NaN values
    prc = prc.dropna()
    return prc

def train_test(prc, window_size=100):
    df_train = prc.loc[:'2023']
    test_thr = prc.loc['2023'].index[-window_size + 1]
    df_test = prc.loc[test_thr:]
    return df_train, df_test

def train_valid(X, y):
    # valid
    train_size = int(0.9 * len(X))
    x_train, x_valid = X[:train_size], X[train_size:]
    y_train, y_valid = y[:train_size], y[train_size:]
    return x_train, x_valid, y_train, y_valid

def scaling(data):
    scaler = StandardScaler()
    data[['do', 'dh', 'dl', 'dc', 'dv']] \
       = np.exp(scaler.fit_transform(data[['do', 'dh', 'dl', 'dc', 'dv']]))
    return data

# preprocess func
def window_x_y(df, window_size=100):
    '''
    df: splitted data: train and valid / test
    '''
    x1_list, y1_list, date = [], [], []

    # Iterate over the DataFrame to create the training and testing sets
    for i in tqdm(range(len(df)-window_size+1)):
        window = df.iloc[i:i+window_size]  # Extract the window of data
        # print(window.T.values)
        x1_values = window[['do', 'dh', 'dl', 'dc', 'dv']].T.values  
        y1_values = window[['o-c-nextday']].iloc[-1].T.values
        x1_list.append(x1_values)
        y1_list.append(y1_values)
        date.append(window.index[-1])

    # Convert the lists to NumPy arrays
    x = np.array(x1_list)
    y = np.array(y1_list)
    return x, y, date

# gaf
def process_x(x):
    X = []
    
    for i in range(len(x)):
        X_element = []
        
        for j in range(len(x[i])):
            X_element.append(gaf(x[i][j]))
            # print(gaf(x[i][j]))

        X.append(X_element)
    X = np.array(X)
    return X

def loader(x, y, batch_size = 300):
    x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

    dataset = TensorDataset(x_tensor, y_tensor)
    dataloader = DataLoader(dataset , batch_size, shuffle=False)

    return dataloader

def gaf(X):
    X_normalized = X.reshape(-1, 1).flatten()
    # Pairwise differences
    X_diff = np.expand_dims(X_normalized, axis=0) - np.expand_dims(X_normalized, axis=1)
    # Gramian Angular Field
    GAF = np.cos(X_diff)
    return GAF

def K(df):
    """
    狀態 K：採用 K 線的各組成部分的長度，意即最高價-最低價、開盤價-收盤價、最高價-開盤價、收盤價-最低價
    """
    df['o-c'] = (df['Open'] - df['Close'])/df['Close']
    df['h-l'] = (df['High'] - df['Low'])/df['Low']
    df['h-o'] = (df['High'] - df['Open'])/df['Open']
    df['c-l'] = (df['Close'] - df['Low'])/df['Low']
    return df

def A(df, rolling):
    """
    狀態 A：採用價格平均線與交易量平均線
    """
    df[f'sma_{rolling}_close'] = df['Close'].rolling(rolling).mean()/df["Close"]
    # df[f'sma_{rolling}_volume'] = df['Volume'].rolling(rolling).mean()/df["Volume"]
    return df

def categorized(data, data_test = None, n_bins=10):
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    transformed_data = pd.DataFrame(columns=data.columns)
    if data_test is not None:
        transformed_data_test = pd.DataFrame(columns=data_test.columns)
    for column in data.columns:
        est.fit(data[[column]])
        transformed_column = est.transform(data[[column]])        
        transformed_data[column] = transformed_column.flatten()
        if data_test is not None:
            transformed_column_test = est.transform(data_test[[column]])
            transformed_data_test[column] = transformed_column_test.flatten()
    if data_test is not None:
        return transformed_data, transformed_data_test
    return transformed_data


def reaname_col(df):
    """
    Use talib columns
    """
    return df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})

def talib_func(df):
    """
    Use talib function
    """
    df['rsi'] = abstract.RSI(df)
    MACD = abstract.MACD(df)
    df['macd'], df['macdsignal'], df['macdhist'] = MACD['macd'], MACD['macdsignal'], MACD['macdhist']
    df['adx'] = abstract.ADX(df)
    df['cci'] = abstract.CCI(df)
    df['atr'] = abstract.ATR(df)
    df['obv'] = abstract.OBV(df)
    df['mom'] = abstract.MOM(df)    
    return df

