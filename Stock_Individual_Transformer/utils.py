from tqdm import tqdm
import numpy as np
import yfinance as yf
import torch
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def gaf(X):
    X = X.reshape(-1)
    # Pairwise differences
    X_diff = X.unsqueeze(0) - X.unsqueeze(1) 
    # Gramian Angular Field
    # GAF = torch.cos(X_diff)
    GAF = X_diff

    return GAF

def mask(GAF):    
    mask =  torch.triu(torch.ones(GAF.shape[0], GAF.shape[1])).to(device)
    GAF = GAF * mask
    return GAF

def normalize(x, mean, std):
    return (x - mean) / std

def fetch_stock_price(stock_symbol, start_date, end_date):
    # 使用 yf.Ticker() 建立 Ticker 對象
    stock = yf.Ticker(stock_symbol)

    # 使用 history() 方法取得歷史價格資訊
    stock_data = stock.history(start=start_date, end=end_date)

    return stock_data


# preprocess func
def window_x_y(df, num_class, window_size=100):
    '''
    df: splitted data: train and valid / test
    '''
    x1_list, y1_list, date = [], [], []

    # Iterate over the DataFrame to create the training and testing sets
    for i in tqdm(range(len(df)-window_size+1)):
        window = df.iloc[i:i+window_size]  # Extract the window of data
        x1_values = window[['do', 'dh', 'dl', 'dc', 'dv', 'Close']].T.values  # Adjust column names as needed
        if num_class == 1:
            y1_values = window[['doc_1']].iloc[-1].T.values
        if num_class == 2:
            y1_values = window[['do_1', 'dc_1']].iloc[-1].T.values
        x1_list.append(x1_values)
        y1_list.append(y1_values)
        date.append(window.index[-1])

    # Convert the lists to NumPy arrays
    x = np.array(x1_list)
    y = np.array(y1_list)
    return x, y, date

def get_src(df, num_class):    
    x, y, date = window_x_y(df, num_class, 1)
    src = x[:2000]
    return torch.tensor(src).to(dtype=torch.float32)   

def process_x(x):
    # gaf
    import gc
    X = []
    x = torch.tensor(x, dtype=torch.float32).to(device)
    for i in tqdm(range(len(x))):
        X_element = []
        for j in range(len(x[i])):
            X_element.append(mask(gaf(x[i][j])).unsqueeze(0))
        X_element = torch.cat(X_element, dim=0).unsqueeze(0)
        X.append(X_element)
    X = torch.cat(X, dim=0)
    return X

def train_test(X, y):

    import random
    # valid
    percentage = 95
    num_numbers = int((percentage / 100) * len(X))
    num_numbers = len(X) - 160

    # Generate a list of randomly selected numbers
    """valid_numbers = random.sample(range(0, len(X)), num_numbers)
    training_numbers = [num for num in range(0, len(X)) if num not in valid_numbers]"""

    # Train, valid
    x_train = X[:num_numbers]
    x_test = X[num_numbers:]

    y_train = y[:num_numbers]
    y_test = y[num_numbers:]
    return x_train, x_test, y_train, y_test

def train_valid(X, y):

    import random
    # valid
    percentage = 95
    num_numbers = int((percentage / 100) * len(X))
    num_numbers = len(X) - 160

    # Generate a list of randomly selected numbers
    """valid_numbers = random.sample(range(0, len(X)), num_numbers)
    training_numbers = [num for num in range(0, len(X)) if num not in valid_numbers]"""

    # Train, valid
    x_train = X[:num_numbers]
    x_valid = X[num_numbers:]

    y_train = y[:num_numbers]
    y_valid = y[num_numbers:]
    return x_train, x_valid, y_train, y_valid

def loader(x, y, batch_size = 16):
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size, shuffle=False, drop_last=True)
    return dataloader

"""
While mask in train, not process
def mask_old(x):
    # Specify the size of the square matrix
    matrix_size = 100
    channels = 5
    batch = x.size(0)

    # Create an empty tensor filled with zeros
    mask = torch.zeros((matrix_size, matrix_size), dtype=torch.float)

    # Fill the upper triangle with ones
    for i in range(matrix_size):
        for j in range(i, matrix_size):
            mask[i, j] = 1
    masks = mask.repeat(batch, 5, 1, 1)
    masked_x = x * masks
    return masked_x
"""