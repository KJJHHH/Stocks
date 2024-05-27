import datetime as dt
import numpy as np
import pandas as pd
import pickle, json
import logging, warnings
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from colorama import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import transformers, datasets
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
import bitsandbytes as bnb
from tqdm import tqdm
from datasets import load_dataset, load_from_disk, Dataset
from peft import (
    PeftModel,
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training, 
    TaskType
        )
from data.utils import *
warnings.filterwarnings('ignore')
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

class finBert_Sentiment(nn.Module):
    def __init__(
        self,
        MEDIA,
        KEYWORD,
        STOCK,
        DAY_RANGE,
        content_type,
        filename_news,
        filename_price,
        filename_log,) -> None:
        super(finBert_Sentiment, self).__init__()
        
        # Some settings        
        self.MEDIA = MEDIA
        self.KEYWORD = KEYWORD
        self.STOCK = STOCK
        self.DAY_RANGE = DAY_RANGE
        self.content_type = content_type
        
        # files
        self.filename_news = filename_news
        self.filename_price = filename_price
        self.output_dir = 'finbert-checkpoints/'
        
        # Store result and error log
        self.filename_log = filename_log
        if os.path.exists(self.filename_log):
            os.remove(self.filename_log)
        logging.basicConfig(filename=self.filename_log, level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.getLogger('transformers').setLevel(logging.ERROR)
        
        # Train settings        
        MICRO_BATCH_SIZE = 4  # 定義微批次的大小
        BATCH_SIZE = 16  # 定義一個批次的大小
        self.GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE  # 計算每個微批次累積的梯度步數
        self.WARMUP_STEP = 50
        self.num_epochs = 100
        self.lr = 1e-6
        self.save_steps = 10000  # No save
        
        # Models ------>
        # 1. 'yiyanghkust/finbert-tone'
        # 2. 'ProsusAI/finbert' 
        # 3. "cardiffnlp/twitter-roberta-base-sentiment-latest": have randomness
        # 4. "bert-base-uncased"
        model_name = 'yiyanghkust/finbert-tone'
        cache_dir = './cache'
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            device_map='cuda',
            num_labels=3,
            cache_dir=cache_dir,
            low_cpu_mem_usage = True,
        )
        # Tokenizer
        try:
            self.tokenizer_type = 'Bert'
            self.tokenizer = BertTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
            )
        except: 
            # Autotokenizer do not have token_type_ids columns
            self.tokenizer_type = 'Auto'
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
            )
            
        # Data ------->
        self.df_train = None
        self.df_val = None
        self.df_test = None
        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None
        self.df_target_stock = None
        self.df_target_price = None
        
    def set_seeds(self):
        seed = 42
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
    def load_data(self, text_type = 'title'):
        """
        English data for finBert
        - filename_news:   json file   | data_news = [{link: [date, title, text]}, ...] 
        - filename_price:  pickle file | data_price = [{date: price}, ...]
        """
        def labelling(price):
            if price > .004:
                return 1
            elif price < -.004:
                return 2
            else:
                return 0
        
        df_all_list = []
        for file_index, (filename_news, filename_price) in enumerate(zip(self.filename_news, self.filename_price)):
            df = pd.DataFrame(columns=['title', 'content', 'label', 'price'])
            assert os.path.exists(filename_news) and os.path.exists(filename_price), \
                f"Error: data file {filename_news} or {filename_price} does not exist"
            with open(filename_news, 'r') as f:
                news = json.load(f)
            with open(filename_price, 'rb') as f:
                price = pickle.load(f)
            price = price.pct_change().shift(-1).dropna()
        
            """
            - If the date has news, then add to the dataframe
            - If the date has news and is not workday, add the news to next workday
            """
            for link, infos in tqdm(news.items()):
                time, title, content = infos
                if title == None or content == None:
                    continue
                date = dt.datetime.strptime(time, '%Y-%m-%d %H:%M:%S').date()
                if date in price.index.date:
                    if date not in df.index:
                        price_ = price.loc[str(date)]
                        label_ = labelling(price_)
                        df.loc[date] = [title, summarise(content), label_, price_]
                    if date in df.index:
                        df.loc[date, 'title'] = df.loc[date, 'title'] + '\n' + title
                        df.loc[date, 'content'] = df.loc[date, 'content'] + '\n' + content

            df_all_list.append(df)
            
            if file_index == 0:
                self.df_target_price = price
                self.df_target_stock = df
            
        # Split data
        """
        NOTE:
        - Evaluate with target stock
        - Train with multiple stock data
        """
        sep_val = pd.Timestamp('2024-01-01').date()
        sep_test = pd.Timestamp('2024-02-15').date()
        
        df_all = pd.concat(df_all_list)
        df_all = df_all.sort_index()
        df_train = df_all.loc[df_all.index < sep_val]
        df_valid = self.df_target_stock.loc[
            (self.df_target_stock.index >= sep_val) &
            (self.df_target_stock.index < sep_test)
            ]
        df_test = self.df_target_stock.loc[self.df_target_stock.index >= sep_test]
        
        # Something
        self.df_test = df_test        
        self.df_target_price = pd.DataFrame(self.df_target_price)
        self.df_target_stock.index = pd.to_datetime(self.df_target_stock.index)
        
        # Dataset
        max_length = 128 if text_type == 'title' else 512
        columns = ['input_ids', 'token_type_ids', 'attention_mask', 'label'] if self.tokenizer_type == 'Bert' else ['input_ids', 'attention_mask', 'label']
        
        self.dataset_target_stock = Dataset.from_pandas(self.df_target_stock[[text_type, 'label', 'price']])
        self.dataset_train = Dataset.from_pandas(df_train[[text_type, 'label']])
        self.dataset_val = Dataset.from_pandas(df_valid[[text_type, 'label']])
        self.dataset_test = Dataset.from_pandas(df_test[[text_type, 'label']])

        self.dataset_target_stock = self.dataset_target_stock.map(lambda e: self.tokenizer(e[text_type], truncation=True, padding='max_length', max_length=max_length), batched=True)
        self.dataset_train = self.dataset_train.map(lambda e: self.tokenizer(e[text_type], truncation=True, padding='max_length', max_length=max_length), batched=True)
        self.dataset_val = self.dataset_val.map(lambda e: self.tokenizer(e[text_type], truncation=True, padding='max_length', max_length=max_length), batched=True)
        self.dataset_test = self.dataset_test.map(lambda e: self.tokenizer(e[text_type], truncation=True, padding='max_length' , max_length=max_length), batched=True)

        self.dataset_target_stock.set_format(type='torch', columns=columns)
        self.dataset_train.set_format(type='torch', columns=columns)
        self.dataset_val.set_format(type='torch', columns=columns)
        self.dataset_test.set_format(type='torch', columns=columns)            
            
        return self.dataset_target_stock, self.dataset_train, self.dataset_val, self.dataset_test 
    
    def backtesting(self, df_backtest):                                    
        df_backtest['pred_label'] = df_backtest['pred_label'].replace(2, -1)
        df_backtest['senti-return'] = df_backtest['pred_label'] * df_backtest['Close']
        df_backtest['senti-asset'] = (df_backtest['senti-return'] + 1).cumprod()
        df_backtest['buyhold-asset'] = (df_backtest['Close'] + 1).cumprod()
        
        senti_mean = np.mean(df_backtest['senti-return'])
        buyhold_mean = np.mean(df_backtest['Close'])
        senti_std = np.std(df_backtest['senti-return'])
        buyhold_std = np.std(df_backtest['Close'])
        print(f'Average daily return | sentiment: {senti_mean} | buyhold: {buyhold_mean}')
        print(f'Std daily return     | sentiment: {senti_std}  | buyhold: {buyhold_std}')        
        
        plt.figure(figsize=(12,6))
        plt.plot(df_backtest['senti-asset'], label = 'sentiment')
        plt.plot(df_backtest['buyhold-asset'], label = 'buyhold')
        plt.legend()
        plt.show()  
    
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {'accuracy' : accuracy_score(predictions, labels)}
        
    def finetuneFull_score_trade(self, train = False):         
        args = TrainingArguments(
                output_dir = self.output_dir,
                evaluation_strategy = 'steps',
                eval_steps = 10,
                save_strategy = 'steps',
                save_steps = self.save_steps,
                learning_rate=self.lr,
                per_device_train_batch_size=32,
                per_device_eval_batch_size=32,
                num_train_epochs=self.num_epochs,
                weight_decay=0.01,
                load_best_model_at_end=True,
                metric_for_best_model='loss',
                gradient_accumulation_steps=self.GRADIENT_ACCUMULATION_STEPS,
                warmup_steps=self.WARMUP_STEP,
        )

        trainer = Trainer(
                model=self.model,                         # the instantiated 🤗 Transformers model to be trained
                args=args,                  # training arguments, defined above
                train_dataset=self.dataset_train,         # training dataset
                eval_dataset=self.dataset_val,            # evaluation dataset
                compute_metrics=self.compute_metrics
        )

        result = trainer.train() 
        print(result)
        
        self.model.eval()        
        self.df_test['pred_label'] = np.argmax(trainer.predict(self.dataset_test)[0], axis = 1)             
        df_backtest = self.df_target_price.merge(                                        
                pd.DataFrame(self.df_test['pred_label']), 
                left_index = True, 
                right_index = True, 
                how = 'left').\
                fillna(0)
        df_backtest = df_backtest.loc[df_backtest.index >= pd.Timestamp(min(self.df_test.index))]   
        self.backtesting(df_backtest)       
        
    def finetuneLora_score_trade(self, train = False):

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, lora_dropout=0.3
        )
        self.model = get_peft_model(self.model, lora_config)          
        
        args = TrainingArguments(
                output_dir = self.output_dir,
                evaluation_strategy = 'steps',
                eval_steps = 10,
                save_strategy = 'steps',
                save_steps = self.save_steps,
                learning_rate=self.lr,
                per_device_train_batch_size=32,
                per_device_eval_batch_size=32,
                num_train_epochs=self.num_epochs,
                weight_decay=0.01,
                load_best_model_at_end=True,
                metric_for_best_model='loss',
                gradient_accumulation_steps=self.GRADIENT_ACCUMULATION_STEPS,
                warmup_steps=self.WARMUP_STEP,
        )

        trainer = Trainer(
                model=self.model,                         # the instantiated 🤗 Transformers model to be trained
                args=args,                  # training arguments, defined above
                train_dataset=self.dataset_train,         # training dataset
                eval_dataset=self.dataset_val,            # evaluation dataset
                compute_metrics=self.compute_metrics
        )
        
        if train:
            trainer.train()
        
        self.model.eval()
        result = np.argmax(trainer.predict(self.dataset_test)[0], axis = 1)

        asset = 1
        asset_hist = []
        asset_buyhold = 1
        asset_buyhold_hist = []
        for predict_label, returns in zip(result, self.df_test['price']):
            asset_hist.append(asset)
            if predict_label == 1:
                asset *= (1+returns)
            asset_buyhold *= (1+returns)
            asset_buyhold_hist.append(asset_buyhold)
        
        self.df_test['asset-senti'] = asset_hist
        self.df_test['asset-buyhold'] = asset_buyhold_hist
        
        plt.figure(figsize=(12,8))
        plt.plot(self.df_test['asset-senti'], label = 'sentiment')
        plt.plot(self.df_test['asset-buyhold'], label = 'buyhold')
        plt.legend()
        plt.show()
        
        return result       
            
    def evaluate_with_pretrained(self):
        """
        Evaluate without train
        """
        args = TrainingArguments(
                output_dir = self.output_dir,
                evaluation_strategy = 'steps',
                eval_steps = 10,
                save_strategy = 'steps',
                save_steps = self.save_steps,
                learning_rate=self.lr,
                per_device_train_batch_size=32,
                per_device_eval_batch_size=32,
                num_train_epochs=self.num_epochs,
                weight_decay=0.01,
                load_best_model_at_end=True,
                metric_for_best_model='accuracy',
                gradient_accumulation_steps=self.GRADIENT_ACCUMULATION_STEPS,
                warmup_steps=self.WARMUP_STEP,
        )
        trainer = Trainer(
                model=self.model,                         # the instantiated 🤗 Transformers model to be trained
                args=args,                  # training arguments, defined above
                train_dataset=self.dataset_train,         # training dataset
                eval_dataset=self.dataset_val,            # evaluation dataset
                compute_metrics=self.compute_metrics
        )
        self.model.eval()    
            
        self.df_target_stock['pred_label'] = np.argmax(trainer.predict(self.dataset_target_stock)[0], axis = 1)             
        df_backtest = self.df_target_price.merge(                                        
                pd.DataFrame(self.df_target_stock['pred_label']), 
                left_index = True, 
                right_index = True, 
                how = 'left').\
                fillna(0)
        df_backtest = df_backtest[-len(self.df_test):]
        self.backtesting(df_backtest)
        
    
    """
    for a in senti.dataset_all:
    del a['label']
    a['input_ids'] = a['input_ids'].reshape(1, -1)
    a['attention_mask'] = a['attention_mask'].reshape(1, -1)
    model(**a)
    """
    
    
    """
    
        asset = 1
        asset_hist = []
        return_hist = []
        asset_buyhold = 1
        asset_buyhold_hist = []
        return_buyhold_hist = []
        for index, (returns, predict_label, _, _, _) in df_backtest.iterrows():    
            asset_hist.append(asset)
            asset_buyhold_hist.append(asset_buyhold)            
            if predict_label == 1:
                asset *= (1+returns)
                return_hist.append(returns)
            elif predict_label == 2:
                asset *= (1-returns)
                return_hist.append(-returns)
            else:
                return_hist.append(0)
            asset_buyhold *= (1+returns)            
            return_buyhold_hist.append(returns)
        
        df_backtest['asset-senti'] = asset_hist
        df_backtest['asset-buyhold'] = asset_buyhold_hist
        df_backtest = df_backtest[-180:]
        # print(self.df_all)
        # print(self.df_price)
        print(len(df_backtest))"""