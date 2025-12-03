import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
from utils.augmentation import run_augmentation_single

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        '''
        __init__() is the constructor used to initialize various parameters and attributes of the dataset. 
        It takes a series of arguments, including the path to the data file, the dataset's flag (e.g., train, validate, test), 
        dataset size, feature type, target variable, whether to scale the data, time encoding, time frequency, and more. 
        These parameters are used to configure how the dataset is loaded and processed.
        '''
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features  # 'MS', 'S', 'M'
        self.target = target
        self.scale = scale
        self.timeenc = timeenc  # 时间编码方式
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__() # 把 file 中的数据给读进来，并做了 mean/std 归一化等预处理工作. 存在 self.data_x, self.data_y, self.data_stamp 中.

    def __read_data__(self):
        '''
        The actual process of managing data file into usable data pieces happens in __read_data__()
        '''
        self.scaler = StandardScaler()

        #get raw data from path
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        # split data set into train, vali, test. border1 is the left border and border2 is the right.
        # Once flag(train, vali, test) is determined, __read_data__ will return certain part of the dataset.
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        #decide which columns to select
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]  # column name list (remove 'date')
            df_data = df_raw[cols_data] # remove the first column, which is time stamp info
        elif self.features == 'S':
            df_data = df_raw[[self.target]]  # target column; self.target = 'OT', 这个感觉不需要

        # scale data by the scaler that fits training data
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]  # shape (N_train, num_features)
            #train_data.values: turn pandas DataFrame into 2D numpy
            self.scaler.fit(train_data.values)  # 用 train data 来算 mean 和 std
            data = self.scaler.transform(df_data.values) # 0-1 normalization along dim=0
        else:
            data = df_data.values

        #time stamp:df_stamp is a object of  and
        # has one column called 'date' like 2016-07-01 00:00:00 
        df_stamp = df_raw[['date']][border1:border2] # 时间戳，e.g. 2016/7/1  3:00:00

        # Since the date format is uncertain across different data file, we need to 
        # standardize it so we call func 'pd.to_datetime'
        df_stamp['date'] = pd.to_datetime(df_stamp.date)


        if self.timeenc == 0: # 时间戳编码方式:time feature encoding is fixed or learned
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)

            #now df_frame has multiple columns recording the month, day etc. time stamp
            # next we delete the 'date' column and turn 'DataFrame' to a list
            data_stamp = df_stamp.drop(['date'], 1).values

        elif self.timeenc == 1: # 时间戳编码方式：更复杂的编码方式,#time feature encoding is timeF
            '''
            when entering this branch, we choose arg.embed as timeF meaning we want to 
            encode the temporal info. 'freq' should be the smallest time step, and has 
            options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
            So you should check the timestep of your data and set 'freq' arg. 
            After the time_features encoding, each date info format will be encoded into 
            a list, with each element denoting the relative position of this time point
            (e.g. Day of Week, Day of Month, Hour of Day) and each normalized within scope[-0.5, 0.5]
            '''
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0) 

        # data_x and data_y are same copy of a certain part of data
        self.data_x = data[border1:border2] # (N, num_features), 这里 data_x 和 data_y 是一样的, 都是 ETT_hour 的数据部分（不包括时间戳）.
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0: # if training and need augmentation
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        # #given an index, calculate the positions after this index to truncate the dataset
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # input and target sequence
        seq_x = self.data_x[s_begin:s_end] # shape (seq_len, num_features)
        seq_y = self.data_y[r_begin:r_end] # shape (label_len + pred_len, num_features)

        # time mark
        seq_x_mark = self.data_stamp[s_begin:s_end] # seq_x 每一个时间点对应的 time mark
        seq_y_mark = self.data_stamp[r_begin:r_end] # seq_y 每一个时间点对应的 time mark

        # return 一个 batch：（eq_x, seq_y, seq_x_mark, seq_y_mark）
        return seq_x, seq_y, seq_x_mark, seq_y_mark  # 这里要注意，这里 return 的一个 sample 是什么样子的.

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1 # 从这个 lenth 可以看出，一个训练 sample 就是原始长数据中的一段，训练 samples 之间是重叠的。

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)  # 数据如果开始被 normalize 到的话，这个函数可以把数据还原回去。


class Dataset_ETT_minute(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.seasonal_patterns = seasonal_patterns

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Berkley_sensor(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='moteid_3_temp_volt.csv',
                 target='voltage', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')  # cols 就是 (other features),i.e. 除了 ‘date’ 和 target feature 之外的所有 feature
        df_raw = df_raw[['date'] + cols + [self.target]] # 懂了，这个操作其实就是把 target feature 放到最后一列而已.
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)  # scale 用的是 train data 的 mean 和 std
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # 不需要时间戳
        # df_stamp = df_raw[['date']][border1:border2]
        # df_stamp['date'] = pd.to_datetime(df_stamp.date)
        # if self.timeenc == 0:
        #     df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        #     df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        #     df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        #     df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        #     data_stamp = df_stamp.drop(['date'], 1).values
        # elif self.timeenc == 1:
        #     data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
        #     data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        # 变成 tensor

        self.data_x = torch.tensor(self.data_x, dtype=torch.float32)
        self.data_y = torch.tensor(self.data_y, dtype=torch.float32)
        # self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        # seq_x_mark = self.data_stamp[s_begin:s_end]
        # seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, None, None

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    


class CMAPSSLoader(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='MS', data_path='train_FD001.txt',
                 target='RUL', scale=True, timeenc=0, freq='c', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 30
            self.label_len = 10
            self.pred_len = 5
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features  # 'MS', 'S', 'M'
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        
        # Read CMAPSS data (space-separated)
        column_names = ['unit_number', 'cycle', 'setting1', 'setting2', 'setting3'] + \
                      [f'sensor{i}' for i in range(1, 22)]
        
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path),
                            sep=r'\s+', names=column_names)
        
        # Group by unit_number
        unit_groups = df_raw.groupby('unit_number')
        
        # For training data, calculate RUL as max_cycle - current_cycle
        # For test data, we'll need to load the RUL file separately
        if 'train' in self.data_path:
            # Calculate RUL for training data
            max_cycles = unit_groups['cycle'].max()
            df_raw['RUL'] = df_raw.apply(lambda row: max_cycles[row['unit_number']] - row['cycle'], axis=1)
            
            # Split data into train, val, test based on unit_number
            all_units = df_raw['unit_number'].unique()
            num_units = len(all_units)
            num_train = int(num_units * 0.7)
            num_val = int(num_units * 0.2)
            
            train_units = all_units[:num_train]
            val_units = all_units[num_train:num_train+num_val]
            test_units = all_units[num_train+num_val:]
            
            # Create masks for each set
            if self.set_type == 0:  # train
                mask = df_raw['unit_number'].isin(train_units)
            elif self.set_type == 1:  # val
                mask = df_raw['unit_number'].isin(val_units)
            else:  # test
                mask = df_raw['unit_number'].isin(test_units)
                
            df_data = df_raw[mask]
        else:  # test data
            # For test data, we'll load the actual RUL values later
            # For now, just use all data
            df_data = df_raw
            
        # Select features
        if self.features == 'M' or self.features == 'MS':
            # Use all features except unit_number and cycle
            cols_data = df_data.columns[2:]  # skip unit_number and cycle
        elif self.features == 'S':
            # Only use the target (RUL) - not sure if this makes sense for CMAPSS
            cols_data = [self.target]
        
        df_features = df_data[cols_data]
        
        # Scale data
        if self.scale:
            self.scaler.fit(df_features.values)
            data = self.scaler.transform(df_features.values)
        else:
            data = df_features.values
        
        # Create sequences for each unit
        self.data_x = []
        self.data_y = []
        self.data_stamp = []
        
        unit_groups = df_data.groupby('unit_number')
        
        for unit_num, group in unit_groups:
            unit_data = data[group.index - df_data.index[0]]  # adjust indices to be within the group
            unit_cycles = group['cycle'].values
            
            # Create sequences
            for i in range(len(unit_data) - self.seq_len - self.pred_len + 1):
                s_begin = i
                s_end = s_begin + self.seq_len
                r_begin = s_end - self.label_len
                r_end = r_begin + self.label_len + self.pred_len
                
                # Ensure we don't exceed the unit's data
                if r_end <= len(unit_data):
                    self.data_x.append(unit_data[s_begin:s_end])
                    
                    # For target, we only need the RUL at the end of the prediction window
                    # Assuming RUL is the last column
                    if 'RUL' in cols_data:
                        rul_idx = cols_data.get_loc('RUL')
                        self.data_y.append(unit_data[r_begin:r_end, rul_idx:rul_idx+1])
                    else:
                        # If RUL not in data, append a dummy value (will be replaced with actual RUL later)
                        self.data_y.append(np.zeros((r_end - r_begin, 1)))

                    
                    # Create time stamps (just using cycle numbers)
                    seq_x_mark = unit_cycles[s_begin:s_end].reshape(-1, 1)
                    seq_y_mark = unit_cycles[r_begin:r_end].reshape(-1, 1)
                    
                    self.data_stamp.append((seq_x_mark, seq_y_mark))
        
        # Convert to numpy arrays
        self.data_x = np.array(self.data_x)
        self.data_y = np.array(self.data_y)

    def __getitem__(self, index):
        seq_x = self.data_x[index]
        seq_y = self.data_y[index]
        seq_x_mark, seq_y_mark = self.data_stamp[index]
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Simulate_ar(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ar_data.csv',
                 target='AR_Value', scale=True, timeenc=0, freq='s', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')  # cols 就是 (other features),i.e. 除了 ‘date’ 和 target feature 之外的所有 feature
        df_raw = df_raw[['date'] + cols + [self.target]] # 懂了，这个操作其实就是把 target feature 放到最后一列而已.
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)  # scale 用的是 train data 的 mean 和 std
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    


class UEAloader(Dataset):
    """
    Dataset class for datasets included in:
        Time Series Classification Archive (www.timeseriesclassification.com)
    Argument:
        limit_size: float in (0, 1) for debug
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, args, root_path, file_list=None, limit_size=None, flag=None):
        self.args = args
        self.root_path = root_path
        self.flag = flag
        self.all_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=flag)
        self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

        # pre_process
        normalizer = Normalizer()
        self.feature_df = normalizer.normalize(self.feature_df)
        print(f"num samples ({self.flag}) is {len(self.all_IDs)}")

    def load_all(self, root_path, file_list=None, flag=None):
        """
        Loads datasets from ts files contained in `root_path` into a dataframe, optionally choosing from `pattern`
        Args:
            root_path: directory containing all individual .ts files
            file_list: optionally, provide a list of file paths within `root_path` to consider.
                Otherwise, entire `root_path` contents will be used.
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """
        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_path, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_path, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_path, '*')))
        if flag is not None:
            data_paths = list(filter(lambda x: re.search(flag, x), data_paths))
        input_paths = [p for p in data_paths if os.path.isfile(p) and p.endswith('.ts')]
        if len(input_paths) == 0:
            pattern='*.ts'
            raise Exception("No .ts files found using pattern: '{}'".format(pattern))

        all_df, labels_df = self.load_single(input_paths[0])  # a single file contains dataset

        return all_df, labels_df

    def load_single(self, filepath):
        df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                             replace_missing_vals_with='NaN')
        labels = pd.Series(labels, dtype="category")
        self.class_names = labels.cat.categories
        labels_df = pd.DataFrame(labels.cat.codes,
                                 dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss

        lengths = df.applymap(
            lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series

        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
            df = df.applymap(subsample)

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0])) # 记录一下这个 dataset 最大的 seq len, 可能在某些 architecture 中需要用到这个 attribute
        else:
            self.max_seq_len = lengths[0, 0]  

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)

        df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0] * [row])) for row in range(df.shape[0])), axis=0)

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df

    def instance_norm(self, case):
        '''
        结合具体输入，看看这个是干嘛的
        '''
        if self.root_path.count('EthanolConcentration') > 0:  # special process for numerical stability
            mean = case.mean(0, keepdim=True)
            case = case - mean
            stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
            case /= stdev
            return case
        else:
            return case

    def __getitem__(self, ind):
        batch_x = self.feature_df.loc[self.all_IDs[ind]].values  # (seq_len, features), 这里叫 batch_x 但它其实只是一个 sample, 这里一个 sample 指的就是一个 sequence, 每一个 sequence 的点对应一个 vector
        labels = self.labels_df.loc[self.all_IDs[ind]].values # (1,) 这个 sample 对应的 label
        if self.flag == "TRAIN" and self.args.augmentation_ratio > 0:
            num_samples = len(self.all_IDs)
            num_columns = self.feature_df.shape[1]
            seq_len = int(self.feature_df.shape[0] / num_samples)
            batch_x = batch_x.reshape((1, seq_len, num_columns))
            batch_x, labels, augmentation_tags = run_augmentation_single(batch_x, labels, self.args)

            batch_x = batch_x.reshape((1 * seq_len, num_columns))

        # 这里的 instance_norm 只用在了 ethanolConcentration 的 dataset 上
        return self.instance_norm(torch.from_numpy(batch_x)), \
               torch.from_numpy(labels)

    def __len__(self):
        return len(self.all_IDs)
    


class Dataset_Real_Doppler_Kaggle(Dataset):
    def __init__(self, flag, args=None, root_path=None):
        assert flag in ['TRAIN', 'TEST', 'VAL'], "Flag must be 'TRAIN', 'VAL', or 'TEST'."
        self.flag = flag
        
        self.base_path = os.path.join('dataset', 'real_doppler_RAD_DAR_database')
        self.processed_data_path = os.path.join(self.base_path, 'processed')
        
        if not os.path.exists(self.processed_data_path):
            print("Processed data directory not found. Running full data preparation pipeline...")
            self._process_raw_data()
            self._split_data()

        self.__read_data__()

        # Compatibility attributes for Exp_Classification
        self.max_seq_len = self.input_data.shape[1]
        self.class_names = ['Cars', 'Drones', 'People']
        self.feature_df = np.ones((1, self.input_data.shape[2]))

    def _process_raw_data(self):
        """Processes raw CSV data from `raw` folder into numpy arrays."""
        print("Processing raw data from CSVs...")
        raw_data_path = os.path.join(self.base_path, "raw")
        os.makedirs(self.processed_data_path, exist_ok=True)

        class_labels = {"Cars": 0, "Drones": 1, "People": 2}
        all_input_data = []
        all_targets = []

        for class_name, label in class_labels.items():
            class_folder = os.path.join(raw_data_path, class_name)
            if not os.path.exists(class_folder):
                print(f"Warning: Raw data folder for class '{class_name}' not found at '{class_folder}'. Skipping.")
                continue

            for root, _, files in os.walk(class_folder):
                for file_name in filter(lambda f: f.endswith(".csv"), files):
                    file_path = os.path.join(root, file_name)
                    try:
                        df = pd.read_csv(file_path, header=None)
                        matrix_data = df.to_numpy(dtype=np.float32)
                        if matrix_data.shape == (11, 61):
                            reshaped_data = matrix_data.T
                            all_input_data.append(reshaped_data)
                            all_targets.append(label)
                    except Exception as e:
                        print(f"Error processing {file_name}: {e}")
        
        if not all_input_data:
            raise RuntimeError(f"No raw data was processed. Check the directory: {raw_data_path}")

        input_data_array = np.array(all_input_data)
        targets_array = np.array(all_targets, dtype=np.int64)

        np.save(os.path.join(self.processed_data_path, "input_data.npy"), input_data_array)
        np.save(os.path.join(self.processed_data_path, "targets.npy"), targets_array)

        print(f"Finished processing raw data. Saved {len(all_input_data)} samples.")

    def _split_data(self):
        """Splits the processed numpy arrays into TRAIN, VAL, and TEST sets."""
        print("Splitting data into TRAIN, VAL, and TEST sets...")
        try:
            input_data = np.load(os.path.join(self.processed_data_path, 'input_data.npy'))
            targets = np.load(os.path.join(self.processed_data_path, 'targets.npy'))
        except FileNotFoundError:
            raise RuntimeError("Base 'input_data.npy' or 'targets.npy' not found for splitting.")

        n_samples = input_data.shape[0]
        indices = np.arange(n_samples)
        np.random.seed(42)
        np.random.shuffle(indices)

        shuffled_input = input_data[indices]
        shuffled_targets = targets[indices]

        train_ratio, val_ratio = 0.6, 0.2
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))

        sets = {
            'TRAIN': (shuffled_input[:train_end], shuffled_targets[:train_end]),
            'VAL': (shuffled_input[train_end:val_end], shuffled_targets[train_end:val_end]),
            'TEST': (shuffled_input[val_end:], shuffled_targets[val_end:])
        }

        for flag, (data, labels) in sets.items():
            np.save(os.path.join(self.processed_data_path, f'input_data_{flag}.npy'), data)
            np.save(os.path.join(self.processed_data_path, f'targets_{flag}.npy'), labels)
        
        print("Data splitting complete.")

    def __read_data__(self):
        """Loads the specific data split based on the self.flag."""
        input_file = os.path.join(self.processed_data_path, f'input_data_{self.flag}.npy')
        targets_file = os.path.join(self.processed_data_path, f'targets_{self.flag}.npy')
        
        self.input_data = torch.from_numpy(np.load(input_file)).float()
        self.targets = torch.from_numpy(np.load(targets_file)).long()

    def __getitem__(self, index):
        return self.input_data[index], self.targets[index]

    def __len__(self):
        return len(self.input_data)