#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import datetime
from imblearn.over_sampling import SMOTE

from zipfile import ZipFile


# In[2]:


def acquire_fraud(zip_file_name_in_quotes):
    '''
    Create a df from zipped file source. Requires ZipFile import
    '''
    zip_file = ZipFile(zip_file_name_in_quotes)
    df_test = pd.read_csv(zip_file.open(zip_file.namelist()[0]))
    df_train = pd.read_csv(zip_file.open(zip_file.namelist()[1]))
    return df_test, df_train


# In[3]:


def prep(df):
    '''
    Apply some clean and prep to the df
    '''
    df = df.set_index(pd.to_datetime(df['trans_date_trans_time'],format= '%Y-%m-%d %H:%M:%S')).sort_index()
    df['age'] = (df.index - pd.DatetimeIndex(df['dob']))// pd.Timedelta('365D')
    df['dayofweek'] = df.index.day_name()
    df['hourofday'] = df.index.hour
    df = df.drop(columns=['Unnamed: 0','trans_date_trans_time','cc_num','first','last','street','city','state','trans_num','lat','long','dob','trans_num','unix_time'])
    df['age_group'] = pd.cut(df['age'],[0,25,35,45,55,65,75,100], labels= ['Youth','Young_Adult','Adult','Early_Mid_Age','Mid_Age','Retirement_Age','Older_Person'],right=False)
    
    return df


# In[4]:


def get_target_and_features(df):
    '''
    A UDF to define the target variable and the rest
    '''
    target = df.columns.to_list()[9]
    features = df.columns[df.columns != target].to_list()
    return target,features


# In[5]:


def train_val(df):
    '''
    A UDF to split the train data set into train and validate (both for x and y variables)
    '''
    target, features = get_target_and_features(df)
    y = df[target]
    x = df[features]
    x_train, x_validate, y_train, y_validate = train_test_split(x,y,test_size=.30, random_state=42)
    return x_train,x_validate,y_train,y_validate


# In[6]:


def test_df_x_y_split(df):
    '''
    A UDF to define the target and featurs and define the X and Y dataframes for the TEST dataset
    '''
    target, features = get_target_and_features(df)
    x_test = df[features]
    y_test = df[target]
    
    return x_test,y_test

