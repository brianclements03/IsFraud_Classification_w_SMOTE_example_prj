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


# In[ ]:




