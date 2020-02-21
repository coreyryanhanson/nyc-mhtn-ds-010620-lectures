#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import numpy as np
from sklearn import metrics
import time
import os
import glob
import re


# In[14]:


#read in the answers
answers = pd.read_csv('test_actuals.csv', index_col = 0, header=None, names=['price'] )


# In[ ]:


# create a list of all file names
# starting with testlabels_ from the file


# In[33]:



path = '/Users/swilson5/Documents/DSC/ds-010620/nyc-ds-010620-lectures/week-6/KC_house_data/answers'
extension = 'csv'
os.chdir(path)
files = glob.glob('*.{}'.format(extension))
print(files)



# In[35]:


#create dictionary to store results
results = {}


# In[ ]:


# loop over the list read the file, and score the answers


# In[36]:


for file in files:
    name = file.split("_")[-1].split(".")[0]
    print(name)
    guesses = pd.read_csv(file, index_col=0, header=None )
    # guesses.sort_index(inplace=True)

    if guesses.shape[0] != 4324:
        print(name, "not the correct size", guesses.shape[0]  )
        # continue
    mse = metrics.mean_squared_error(answers,guesses, average='weighted')
    rmse = np.sqrt(mse)
    results[name]= [mse, rmse]


# In[46]:


sorted_results = sorted(results.items(), key=lambda x: x[1][0])

print('Now printing the results with in ascending order')
# In[50]:
# time.sleep(10)

for k,v in sorted_results:
#     print(k,v)
    print('{} achieved an RMSE of {} and an accuracy score of {} \n'.format(
          k.upper(), v[1]))

    time.sleep(1)
