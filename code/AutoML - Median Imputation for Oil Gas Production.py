#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import h2o
from h2o.automl import H2OAutoML
h2o.init()


# In[ ]:


data = pd.read_excel("MI_dataset_final.xlsx")


# In[ ]:


data_f = h2o.H2OFrame(data)


# In[ ]:


train, valid = data_f.split_frame(ratios = [.9])


# In[ ]:


testing_data = pd.read_excel("/Users/sreeya/Documents/Uni/Thesis/cleaned/cleaned.xlsx")


# In[ ]:


testing_data = testing_data.dropna()


# In[ ]:


testing_data.isnull().sum()


# In[ ]:


data_pred = h2o.H2OFrame(testing_data)


# In[ ]:


x = train.columns
y = "Oil (m3)"
x.remove(y)
x.remove('Gas (m3)')


# In[ ]:


aml = H2OAutoML(max_runtime_secs = 3600, sort_metric = "RMSE", include_algos = ["GBM", "DRF"],nfolds = 0)
aml.train(x=x, y=y, training_frame=train,validation_frame=valid)


# In[ ]:


lb = aml.leaderboard
lb.head(rows=lb.nrows)


# In[ ]:


h = h2o.get_model("GBM_grid_1_AutoML_2_20220522_144738_model_259")


# In[ ]:


preds = h.predict(data_pred)


# In[ ]:


preds


# In[ ]:


data_as_list = h2o.as_list(preds, use_pandas=False)
pr = h2o.as_list(preds)


# In[ ]:


sqrt(mean_squared_error(pr['predict'], testing_data['Oil (m3)'])) 


# In[ ]:


h2 = h2o.get_model("DRF_1_AutoML_2_20220522_144738")


# In[ ]:


preds2 = h2.predict(data_pred)


# In[ ]:


preds2


# In[ ]:


data_as_list2 = h2o.as_list(preds2, use_pandas=False)
pr2 = h2o.as_list(preds2)


# In[ ]:


sqrt(mean_squared_error(pr2['predict'], testing_data['Oil (m3)'])) 


# # Gas

# In[ ]:


x1 = train.columns
y1 = "Gas (m3)"
x1.remove(y1)
x1.remove('Oil (m3)')


# In[ ]:


aml = H2OAutoML(max_runtime_secs = 60, sort_metric = "RMSE", include_algos = ["GBM", "DRF"],nfolds = 0)
aml.train(x=x1, y=y1, training_frame=train,validation_frame=valid)


# In[ ]:


lb = aml.leaderboard
lb.head(rows=lb.nrows)


# In[ ]:


h = h2o.get_model("GBM_grid_1_AutoML_3_20220509_204619_model_1")


# In[ ]:


preds = h.predict(data_pred)


# In[ ]:


data_as_list = h2o.as_list(preds, use_pandas=False)
pr = h2o.as_list(preds)


# In[ ]:


pr


# In[ ]:


sqrt(mean_squared_error(pr['predict'], testing_data['Gas (m3)'])) 


# In[ ]:


h2 = h2o.get_model("DRF_1_AutoML_3_20220509_204619")


# In[ ]:


preds2 = h2.predict(data_pred)


# In[ ]:


preds2


# In[ ]:


data_as_list2 = h2o.as_list(preds2, use_pandas=False)
pr2 = h2o.as_list(preds2)


# In[ ]:


sqrt(mean_squared_error(pr2['predict'], testing_data['Gas (m3)'])) 

