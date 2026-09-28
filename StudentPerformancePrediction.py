#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,StandardScaler,MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


# In[10]:


data=pd.read_csv(r"C:\Users\rites\Downloads\student_dataset_10000_rows.csv")


# In[11]:


data


# In[17]:


data.columns


# In[12]:


data.info()


# In[13]:


data.isnull().sum()


# In[18]:


data.shape


# In[19]:


data.describe()


# In[32]:


data["placement_status"]=data["placement_status"].map({'Placed':1,'Not Placed':0})


# In[33]:


#Features
X=data.drop(['placement_status'],axis=1) #input
#Label
y=data['placement_status'] #output


# In[34]:


X


# In[35]:


y


# In[36]:


x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[39]:


model=DecisionTreeRegressor(
    max_depth=5,
    random_state=42
)


# In[38]:





# In[40]:


model.fit(x_train,y_train)


# In[41]:


data.columns


# In[42]:


new_student=[[12,
              70,
              8,
              7,
              8,
              80,
              80
             ]]


# In[44]:


predicted_score=model.predict(new_student)


# In[45]:


predicted_score

