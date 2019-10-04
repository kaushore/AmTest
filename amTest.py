#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 23:37:17 2019

@author: kaushal
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
#le = preprocessing.LabelEncoder()

campaign_data = pd.read_csv("train/campaign_data.csv")
coupon_item_mapping = pd.read_csv("train/coupon_item_mapping.csv")
customer_demographics = pd.read_csv("train/customer_demographics.csv")
customer_transaction_data= pd.read_csv("train/customer_transaction_data.csv")
item_data = pd.read_csv("train/item_data.csv")
df_train = pd.read_csv('train/train.csv')
#smdft = df_train.sample(100)
customer_demographics = customer_demographics.drop(["marital_status","no_of_children"],axis = 1)

campaign_data["sd"]= pd.to_numeric(campaign_data.start_date.str.split('/').str[0])
campaign_data["sm"]= pd.to_numeric(campaign_data.start_date.str.split('/').str[1])
campaign_data["sy"]= pd.to_numeric(campaign_data.start_date.str.split('/').str[2])
campaign_data["ed"]= pd.to_numeric(campaign_data.end_date.str.split('/').str[0])
campaign_data["em"]= pd.to_numeric(campaign_data.end_date.str.split('/').str[1])
campaign_data["ey"]= pd.to_numeric(campaign_data.end_date.str.split('/').str[2])
campaign_data = campaign_data.drop(["start_date","end_date"],axis = 1)

#cols = list(customer_demographics.columns)
#for index, row in customer_demographics.iterrows():
#    print(index)
    
cdf = pd.merge(df_train,customer_demographics,on='customer_id')
cdf = pd.merge(cdf,coupon_item_mapping,on='coupon_id')
cdf = pd.merge(cdf,campaign_data,on='campaign_id')
cdf = pd.merge(cdf,item_data,on='item_id')
#cdf = pd.merge(cdf,customer_transaction_data ,on='customer_id') 

cdf = cdf.drop(['id','campaign_id', 'coupon_id', 'customer_id','item_id'],axis = 1)

m = np.zeros(len(cdf))
m[cdf.family_size == '5+'] = 5
m[cdf.family_size == '4'] = 4
m[cdf.family_size == '3'] = 3
m[cdf.family_size == '2'] = 2
m[cdf.family_size == '1'] = 1
cdf.drop('family_size',axis =1)
cdf.family_size = m

m = np.zeros(len(cdf))
m[cdf.age_range == '46-55'] = 50 
m[cdf.age_range == '18-25'] = 20
m[cdf.age_range == '36-45'] = 40
m[cdf.age_range == '26-35'] = 30
m[cdf.age_range == '70+'] = 70
cdf.drop('age_range',axis =1)
cdf.age_range = m

del campaign_data
del coupon_item_mapping
del customer_demographics
del customer_transaction_data
del item_data
del df_train

#samplecdf = cdf.sample(200)
#samplecdf = pd.get_dummies(samplecdf)

cdf = pd.get_dummies(cdf)
#le.fit([1, 2, 2, 6])


#del samplecdf
y = cdf['redemption_status'].values
X = cdf.loc[:, cdf.columns != 'redemption_status']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=42)
feature_names = X.columns.tolist()



#categorical = ['age_range', 'rented', 'family_size',
#       'income_bracket', 'sd', 'sm', 'sy', 'ed', 'em', 'ey', 'brand',
#       'campaign_type_X', 'campaign_type_Y', 'brand_type_Established',
#       'brand_type_Local', 'category_Bakery',
#       'category_Dairy, Juices & Snacks', 'category_Flowers & Plants',
#       'category_Garden', 'category_Grocery', 'category_Meat',
#       'category_Miscellaneous', 'category_Natural Products',
#       'category_Packaged Meat', 'category_Pharmaceutical',
#       'category_Prepared Food', 'category_Restauarant', 'category_Salads',
#       'category_Seafood', 'category_Skin & Hair Care', 'category_Travel',
#       'category_Vegetables (cut)']


#Convert data into LightGBM dataset format. This is mandatory for LightGBM training.
lgtrain = lgb.Dataset(X_train, y_train,
                feature_name=feature_names)

lgvalid = lgb.Dataset(X_valid, y_valid,
                feature_name=feature_names)


params = {
    'objective' : 'binary',
    'metric' : 'binary_logloss',
    'num_leaves' : 200,
    'max_depth': 15,
    'learning_rate' : 0.01,
    'feature_fraction' : 0.6,
    'verbosity' : -1
}
lgb_clf = lgb.train(
    params,
    lgtrain,
    num_boost_round = 5000,
    valid_sets=[lgtrain, lgvalid],
    valid_names=["train", "valid"],
    early_stopping_rounds=50,
    verbose_eval=50
)

lgb_clf.save_model("model.text")
json_model = lgb_clf.dump_model()
print("RMSE of the validation set:", np.sqrt(mean_squared_error(y_valid, lgb_clf.predict(X_valid))))

## test Part


