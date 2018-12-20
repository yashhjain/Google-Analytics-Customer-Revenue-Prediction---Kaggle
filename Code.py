
# coding: utf-8

# In[141]:


#Authors Arya Reddy
#        Vinyas Raju
#        Vishwanath D C
#        Yash Jain
#Date: 12/02/2018
#Predict log revenue for Google Analytics Revenue PRediction Kaggle competition

#import necessary libraries
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib 
from sklearn.metrics import mean_squared_error
import math
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import seaborn as sns
from matplotlib import pyplot as plt


# In[142]:


#1. convert json to data and add new columns
def load_df(path):
    df = pd.read_csv(path, dtype={'fullVisitorId': 'str'})
    json_columns = ['totals','trafficSource','device','geoNetwork']
    for json_col in json_columns: 
        in_df = pd.DataFrame(df.pop(json_col).apply(pd.io.json.loads).values.tolist(), index=df.index)
        df = df.join(in_df) 
     
    #json columns of columns
    in_df = pd.DataFrame(df.pop('adwordsClickInfo').values.tolist(), index=df.index)
    df = df.join(in_df)     
    return df


# In[143]:


#load csv file to a dataframe
df = load_df(r"D:\my_subjects_sem1\Anurag\Final_Project\dataset\train.csv")


# In[144]:


#load csv file to a dataframe
test_data = load_df(r"D:\my_subjects_sem1\Anurag\Final_Project\dataset\test.csv")


# In[145]:


#replace all empty fields with NaN
df.replace('(not set)', np.nan)
df.replace("not available in demo dataset", np.nan)
df.replace('(not provided)', np.nan)
df.replace('unknown.unknown', np.nan)
df.replace('(none)', np.nan)
df.replace('/', np.nan)
df.replace('Not Socially Engaged', np.nan)
test_data.replace('(not set)', np.nan)
test_data.replace("not available in demo dataset", np.nan)
test_data.replace('(not provided)', np.nan)
test_data.replace('unknown.unknown', np.nan)
test_data.replace('(none)', np.nan)
test_data.replace('/', np.nan)
test_data.replace('Not Socially Engaged', np.nan)


# In[146]:


#load dataframe into a text file to verify json to field conversion "comment this line if not required"
df.to_csv(r"D:\my_subjects_sem1\Anurag\Final_Project\dataset\train_revised.csv")
test_data.to_csv(r"D:\my_subjects_sem1\Anurag\Final_Project\dataset\test_revised.csv")


# In[147]:


df.head(5)


# In[148]:


#drop all columns which have all null values
df.dropna(axis=1, how='all')
test_data.dropna(axis=1, how='all')


# In[149]:


df.head(5)


# In[150]:


test_data.head(5)


# In[151]:


#hits and pageviews are the only numerical values, convert them to numpy object float
df["hits"].fillna(0, inplace=True)
df["hits"] = df["hits"].astype('float')
df["pageviews"].fillna(0, inplace=True)
df["pageviews"] = df["pageviews"].astype('float')


# In[152]:


#hits and pageviews are the only numerical values, convert them to numpy object float
test_data["hits"].fillna(0, inplace=True)
test_data["hits"] = df["hits"].astype('float')
test_data["pageviews"].fillna(0, inplace=True)
test_data["pageviews"] = df["pageviews"].astype('float')


# In[153]:


test_data.shape


# In[154]:


#convert date to year, month and day
df['date'] = pd.to_datetime((df['date']).astype(str), format = '%Y-%m-%d')
df[['year', 'month', 'day']] = df['date'].astype(str).str.split('-', expand = True)    
df.drop('date', axis =1,inplace = True)


# In[155]:


#convert date to year, month and day
test_data['date'] = pd.to_datetime((test_data['date']).astype(str), format = '%Y-%m-%d')
test_data[['year', 'month', 'day']] = test_data['date'].astype(str).str.split('-', expand = True)    
test_data.drop('date', axis =1,inplace = True)


# In[156]:


#apply natural log on transactionRevenue which gives transactionRevenue_log, this is label to predict 
df["transactionRevenue"].fillna(0, inplace=True)
df["transactionRevenue"] = df["transactionRevenue"].astype('float')
df['transactionRevenue_log'] = np.log(df[df['transactionRevenue'] > 0]["transactionRevenue"] + 0.01)


# In[157]:


#copy dataframe to temporary dataframe to build model
train_df = df.copy()
test_df = test_data.copy()

#Label to predict
y_label = train_df['transactionRevenue_log'].fillna(0)
y_label.head(5)


# In[158]:


#drop columns which donot help in training
train_df.drop('fullVisitorId', axis=1, inplace=True)
train_df.drop('sessionId', axis=1, inplace=True)
train_df.drop('transactionRevenue', axis=1, inplace=True)
train_df.drop('visitId', axis=1, inplace=True)
train_df.drop('visitStartTime', axis=1, inplace=True)
train_df.drop('transactionRevenue_log', axis=1, inplace=True)
train_df.drop('visits', axis=1, inplace=True)
train_df.drop('visitNumber', axis=1, inplace=True)
train_df.drop('isTrueDirect', axis=1, inplace=True)
train_df.drop('adContent', axis=1, inplace=True)
train_df.drop('networkDomain', axis=1, inplace=True)
train_df.drop('isVideoAd', axis=1, inplace=True)
train_df.drop('gclId', axis=1, inplace=True)
train_df.drop('slot', axis=1, inplace=True)
train_df.drop('adNetworkType', axis=1, inplace=True)
train_df.drop('keyword', axis=1, inplace=True)
train_df.drop('referralPath', axis=1, inplace=True)
train_df.drop('page', axis=1, inplace=True)
train_df.drop('campaignCode', axis=1, inplace=True)
train_df.drop('newVisits', axis=1, inplace=True)
train_df.drop('bounces', axis=1, inplace=True)
train_df.drop('targetingCriteria', axis=1, inplace=True)

train_df.drop('socialEngagementType', axis=1, inplace=True)
train_df.drop('browserSize', axis=1, inplace=True)
train_df.drop('browserVersion', axis=1, inplace=True)
train_df.drop('flashVersion', axis=1, inplace=True)
train_df.drop('language', axis=1, inplace=True)
train_df.drop('mobileDeviceBranding', axis=1, inplace=True)
train_df.drop('mobileDeviceInfo', axis=1, inplace=True)
train_df.drop('mobileDeviceMarketingName', axis=1, inplace=True)
train_df.drop('mobileDeviceModel', axis=1, inplace=True)
train_df.drop('mobileInputSelector', axis=1, inplace=True)
train_df.drop('operatingSystemVersion', axis=1, inplace=True)
train_df.drop('screenColors', axis=1, inplace=True)
train_df.drop('screenResolution', axis=1, inplace=True)
train_df.drop('cityId', axis=1, inplace=True)
train_df.drop('longitude', axis=1, inplace=True)
train_df.drop('latitude', axis=1, inplace=True)
train_df.drop('networkLocation', axis=1, inplace=True)
train_df.drop('criteriaParameters', axis=1, inplace=True)

train_df.head(5)


# In[159]:


#drop columns which donot help in testing
test_df.drop('fullVisitorId', axis=1, inplace=True)
test_df.drop('sessionId', axis=1, inplace=True)

#test_df.drop('transactionRevenue', axis=1, inplace=True)

test_df.drop('visitId', axis=1, inplace=True)
test_df.drop('visitStartTime', axis=1, inplace=True)

#test_df.drop('transactionRevenue_log', axis=1, inplace=True)

test_df.drop('visits', axis=1, inplace=True)
test_df.drop('visitNumber', axis=1, inplace=True)
test_df.drop('isTrueDirect', axis=1, inplace=True)
test_df.drop('adContent', axis=1, inplace=True)
test_df.drop('networkDomain', axis=1, inplace=True)
test_df.drop('isVideoAd', axis=1, inplace=True)
test_df.drop('gclId', axis=1, inplace=True)
test_df.drop('slot', axis=1, inplace=True)
test_df.drop('adNetworkType', axis=1, inplace=True)
test_df.drop('keyword', axis=1, inplace=True)
test_df.drop('referralPath', axis=1, inplace=True)
test_df.drop('page', axis=1, inplace=True)

#test_df.drop('campaignCode', axis=1, inplace=True)

test_df.drop('newVisits', axis=1, inplace=True)
test_df.drop('bounces', axis=1, inplace=True)
test_df.drop('targetingCriteria', axis=1, inplace=True)

test_df.drop('socialEngagementType', axis=1, inplace=True)
test_df.drop('browserSize', axis=1, inplace=True)
test_df.drop('browserVersion', axis=1, inplace=True)
test_df.drop('flashVersion', axis=1, inplace=True)
test_df.drop('language', axis=1, inplace=True)
test_df.drop('mobileDeviceBranding', axis=1, inplace=True)
test_df.drop('mobileDeviceInfo', axis=1, inplace=True)
test_df.drop('mobileDeviceMarketingName', axis=1, inplace=True)
test_df.drop('mobileDeviceModel', axis=1, inplace=True)
test_df.drop('mobileInputSelector', axis=1, inplace=True)
test_df.drop('operatingSystemVersion', axis=1, inplace=True)
test_df.drop('screenColors', axis=1, inplace=True)
test_df.drop('screenResolution', axis=1, inplace=True)
test_df.drop('cityId', axis=1, inplace=True)
test_df.drop('longitude', axis=1, inplace=True)
test_df.drop('latitude', axis=1, inplace=True)
test_df.drop('networkLocation', axis=1, inplace=True)
test_df.drop('criteriaParameters', axis=1, inplace=True)

test_df.head(5)


# In[160]:


list(train_df.columns.values)


# In[161]:


#use labelencoder to convert string categorical values to numerical values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
train_df['channelGrouping'] = labelencoder_X.fit_transform(train_df['channelGrouping'])
train_df['campaign'] = labelencoder_X.fit_transform(train_df['campaign'])
train_df['medium'] = labelencoder_X.fit_transform(train_df['medium'])
train_df['source'] = labelencoder_X.fit_transform(train_df['source'])
train_df['browser'] = labelencoder_X.fit_transform(train_df['browser'])
train_df['deviceCategory'] = labelencoder_X.fit_transform(train_df['deviceCategory'])
train_df['isMobile'] = labelencoder_X.fit_transform(train_df['isMobile'])
train_df['operatingSystem'] = labelencoder_X.fit_transform(train_df['operatingSystem'])
train_df['city'] = labelencoder_X.fit_transform(train_df['city'])
train_df['continent'] = labelencoder_X.fit_transform(train_df['continent'])
train_df['country'] = labelencoder_X.fit_transform(train_df['country'])
train_df['metro'] = labelencoder_X.fit_transform(train_df['metro'])
train_df['region'] = labelencoder_X.fit_transform(train_df['region'])
train_df['subContinent'] = labelencoder_X.fit_transform(train_df['subContinent'])
train_df['year'] = labelencoder_X.fit_transform(train_df['year'])
train_df['month'] = labelencoder_X.fit_transform(train_df['month'])
train_df['day'] = labelencoder_X.fit_transform(train_df['day'])


# In[162]:


train_df.head()


# In[163]:


#use labelencoder to convert string categorical values to numerical values
test_df['channelGrouping'] = labelencoder_X.fit_transform(test_df['channelGrouping'])
test_df['campaign'] = labelencoder_X.fit_transform(test_df['campaign'])
test_df['medium'] = labelencoder_X.fit_transform(test_df['medium'])
test_df['source'] = labelencoder_X.fit_transform(test_df['source'])
test_df['browser'] = labelencoder_X.fit_transform(test_df['browser'])
test_df['deviceCategory'] = labelencoder_X.fit_transform(test_df['deviceCategory'])
test_df['isMobile'] = labelencoder_X.fit_transform(test_df['isMobile'])
test_df['operatingSystem'] = labelencoder_X.fit_transform(test_df['operatingSystem'])
test_df['city'] = labelencoder_X.fit_transform(test_df['city'])
test_df['continent'] = labelencoder_X.fit_transform(test_df['continent'])
test_df['country'] = labelencoder_X.fit_transform(test_df['country'])
test_df['metro'] = labelencoder_X.fit_transform(test_df['metro'])
test_df['region'] = labelencoder_X.fit_transform(test_df['region'])
test_df['subContinent'] = labelencoder_X.fit_transform(test_df['subContinent'])
test_df['year'] = labelencoder_X.fit_transform(test_df['year'])
test_df['month'] = labelencoder_X.fit_transform(test_df['month'])
test_df['day'] = labelencoder_X.fit_transform(test_df['day'])


# In[164]:


test_df.head()


# In[165]:


train_df.shape


# In[166]:


#heatmap to show correlation
corelation_train = train_df.corr() 
sns.heatmap(corelation_train, square = True)


# In[167]:


#heatmap to show correlation
corelation_test = test_df.corr() 
sns.heatmap(corelation_test, square = True)


# In[168]:


X_train, X_test, Y_train, Y_test = train_test_split(train_df ,y_label, test_size=0.2)


# In[169]:


#apply xgb
XGB_model = XGBRegressor()
XGB_model.fit(X_train, Y_train, eval_metric='rmse',verbose=130)


# In[170]:


#apply random forest
RF_model = RandomForestRegressor(n_estimators = 100, random_state = 42)
RF_model.fit(X_train, Y_train)


# In[171]:


#apply lgb
LGB_model = lgb.LGBMRegressor()
LGB_model.fit(X_train, Y_train, eval_metric='rmse', verbose=100)


# In[172]:


#build model
y_pred_XGB = XGB_model.predict(X_test)
y_pred_LGB = LGB_model.predict(X_test)
y_pred_rf = RF_model.predict(X_test)


# In[173]:


#replace all -ve values with 0
y_pred_XGB[y_pred_XGB < 0] =0
y_pred_XGB
y_pred_LGB[y_pred_LGB < 0] =0
y_pred_LGB
y_pred_rf[y_pred_rf < 0] =0
y_pred_rf


# In[174]:


plt.scatter(Y_test, y_pred_XGB)
plt.xlabel('True Values')
plt.ylabel('Predictions')


# In[175]:


plt.scatter(Y_test, y_pred_LGB)
plt.xlabel('True Values')
plt.ylabel('Predictions')


# In[176]:


plt.scatter(Y_test, y_pred_rf )
plt.xlabel('True Values')
plt.ylabel('Predictions')


# In[177]:


#evaluate using root mean squared error
print("XGBoost: sqrt  ", math.sqrt(mean_squared_error(Y_test, y_pred_XGB)))
print("LGB: sqrt  ", math.sqrt(mean_squared_error(Y_test, y_pred_LGB)))
print("RandomForest: sqrt  ", math.sqrt(mean_squared_error(Y_test, y_pred_rf)))


# In[178]:


#predict revenue log on test dataset
final_result= LGB_model.predict(test_df)


# In[179]:


final_result.size


# In[185]:


#set all negative values to 0
final_result[final_result < 0] =0
final_result


# In[189]:


#take visitorId list to add corresponding logrevenues
fullVisitorId_1 = test_data['fullVisitorId']
fullVisitorId_1.size


# In[190]:



# submit to csv file for kaggle submission
submit_1 = pd.DataFrame({'fullVisitorId': fullVisitorId_1,'PredictedLogRevenue':final_result},columns=['fullVisitorId','PredictedLogRevenue'])


# In[191]:


submit_1 = submit_1.groupby('fullVisitorId').sum().reset_index()

# display user vs transaction for predicted values
x = range(submit_1.shape[0])
y = np.sort((submit_1["PredictedLogRevenue"].values))
plt.figure(figsize=(8,6))
plt.scatter(x, y)
plt.xlabel('Index')
plt.ylabel('Log Transaction Revenue')
plt.show()


# In[192]:


submit_1.to_csv(r"D:\my_subjects_sem1\Anurag\Final_Project\result_2.csv",index=False)

