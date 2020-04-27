#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings 
warnings.filterwarnings(action='ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train=pd.read_csv('C:/Users/study gil/Documents/train.csv')
print(train.shape)
train.columns


# In[3]:


test=pd.read_csv('C:/Users/study gil/Documents/test.csv')


# In[4]:


train_30day = train.iloc[0:4320,:]
train_30day.shape


# In[5]:


train_3day = train.iloc[4320:4753,:]
train_3day.shape


# In[6]:


train_30day.head(5)


# In[6]:


target = ['Y00', 'Y01', 'Y02', 'Y03', 'Y04', 'Y05', 'Y06', 'Y07', 'Y08','Y09', 'Y10', 'Y11', 'Y12', 'Y13', 'Y14', 'Y15', 'Y16', 'Y17']
total_target = train_30day[target]
total_target.describe()
#y 3,4는 std가 유독 작음 = 실내 센서일 것임


# In[7]:


correlations = total_target.corr()


# In[8]:


plt.figure(figsize = (14, 12))
sns.heatmap(correlations,  vmin = 0.2, annot = True, vmax = 0.9);
# y들간의 높은 상관계수 = > clustering 방법을 통한 데이터 분석 적절해보임


# In[10]:


plt.figure(figsize = (10, 5))
plt.plot(train_30day.Y04,label='Y04')
plt.plot(train_3day.Y18,label='Y18')
plt.legend();
#y 3,4과는 매우 양상을 띄는 것을 예시로 볼 수 있음 


# In[11]:


no_y18_target = ['Y00', 'Y01', 'Y02', 'Y03', 'Y04', 'Y05', 'Y06', 'Y07', 'Y08','Y09', 'Y10', 'Y11', 'Y12', 'Y13', 'Y14', 'Y15', 'Y16', 'Y17']
no_y18 = train_30day[no_y18_target]
x_local_pressure = ['X01', 'X06', 'X22', 'X27', 'X29']
temperature = train_30day[x_local_pressure]
no_y18_winddir = pd.concat((no_y18, temperature), axis=1)
cor= no_y18_winddir.corr()
plt.figure(figsize = (15, 15))
sns.heatmap(cor,  vmin = 0.2, annot = True, vmax = 0.9);
#실제로 y 3,4는 풍향과 매우 낮은 correlation = 실내 관측치 일것


# In[13]:


plt.figure(figsize=(10,5))
plt.plot(train_30day['Y00']);
#특정 데이터를 제외하고는 강한 반복 패턴을 보이고 평균도 비슷함


# In[14]:


from statsmodels.tsa.stattools import adfuller
y_00 = train_30day['Y00']
adfuller(y_00)
#it is stationary


# In[15]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(y_00)
plot_acf(y_00,lags=100)
plot_pacf(y_00)
plt.show()
# nonstationary 하게 보이지만 10분 단위인 데이터를 생각해보면 12시간 기준으로 up down 하는 패턴을 생각해보면 아마 한 패턴이 지나지 않을것임


# In[16]:


diff_1=y_00.diff(periods=1).iloc[1:]
plot_acf(diff_1)
plot_pacf(diff_1)
plt.show()


# In[18]:


import pyramid
from pyramid.arima import auto_arima


# In[19]:


import pmdarima as pm
def arimamodel(timeseries):
    automodel = pm.auto_arima(timeseries, 
                              start_p=1, 
                              start_q=1,
                              test="adf",
                              seasonal = True,
                              trace=True)
    return automodel


# In[24]:


index= pd.date_range('20130101', periods=4320, freq='10T')
train_30day_time = pd.DataFrame(train_30day.values,index=index, columns = train.columns)
#make ts 
train_30day_time.head()


# In[25]:


train_30day_time_day = train_30day_time.resample('D').mean() #make average temperauture a day
train_30day_time_day.head()


# In[26]:


plt.figure(figsize = (10,5))
plt.plot(train_30day_time_day['Y00']);#also stationary


# In[27]:


adfuller(train_30day_time_day['Y00'])
#it is stationary


# In[30]:


model1= arimamodel(train_30day_time_day['Y00'])


# In[31]:


model = auto_arima(train_30day_time_day['Y00'],
                   start_p=1, 
                   start_q=1, 
                   max_p=10, 
                   max_q=10,
                   trace=True, 
                   error_action='ignore',
                   suppress_warnings=True)
model.fit(train_30day_time_day['Y00'])


# In[32]:


model1.predict(n_periods=3)


# In[60]:


train_30day_time_hour = train_30day_time.resample('H').mean() #make average temperauture an hour
print(train_30day_time_hour.shape)
model = auto_arima(train_30day_time_hour['Y00'],
                   start_p=1, 
                   start_q=1, 
                   max_p=10, 
                   max_q=10,
                   trace=True, 
                   error_action='ignore',
                   suppress_warnings=True)
model1 = model.fit(train_30day_time_hour['Y00'])


# In[39]:


model = auto_arima(train_30day_time['Y00'],
    start_p=1, start_q=1, max_p=10, max_q=10,
    trace=True, error_action='ignore',suppress_warnings=True)


# In[61]:


model.fit(train_30day_time['Y00'])


# In[58]:


plt.plot(train_30day_time_hour['Y00']);


# In[59]:


plot_acf(train_30day_time_hour['Y00'])
plot_pacf(train_30day_time_hour['Y00'])
plt.show()


# In[36]:


def arimamodelling(timeseries):
    automodel = auto_arima(timeseries,
                           start_p=1, 
                           start_q=1, 
                           max_p=10, 
                           max_q=10,
                           trace=True, error_action='ignore',suppress_warnings=True)
    return automodel


# In[69]:


target_index = ['Y00', 'Y01', 'Y02', 'Y03', 'Y04', 'Y05', 'Y06', 'Y07', 'Y08','Y09', 'Y10', 'Y11', 'Y12', 'Y13', 'Y14', 'Y15', 'Y16', 'Y17']
error = list()


# In[79]:


#y00 부터 y17 까지 예측값에 대한 오차제곱합 계산
for i in target_index:
    model= arimamodelling(train_30day[i])
    model = model.fit(train_30day[i])
    predict = model.predict(n_periods=432)
    error.append(sum((predict-train_3day['Y18'])**2))


# In[84]:


model = auto_arima(train_30day['Y00'],start_p=1, start_q=1, max_p=10, max_q=10,trace=True, error_action='ignore',suppress_warnings=True)


# In[85]:


model.fit(train_30day['Y00'])


# In[86]:


forecasting = model.predict(n_periods = 432)
sum((forecasting-train_3day['Y18'])**2)
#일치함


# In[87]:


model = auto_arima(train_30day['Y15'],start_p=1, start_q=1, max_p=10, max_q=10,trace=True, error_action='ignore',suppress_warnings=True)


# In[89]:


model.fit(train_30day['Y15'])
forecasting = model.predict(n_periods = 432)
sum((forecasting-train_3day['Y18'])**2)
#일치함


# In[90]:


model = auto_arima(train_30day['Y17'],start_p=1, start_q=1, max_p=10, max_q=10,trace=True, error_action='ignore',suppress_warnings=True)


# In[91]:


model.fit(train_30day['Y17'])
forecasting = model.predict(n_periods = 432)
sum((forecasting-train_3day['Y18'])**2)
#일치함


# In[97]:


error = pd.Series(error[-18:],index=target_index) 
error
#01, 02, 05, 08, 10, 11, 16, 17 are close to 18


# In[9]:


no_y18_target = ['Y00', 'Y01', 'Y02', 'Y03', 'Y04', 'Y05', 'Y06', 'Y07', 'Y08','Y09', 'Y10', 'Y11', 'Y12', 'Y13', 'Y14', 'Y15', 'Y16', 'Y17']
no_y18 = train[(train['Y18'].isnull()) & (train['id'] > 3887)][no_y18_target].reset_index(drop = True)
y18 = train[~train['Y18'].isnull()]['Y18'].reset_index(drop= True)
check_target = pd.concat([no_y18, y18], axis = 1)
check_target.head()


# In[110]:


correlations =  check_target.corr()
plt.figure(figsize = (14, 12))

# Heatmap of correlations
sns.heatmap(correlations, cmap = plt.cm.RdYlBu_r,  vmin = 0.2, annot = True, vmax = 0.9)
plt.title('Correlation Heatmap');


# In[34]:


target_index = ['Y00', 'Y01', 'Y02', 'Y03', 'Y04', 'Y05', 'Y06', 'Y07', 'Y08','Y09', 'Y10', 'Y11', 'Y12', 'Y13', 'Y14', 'Y15', 'Y16', 'Y17']
df_forecast = pd.DataFrame(0, index = np.arange(1,433),columns = target_index)
df_forecast.shape


# In[39]:


for i in target_index:
    model= arimamodelling(train_30day[i])
    model.fit(train_30day[i])
    predict = model.predict(n_periods=432)
    df_forecast[i] = predict


# In[54]:


df_forecast.index = index
df_forecast.tail()


# In[56]:


index = np.arange(4320,4752)
predict_y18 = pd.concat((df_forecast, train_3day['Y18']), axis=1)
predict_y18.head()


# In[60]:


plt.figure(figsize=(15,15))
predict_cor = predict_y18.corr()
sns.heatmap(predict_cor,  vmin = 0.2, annot = True, vmax = 0.9);

