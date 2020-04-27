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


# In[3]:


test=pd.read_csv('C:/Users/study gil/Documents/test.csv')


# In[19]:


train_30day = train.iloc[0:4320,:]
train_30day.shape


# In[5]:


train_3day = train.iloc[4320:4753,:]
train_3day.shape


# In[6]:


#x칼럼 정리.
x_time = ['id'] #시간
x_temperature = ['X00', 'X07', 'X28', 'X31', 'X32'] #기온
x_local_pressure = ['X01', 'X06', 'X22', 'X27', 'X29'] #현지기압
x_wind_speed = ['X02', 'X03', 'X18', 'X24', 'X26'] #풍속
x_daily_rainfall = ['X04', 'X10', 'X21', 'X36', 'X39'] #일일 누적강수량
x_sealevel_pressure = ['X05', 'X08', 'X09', 'X23', 'X33'] #해면기압
x_daily_sun = ['X11', 'X34'] #일일 누적일사량 (X14, X16, X19는 제거됨)
x_humidity = ['X12', 'X20', 'X30', 'X37', 'X38'] #습도
x_wind_direction = ['X13', 'X15', 'X17', 'X25', 'X35'] #풍향

#y칼럼 정리
y_sensor = ['Y00', 'Y01','Y02','Y03','Y04', 'Y05', 'Y06', 'Y07', 'Y08', 
            'Y09', 'Y10', 'Y11', 'Y12', 'Y13', 'Y14', 'Y15', 'Y16', 'Y17'] #센서측정온도
y_target = ['Y18'] #예측대상


#id값들을 시간형 데이터로 변경하는 함수 생성
def make_timedata(df):
    minute = (df['id'] % 144).astype(int)
    min_in_day = 24*6

    df['id'] = np.cos(np.pi * minute / min_in_day)
print("id를 cos데이터로 변경 완료")


# In[7]:


train_30day = train_30day.drop(['X14', 'X16', 'X19'], axis =1)
train_3day = train_3day.drop(['X14', 'X16', 'X19'], axis =1)


# In[8]:


train_3day.shape


# In[9]:


#4

"""
계산식을 변경하고 싶으면 
train[col_num].mean(axis = 1)를 변경해주세요. (초기 설정은 평균입니다)
"""

#train데이터에서 x값 대표를 만드는 함수. 일단은 x값들의 평균값.
def make_train_x(df, col_name, col_num=[]):
    df[col_name] = pd.Series(train[col_num].mean(axis = 1))

#train데이터에서 y값 대표를 만드는 함수. (평균)
def make_train_y(df, col_name, col_num=[]):
    df[col_name] = pd.Series(train[col_num].mean(axis = 1))

#test 데이터에서 x값 대표를 만드는 함수. 일단은 x값들의 평균값
def make_test_x(df, col_name, col_num=[]):
    df[col_name] = pd.Series(test[col_num].mean(axis = 1))


# In[10]:


#학습용 데이터프레임 생성
train_df = pd.DataFrame()

#id값 추가
train_df['id'] = train_30day['id']

#id값을 시간형 데이터로 변경
make_timedata(train_df)

"""
#X 대표값 추가
make_train_x(train_df, 'x_temperature', x_temperature)
make_train_x(train_df, 'x_local_pressure', x_local_pressure)
make_train_x(train_df, 'x_wind_speed', x_wind_speed)
make_train_x(train_df, 'x_daily_rainfall', x_daily_rainfall)
make_train_x(train_df, 'x_sealevel_pressure', x_sealevel_pressure)
make_train_x(train_df, 'x_daily_sun', x_daily_sun)
make_train_x(train_df, 'x_humidity', x_humidity)
make_train_x(train_df, 'x_wind_direction', x_wind_direction)
"""
#x변수의 개수. 모든 값 이용시 38, 평균값 이용시 9 -> 값 변경 시, train_df부터 model까지 일괄 변경
x_num = 38 

#X값 전체를 추가
train_df[train_30day.columns[1 : x_num]] = train_30day.iloc[:, 1 : x_num]

#Y값 추가 (30일의 Y값)
make_train_y(train_df, 'Y', y_sensor)

train_df.shape


# In[11]:


#현지기압 graph
plt.figure(figsize=(10,10))
plt.plot(train_30day['X00'],label='00')
plt.plot(train_30day['X07'],label='07')
plt.plot(train_30day['X28'],label='28')
plt.plot(train_30day['X31'],label='31')
plt.plot(train_30day['X32'],label='32')
plt.legend();


# In[11]:


no_y18_target = ['Y00', 'Y01', 'Y02', 'Y03', 'Y04', 'Y05', 'Y06', 'Y07', 'Y08','Y09', 'Y10', 'Y11', 'Y12', 'Y13', 'Y14', 'Y15', 'Y16', 'Y17']
no_y18 = train_30day[no_y18_target]
x_temperature = ['X00', 'X07', 'X28', 'X31', 'X32'] #현지기압
temperature = train_30day[x_temperature]
no_y18_localpres = pd.concat((temperature,no_y18), axis=1)
cor= no_y18_localpres.corr()
plt.figure(figsize = (19, 19))
sns.heatmap(cor,  vmin = 0.2, annot = True, vmax = 0.9);


# In[ ]:





# In[13]:


import statsmodels.api as sm

reg = sm.OLS.from_formula("Y ~ X00 + X07 + X28 + X31 + X32", train_df).fit()
reg.summary()


# In[54]:


plt.figure(figsize=(10,10))
x_sealevel_pressure = ['X05', 'X08', 'X09', 'X23', 'X33'] 
plt.plot(train_30day[x_sealevel_pressure]);


# In[ ]:





# In[55]:


x_sealevel_pressure = ['X05', 'X08', 'X09', 'X23', 'X33'] #해면기압
sealevel_pressure = train_30day[x_sealevel_pressure]
x_local_pressure = ['X01', 'X06', 'X22', 'X27', 'X29'] #현지기압
local_pressure = train_30day[x_local_pressure]
seapres_localpres = pd.concat((sealevel_pressure,local_pressure), axis=1)
cor= seapres_localpres.corr()
plt.figure(figsize = (10, 10))
sns.heatmap(cor,  vmin = 0.2, annot = True, vmax = 0.9);
#현지기압, 해면기압 매우 높은 상관계수를 보임 = 


# In[56]:


no_y18_target = ['Y00', 'Y01', 'Y02', 'Y03', 'Y04', 'Y05', 'Y06', 'Y07', 'Y08','Y09', 'Y10', 'Y11', 'Y12', 'Y13', 'Y14', 'Y15', 'Y16', 'Y17']
no_y18 = train_30day[no_y18_target]
x_sealevel_pressure = ['X05', 'X08', 'X09', 'X23', 'X33'] #해면기압 
sealevel_pressure = train_30day[x_sealevel_pressure]
no_y18_seapres = pd.concat((no_y18,sealevel_pressure), axis=1)
cor= no_y18_seapres.corr()
plt.figure(figsize = (19, 19))
sns.heatmap(cor,  vmin = 0.2, annot = True, vmax = 0.9);


# In[57]:


no_y18_target = ['Y00', 'Y01', 'Y02', 'Y03', 'Y04', 'Y05', 'Y06', 'Y07', 'Y08','Y09', 'Y10', 'Y11', 'Y12', 'Y13', 'Y14', 'Y15', 'Y16', 'Y17']
no_y18 = train_30day[no_y18_target]
x_local_pressure = ['X01', 'X06', 'X22', 'X27', 'X29'] #현지기압
local_pressure = train_30day[x_local_pressure]
no_y18_localpres = pd.concat((no_y18,local_pressure), axis=1)
cor= no_y18_localpres.corr()
print(abs(cor.iloc[18:23,:18]).sum(axis=1))
print(abs(cor.iloc[18:23,:18]).sum(axis=0))


# In[77]:


no_y18_target = ['Y00', 'Y01', 'Y02', 'Y03', 'Y04', 'Y05', 'Y06', 'Y07', 'Y08','Y09', 'Y10', 'Y11', 'Y12', 'Y13', 'Y14', 'Y15', 'Y16', 'Y17']
no_y18 = train_30day[no_y18_target]
x_local_pressure = ['X01', 'X06', 'X22', 'X27', 'X29'] #현지기압
local_pressure = train_30day[x_local_pressure]
no_y18_localpres = pd.concat((no_y18,local_pressure), axis=1)
cor= no_y18_localpres.corr()
cor.iloc[5:,:5]


# In[20]:


y18 = train_30day[y_sensor]
x_temperature = ['X00', 'X07', 'X28', 'X31', 'X32'] 
temperature = train_30day[x_temperature]
no_y18_localpres = pd.concat((temperature,y18), axis=1)
cor= no_y18_localpres.corr()
plt.figure(figsize = (19, 19))
sns.heatmap(cor,  vmin = 0.2, annot = True, vmax = 0.9);
# 대체로 다 높음 (Y3,4 빼고) = 단순 평균해도 될듯?


# In[21]:


y18 = train_3day['Y18']
x_temperature = ['X00', 'X07', 'X28', 'X31', 'X32'] 
temperature = train_3day[x_temperature]
no_y18_localpres = pd.concat((temperature,y18), axis=1)
cor= no_y18_localpres.corr()
plt.figure(figsize = (19, 19))
sns.heatmap(cor,  vmin = 0.2, annot = True, vmax = 0.9);
# 대체로 다 높음 = 3일치의 데이터를 감안하면 별 차이가 없는 것 같음

