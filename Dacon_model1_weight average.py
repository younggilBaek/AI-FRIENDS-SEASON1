#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
import os

warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[87]:


#1
#csv파일 불러오기
train = pd.read_csv('C:/Users/study gil/Documents/train.csv')
test = pd.read_csv('C:/Users/study gil/Documents/test.csv')
    
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


# In[88]:


#2

#고장난 센서 X14, X16, X19 삭제
train = train.drop(['X14', 'X16', 'X19'], axis =1)
test = test.drop(['X14', 'X16', 'X19'], axis =1)
print('X14, X16, X19 삭제 완료')


# In[89]:


# 데이터 정규화
from sklearn.preprocessing import StandardScaler  


# In[90]:


#3

#풍향을 0은 1, 360은 -1로 가지는 값으로 변경
def make_wind_direction(df):
    df['X13'] = np.cos(df['X13'] * np.pi /360)
    df['X15'] = np.cos(df['X15'] * np.pi /360)
    df['X17'] = np.cos(df['X17'] * np.pi /360)
    df['X25'] = np.cos(df['X25'] * np.pi /360)
    df['X35'] = np.cos(df['X35'] * np.pi /360)
    print('풍향을 cos데이터로 변경 완료')
    
#wind_direction을 cos형 데이터로 변경
make_wind_direction(train)
make_wind_direction(test)


# In[91]:


#4 단순 평균계산식

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


# In[92]:


#가중 평균 내는 함수
def weighted_average(df, col_name, col_num=[]):
    mat = pd.concat([train.loc[:4319,col_num], train.loc[:4319,y_sensor]], axis=1)
    cor_matrix = mat.corr().iloc[5:,:5]
    cor = pd.DataFrame(np.abs(cor_matrix).sum())
    cor_sum = pd.DataFrame(cor.sum(axis=0))
    weight = pd.Series()
    weighted_mean = []
    for i in col_num:
        weight[i] = (cor.loc[i]/cor_sum).values
    for j in np.arange(len(train["X12"])):
        weighted_mean.append(np.average(train.loc[j, col_num], 
                                        weights = weight.values, axis=0)[0][0])
    weighted_mean = pd.Series(weighted_mean)
    df[col_name] = weighted_mean
    print(df)
    print("가중평균 변환 완료")


# In[93]:


#가중 평균 내는 함수
def weighted_average_1(df, col_name, col_num=[]):
    mat = pd.concat([train.loc[4320:,col_num], train.loc[4320:,y_target]], axis=1)
    cor_matrix = mat.corr().iloc[5:,:5]
    cor = pd.DataFrame(np.abs(cor_matrix).sum())
    cor_sum = pd.DataFrame(cor.sum(axis=0))
    weight = pd.Series()
    weighted_mean = []
    for i in col_num:
        weight[i] = (cor.loc[i]/cor_sum).values
    for j in np.arange(len(train["X12"])):
        weighted_mean.append(np.average(train.loc[j, col_num], 
                                        weights = weight.values, axis=0)[0][0])
    weighted_mean = pd.Series(weighted_mean)
    df[col_name] = weighted_mean
    print(df)
    print("가중평균 변환 완료")


# In[94]:


def weighted_average_1_test(df, col_name, col_num=[]):
    mat = pd.concat([train.loc[4320:,col_num], train.loc[4320:,y_target]], axis=1)
    cor_matrix = mat.corr().iloc[5:,:5]
    cor = pd.DataFrame(np.abs(cor_matrix).sum())
    cor_sum = pd.DataFrame(cor.sum(axis=0))
    weight = pd.Series()
    weighted_mean = []
    for i in col_num:
        weight[i] = (cor.loc[i]/cor_sum).values
    for j in np.arange(len(test["X12"])):
        weighted_mean.append(np.average(test.loc[j, col_num], 
                                        weights = weight.values, axis=0)[0][0])
    weighted_mean = pd.Series(weighted_mean)
    df[col_name] = weighted_mean


# In[95]:


def weighted_average_test(df, col_name, col_num=[]):
    mat = pd.concat([train.loc[:4319,col_num], train.loc[:4319,y_sensor]], axis=1)
    cor_matrix = mat.corr().iloc[5:,:5]
    cor = pd.DataFrame(np.abs(cor_matrix).sum())
    cor_sum = pd.DataFrame(cor.sum(axis=0))
    weight = pd.Series()
    weighted_mean = []
    for i in col_num:
        weight[i] = (cor.loc[i]/cor_sum).values
    for j in np.arange(len(test["X12"])):
        weighted_mean.append(np.average(test.loc[j, col_num], 
                                        weights = weight.values, axis=0)[0][0])
    weighted_mean = pd.Series(weighted_mean)
    df[col_name] = weighted_mean


# In[96]:


def transform(df, col):
    for i in col:
        df.loc[:,i] = df[i] - df[i].shift(1)
        df.loc[0,i] = 0
    print("누적 변화 완료")
    return df[col]

def transform2(df, col):
    for i in col:
        for j in np.arange(0,len(df[i])):
            if df.loc[j,i] < 0:
                df.loc[j,i] = 0
    print("24-00시 수정완료")
    return df[col]


# In[98]:


transform(train, x_daily_rainfall)
transform2(train, x_daily_rainfall)
transform(train, x_daily_sun)
transform2(train, x_daily_sun)


# In[99]:


#학습용 데이터프레임 생성
train_df = pd.DataFrame()

#id값 추가
train_df['id'] = train['id']

#id값을 시간형 데이터로 변경
make_timedata(train_df)

#X 대표값 추가
weighted_average_1(train_df, 'x_temperature', x_temperature)
weighted_average_1(train_df, 'x_local_pressure', x_local_pressure)
make_train_x(train_df, 'x_wind_speed', x_wind_speed)
make_train_x(train_df, 'x_daily_rainfall', x_daily_rainfall)
weighted_average_1(train_df, 'x_sealevel_pressure', x_sealevel_pressure)
make_train_x(train_df, 'x_daily_sun', x_daily_sun)
make_train_x(train_df, 'x_humidity', x_humidity)
make_train_x(train_df, 'x_wind_direction', x_wind_direction)

#x변수의 개수. 모든 값 이용시 38, 평균값 이용시 9 -> 값 변경 시, train_df부터 model까지 일괄 변경
x_num = 9 

#Y값 추가 (30일의 Y값)
make_train_y(train_df, 'Y', y_sensor)

#Y0~17로 추정한 Y값(0:4320) 밑에 Y18(4320:)을 추가
train_df.iloc[4320: , x_num] = train.iloc[4320: , 56]


# In[100]:


#6

#학습용 x데이터와 y데이터로 분류
x_train = train_df.drop('Y',axis=1).iloc[:4320,:]
y_train = train_df[['Y']].iloc[:4320,:]

#데이터프레임에서 2차원 array로 변경
x_train = x_train.values
y_train = y_train.values


#검증용 x데이터와 y데이터로 분류
x_valid = train_df.drop('Y',axis=1).iloc[4320:,:]
y_valid = train_df[['Y']].iloc[4320:,:]

#데이터프레임에서 array로 변경
x_valid = x_valid.values
y_valid= y_valid.values

print(x_train.shape)
#2차원 array에서 딥러닝을 위해 3차원 array로 변경
x_train_t = x_train.reshape(x_train.shape[0] , x_num, 1) 
x_valid_t = x_valid.reshape(x_valid.shape[0] , x_num, 1)
print(y_train)


# In[101]:


import tensorflow as tf
import random as rn

seed_num = 42
np.random.seed(seed_num)
rn.seed(seed_num)
tf.set_random_seed(seed_num)


from keras import backend as K

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(),config=session_conf)
K.set_session(sess)


# In[102]:


#7

#딥러닝 모델 생성
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

#LSTM(20), loss = mse로 설정해서 mse를 줄이는 방향으로 모델 생성
model = Sequential()  
model.add(LSTM(20, input_shape=(x_num, 1))) # input_shape(timestep=몇개를 묶어서 볼것인가, feature = 변수가 몇개인가) 
model.add(Dense(1)) # output = 1
model.compile(loss='mean_squared_error', optimizer= 'adam')
model.summary()

#patience=val_loss가 5번의 Epoch동안 줄어들지 않으면, 조기종료. (과적합 방지를 위함)
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)


# In[103]:


history = model.fit(x_train_t, y_train, epochs=150, callbacks=[early_stop],
                    validation_data=(x_valid_t,y_valid))


#학습셋의 오차
y_loss = history.history['loss']

#테스트셋의 오차
y_vloss = history.history['val_loss']

# 결과 그래프 생성
x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c="red", label="Validset_loss")
plt.plot(x_len, y_loss, marker='.', c="blue", label="Trainset_loss")
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


# In[104]:


transform(test, x_daily_rainfall)
transform2(test, x_daily_rainfall)
transform(test, x_daily_sun)
transform2(test, x_daily_sun)


# In[105]:


#9

#테스트용 데이터프레임 생성
test_df = pd.DataFrame()

#id값 추가
test_df['id'] = test['id']

#id값을 시간형 데이터로 변경
make_timedata(test_df)

#X 대표값 추가
weighted_average_1_test(test_df, 'x_temperature', x_temperature)
weighted_average_1_test(test_df, 'x_local_pressure', x_local_pressure)
make_test_x(test_df, 'x_wind_speed', x_wind_speed)
make_test_x(test_df, 'x_daily_rainfall', x_daily_rainfall)
weighted_average_1_test(test_df, 'x_sealevel_pressure', x_sealevel_pressure)
make_test_x(test_df, 'x_daily_sun', x_daily_sun)
make_test_x(test_df, 'x_humidity', x_humidity)
make_test_x(test_df, 'x_wind_direction', x_wind_direction)
test_df


# In[106]:


x_test = test_df
x_test= x_test.values
x_test_t = x_test.reshape(x_test.shape[0], 9, 1)

y_pred = model.predict(x_test_t)


# In[107]:


result = pd.DataFrame()
result["id"] = test["id"]
result["Y18"] = y_pred
result

result.to_csv("C:/Users/study gil/Documents/0323_7.csv", index=False)
# x 대표값, y 034만 제외 = 4.760

