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


# In[2]:


"""
Read.me

전반적 흐름
1. train, test.csv읽어오기
2. X14, 16, 19를 삭제, 시간 데이터를 만드는 함수 생성
3. 풍향데이터를 cos데이터로 바꿔주기 (0도:1, 360도:-1)
4. train_df를 만들기 위해, x값들을 처리할 함수 생성 (묶여있는 데이터들의 평균값을 x, y로 설정)
5. train_df를 생성(x값을 1차가공, y값도 1차가공)
6. train_df를 train용, validation용으로 데이터 분류
7. LSTM모델 생성
8. LSTM모델 학습 & 그래프 생성
9. test_df를 생성
10. test데이터 분류, LSTM모델로 test데이터를 predict
11. 결과값을 submission.csv로 저장
"""


# In[2]:


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


# In[3]:


#2

#고장난 센서 X14, X16, X19 삭제
train = train.drop(['X14', 'X16', 'X19'], axis =1)
test = test.drop(['X14', 'X16', 'X19'], axis =1)
print('X14, X16, X19 삭제 완료')

train.shape


# In[4]:


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


# In[5]:


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


# In[6]:


train.shape


# In[6]:


a = pd.DataFrame()

# train.columns[1:38]

a


# In[7]:


#5

#학습용 데이터프레임 생성
train_df = pd.DataFrame()

#id값 추가
train_df['id'] = train['id']

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
train_df[train.columns[1 : x_num]] = train.iloc[:, 1 : x_num]

#Y값 추가 (30일의 Y값)
make_train_y(train_df, 'Y', y_sensor)

#Y0~17로 추정한 Y값(0:4320) 밑에 Y18(4320:)을 추가
train_df.iloc[4320: , x_num] = train.iloc[4320: , 56]
train_df


# In[8]:


#6

"""
머신러닝을 위한 데이터 분류 과정

0~4320은 train용으로
4320~4752는 validation용으로 분할
"""

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
x_train_t


# In[9]:


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


# In[10]:


#7

"""
케라스 모델 생성
초기값 : LSTM(20), Dense(1)
"""

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


# In[12]:


#8

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


# In[13]:


#9

#테스트용 데이터프레임 생성
test_df = pd.DataFrame()

#id값 추가
test_df['id'] = test['id']

#id값을 시간형 데이터로 변경
make_timedata(test_df)
"""
#X 대표값을 평균값으로 추가
make_test_x(test_df, 'x_temperature', x_temperature)
make_test_x(test_df, 'x_local_pressure', x_local_pressure)
make_test_x(test_df, 'x_wind_speed', x_wind_speed)
make_test_x(test_df, 'x_daily_rainfall', x_daily_rainfall)
make_test_x(test_df, 'x_sealevel_pressure', x_sealevel_pressure)
make_test_x(test_df, 'x_daily_sun', x_daily_sun)
make_test_x(test_df, 'x_humidity', x_humidity)
make_test_x(test_df, 'x_wind_direction', x_wind_direction)
"""

#X값 전체를 추가
test_df[test.columns[1 : x_num]] = test.iloc[:, 1 : x_num]



test_df


# In[14]:


#10

x_test = test_df
x_test= x_test.values
x_test_t = x_test.reshape(x_test.shape[0], x_num, 1)

y_pred = model.predict(x_test_t)


# In[15]:


#11

result = pd.DataFrame()
result["id"] = test["id"]
result["Y18"] = y_pred
result

result.to_csv("C:/Users/study gil/Documents/submission.csv", index=False)
#원래 데이터 (y 전체평균, x 전체 사용 = 9.434)


# In[16]:


"""
#X데이터 정규화
x_name = train_df.iloc[:,1:].columns


#스탠다드스케일러를 사용하여, X데이터를 N(0,1)정규화.
from sklearn.preprocessing import StandardScaler  

Scaler = StandardScaler()
train_X_normed = Scaler.fit_transform( train_df[x_name] ) 
test_X_normed = Scaler.transform(test_df[x_name])   

#train데이터와 test데이터에 저장.
train_df = pd.DataFrame(columns=x_name, data = train_X_normed)
test_df = pd.DataFrame(columns=x_name, data = test_X_normed)
"""


# In[17]:


#상관계수 낮은거 삭제한 y센서
y_sensor_nolow = ['Y01','Y02', 'Y05', 'Y06', 'Y08', 
            'Y09', 'Y10', 'Y11', 'Y12', 'Y14', 'Y15', 'Y16', 'Y17']

#학습용 데이터프레임 생성
train_df_sample = pd.DataFrame()

#id값 추가
train_df_sample['id'] = train['id']

#id값을 시간형 데이터로 변경
make_timedata(train_df_sample)

#x변수의 개수. 모든 값 이용시 38, 평균값 이용시 9 -> 값 변경 시, train_df부터 model까지 일괄 변경
x_num = 38 

#X값 전체를 추가
train_df_sample[train.columns[1 : x_num]] = train.iloc[:, 1 : x_num]

#Y값 추가 (30일의 Y값)
make_train_y(train_df_sample, 'Y', y_sensor_nolow)

#Y0~17로 추정한 Y값(0:4320) 밑에 Y18(4320:)을 추가
train_df_sample.iloc[4320: , x_num] = train.iloc[4320: , 56]
train_df_sample


# In[18]:


#학습용 x데이터와 y데이터로 분류
x_train = train_df_sample.drop('Y',axis=1).iloc[:4320,:]
y_train = train_df_sample[['Y']].iloc[:4320,:]

#데이터프레임에서 2차원 array로 변경
x_train = x_train.values
y_train = y_train.values


#검증용 x데이터와 y데이터로 분류
x_valid = train_df_sample.drop('Y',axis=1).iloc[4320:,:]
y_valid = train_df_sample[['Y']].iloc[4320:,:]

#데이터프레임에서 array로 변경
x_valid = x_valid.values
y_valid= y_valid.values

print(x_train.shape)
#2차원 array에서 딥러닝을 위해 3차원 array로 변경
x_train_t = x_train.reshape(x_train.shape[0] , x_num, 1) 
x_valid_t = x_valid.reshape(x_valid.shape[0] , x_num, 1)


# In[19]:


model = Sequential()  
model.add(LSTM(20, input_shape=(x_num, 1))) # input_shape(timestep=몇개를 묶어서 볼것인가, feature = 변수가 몇개인가) 
model.add(Dense(1)) # output = 1
model.compile(loss='mean_squared_error', optimizer= 'adam')
model.summary()

#patience=val_loss가 5번의 Epoch동안 줄어들지 않으면, 조기종료. (과적합 방지를 위함)
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)


# In[20]:


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


# In[24]:


y_pred = model.predict(x_test_t)


# In[25]:


result = pd.DataFrame()
result["id"] = test["id"]
result["Y18"] = y_pred
result

result.to_csv("C:/Users/study gil/Documents/submission1.csv", index=False)
#X 전체 사용, Y 상관계수 낮은거 제외 = 7.124


# In[28]:


#상관계수 높은거 y센서
y_sensor_high = ['Y01','Y02','Y09','Y11', 'Y14', 'Y15', 'Y16', 'Y17']

#학습용 데이터프레임 생성
train_df_sample = pd.DataFrame()

#id값 추가
train_df_sample['id'] = train['id']

#id값을 시간형 데이터로 변경
make_timedata(train_df_sample)

#x변수의 개수. 모든 값 이용시 38, 평균값 이용시 9 -> 값 변경 시, train_df부터 model까지 일괄 변경
x_num = 38 

#X값 전체를 추가
train_df_sample[train.columns[1 : x_num]] = train.iloc[:, 1 : x_num]

#Y값 추가 (30일의 Y값)
make_train_y(train_df_sample, 'Y',y_sensor_high)

#Y0~17로 추정한 Y값(0:4320) 밑에 Y18(4320:)을 추가
train_df_sample.iloc[4320: , x_num] = train.iloc[4320: , 56]
train_df_sample


# In[29]:


#학습용 x데이터와 y데이터로 분류
x_train = train_df_sample.drop('Y',axis=1).iloc[:4320,:]
y_train = train_df_sample[['Y']].iloc[:4320,:]

#데이터프레임에서 2차원 array로 변경
x_train = x_train.values
y_train = y_train.values


#검증용 x데이터와 y데이터로 분류
x_valid = train_df_sample.drop('Y',axis=1).iloc[4320:,:]
y_valid = train_df_sample[['Y']].iloc[4320:,:]

#데이터프레임에서 array로 변경
x_valid = x_valid.values
y_valid= y_valid.values

print(x_train.shape)
#2차원 array에서 딥러닝을 위해 3차원 array로 변경
x_train_t = x_train.reshape(x_train.shape[0] , x_num, 1) 
x_valid_t = x_valid.reshape(x_valid.shape[0] , x_num, 1)
print(y_train)


# In[30]:


model = Sequential()  
model.add(LSTM(20, input_shape=(x_num, 1))) # input_shape(timestep=몇개를 묶어서 볼것인가, feature = 변수가 몇개인가) 
model.add(Dense(1)) # output = 1
model.compile(loss='mean_squared_error', optimizer= 'adam')
model.summary()

#patience=val_loss가 5번의 Epoch동안 줄어들지 않으면, 조기종료. (과적합 방지를 위함)
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)


# In[29]:


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


# In[44]:


y_pred = model.predict(x_test_t)


# In[45]:


result = pd.DataFrame()
result["id"] = test["id"]
result["Y18"] = y_pred
result

result.to_csv("C:/Users/study gil/Documents/submission2.csv", index=False)


# In[11]:


# Y 3,4 만 삭제한거, X 대표값 사용
y_sensor_no34 = ['Y00', 'Y01','Y02','Y05', 'Y06', 'Y07', 'Y08', 
            'Y09', 'Y10', 'Y11', 'Y12', 'Y13', 'Y14', 'Y15', 'Y16', 'Y17']
#학습용 데이터프레임 생성
train_df = pd.DataFrame()

#id값 추가
train_df['id'] = train['id']

#id값을 시간형 데이터로 변경
make_timedata(train_df)

#X 대표값 추가
make_train_x(train_df, 'x_temperature', x_temperature)
make_train_x(train_df, 'x_local_pressure', x_local_pressure)
make_train_x(train_df, 'x_wind_speed', x_wind_speed)
make_train_x(train_df, 'x_daily_rainfall', x_daily_rainfall)
make_train_x(train_df, 'x_sealevel_pressure', x_sealevel_pressure)
make_train_x(train_df, 'x_daily_sun', x_daily_sun)
make_train_x(train_df, 'x_humidity', x_humidity)
make_train_x(train_df, 'x_wind_direction', x_wind_direction)

#x변수의 개수. 모든 값 이용시 38, 평균값 이용시 9 -> 값 변경 시, train_df부터 model까지 일괄 변경
x_num = 9 

#Y값 추가 (30일의 Y값)
make_train_y(train_df, 'Y', y_sensor_no34)

#Y0~17로 추정한 Y값(0:4320) 밑에 Y18(4320:)을 추가
train_df.iloc[4320: , x_num] = train.iloc[4320: , 56]


# In[15]:


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


# In[16]:


model = Sequential()  
model.add(LSTM(20, input_shape=(x_num, 1))) # input_shape(timestep=몇개를 묶어서 볼것인가, feature = 변수가 몇개인가) 
model.add(Dense(1)) # output = 1
model.compile(loss='mean_squared_error', optimizer= 'adam')
model.summary()

#patience=val_loss가 5번의 Epoch동안 줄어들지 않으면, 조기종료. (과적합 방지를 위함)
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)


# In[17]:


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


# In[34]:


#9

#테스트용 데이터프레임 생성
test_df = pd.DataFrame()

#id값 추가
test_df['id'] = test['id']

#id값을 시간형 데이터로 변경
make_timedata(test_df)

#X 대표값 추가
make_test_x(test_df, 'x_temperature', x_temperature)
make_test_x(test_df, 'x_local_pressure', x_local_pressure)
make_test_x(test_df, 'x_wind_speed', x_wind_speed)
make_test_x(test_df, 'x_daily_rainfall', x_daily_rainfall)
make_test_x(test_df, 'x_sealevel_pressure', x_sealevel_pressure)
make_test_x(test_df, 'x_daily_sun', x_daily_sun)
make_test_x(test_df, 'x_humidity', x_humidity)
make_test_x(test_df, 'x_wind_direction', x_wind_direction)

test_df


# In[35]:


x_test = test_df
x_test= x_test.values
x_test_t = x_test.reshape(x_test.shape[0], 9, 1)

y_pred = model.predict(x_test_t)


# In[38]:


result = pd.DataFrame()
result["id"] = test["id"]
result["Y18"] = y_pred
result

result.to_csv("C:/Users/study gil/Documents/submission_sam.csv", index=False)
# x 대표값, y 34만 제외 = 4.020


# In[43]:


y_sensor_no034 = ['Y01','Y02','Y05', 'Y06', 'Y07', 'Y08', 
            'Y09', 'Y10', 'Y11', 'Y12', 'Y13', 'Y14', 'Y15', 'Y16', 'Y17']
#학습용 데이터프레임 생성
train_df = pd.DataFrame()

#id값 추가
train_df['id'] = train['id']

#id값을 시간형 데이터로 변경
make_timedata(train_df)

#X 대표값 추가
make_train_x(train_df, 'x_temperature', x_temperature)
make_train_x(train_df, 'x_local_pressure', x_local_pressure)
make_train_x(train_df, 'x_wind_speed', x_wind_speed)
make_train_x(train_df, 'x_daily_rainfall', x_daily_rainfall)
make_train_x(train_df, 'x_sealevel_pressure', x_sealevel_pressure)
make_train_x(train_df, 'x_daily_sun', x_daily_sun)
make_train_x(train_df, 'x_humidity', x_humidity)
make_train_x(train_df, 'x_wind_direction', x_wind_direction)

#x변수의 개수. 모든 값 이용시 38, 평균값 이용시 9 -> 값 변경 시, train_df부터 model까지 일괄 변경
x_num = 9 

#Y값 추가 (30일의 Y값)
make_train_y(train_df, 'Y', y_sensor_no034)

#Y0~17로 추정한 Y값(0:4320) 밑에 Y18(4320:)을 추가
train_df.iloc[4320: , x_num] = train.iloc[4320: , 56]


# In[44]:


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


# In[45]:


model = Sequential()  
model.add(LSTM(20, input_shape=(x_num, 1))) # input_shape(timestep=몇개를 묶어서 볼것인가, feature = 변수가 몇개인가) 
model.add(Dense(1)) # output = 1
model.compile(loss='mean_squared_error', optimizer= 'adam')
model.summary()

#patience=val_loss가 5번의 Epoch동안 줄어들지 않으면, 조기종료. (과적합 방지를 위함)
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)


# In[46]:


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


# In[48]:


#9

#테스트용 데이터프레임 생성
test_df = pd.DataFrame()

#id값 추가
test_df['id'] = test['id']

#id값을 시간형 데이터로 변경
make_timedata(test_df)

#X 대표값 추가
make_test_x(test_df, 'x_temperature', x_temperature)
make_test_x(test_df, 'x_local_pressure', x_local_pressure)
make_test_x(test_df, 'x_wind_speed', x_wind_speed)
make_test_x(test_df, 'x_daily_rainfall', x_daily_rainfall)
make_test_x(test_df, 'x_sealevel_pressure', x_sealevel_pressure)
make_test_x(test_df, 'x_daily_sun', x_daily_sun)
make_test_x(test_df, 'x_humidity', x_humidity)
make_test_x(test_df, 'x_wind_direction', x_wind_direction)


# In[49]:


x_test = test_df
x_test= x_test.values
x_test_t = x_test.reshape(x_test.shape[0], 9, 1)

y_pred = model.predict(x_test_t)


# In[50]:


result = pd.DataFrame()
result["id"] = test["id"]
result["Y18"] = y_pred
result

result.to_csv("C:/Users/study gil/Documents/submission_5.csv", index=False)
# x 대표값, y 034만 제외 = 4.760


# In[64]:


y_sensor_high = ['Y05', 'Y09', 'Y10', 'Y11', 'Y12', 'Y13', 'Y15', 'Y16', 'Y17']
#학습용 데이터프레임 생성
train_df = pd.DataFrame()

#id값 추가
train_df['id'] = train['id']

#id값을 시간형 데이터로 변경
make_timedata(train_df)

#X 대표값 추가
make_train_x(train_df, 'x_temperature', x_temperature)
make_train_x(train_df, 'x_local_pressure', x_local_pressure)
make_train_x(train_df, 'x_wind_speed', x_wind_speed)
make_train_x(train_df, 'x_daily_rainfall', x_daily_rainfall)
make_train_x(train_df, 'x_sealevel_pressure', x_sealevel_pressure)
make_train_x(train_df, 'x_daily_sun', x_daily_sun)
make_train_x(train_df, 'x_humidity', x_humidity)
make_train_x(train_df, 'x_wind_direction', x_wind_direction)

#x변수의 개수. 모든 값 이용시 38, 평균값 이용시 9 -> 값 변경 시, train_df부터 model까지 일괄 변경
x_num = 9 

#Y값 추가 (30일의 Y값)
make_train_y(train_df, 'Y', y_sensor_high)

#Y0~17로 추정한 Y값(0:4320) 밑에 Y18(4320:)을 추가
train_df.iloc[4320: , x_num] = train.iloc[4320: , 56]


# In[65]:


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


# In[70]:


model = Sequential()  
model.add(LSTM(20, input_shape=(x_num, 1))) # input_shape(timestep=몇개를 묶어서 볼것인가, feature = 변수가 몇개인가) 
model.add(Dense(1)) # output = 1
model.compile(loss='mean_squared_error', optimizer= 'adam')
model.summary()

#patience=val_loss가 5번의 Epoch동안 줄어들지 않으면, 조기종료. (과적합 방지를 위함)
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)


# In[71]:


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


# In[72]:


#9

#테스트용 데이터프레임 생성
test_df = pd.DataFrame()

#id값 추가
test_df['id'] = test['id']

#id값을 시간형 데이터로 변경
make_timedata(test_df)

#X 대표값 추가
make_test_x(test_df, 'x_temperature', x_temperature)
make_test_x(test_df, 'x_local_pressure', x_local_pressure)
make_test_x(test_df, 'x_wind_speed', x_wind_speed)
make_test_x(test_df, 'x_daily_rainfall', x_daily_rainfall)
make_test_x(test_df, 'x_sealevel_pressure', x_sealevel_pressure)
make_test_x(test_df, 'x_daily_sun', x_daily_sun)
make_test_x(test_df, 'x_humidity', x_humidity)
make_test_x(test_df, 'x_wind_direction', x_wind_direction)


# In[73]:


x_test = test_df
x_test= x_test.values
x_test_t = x_test.reshape(x_test.shape[0], 9, 1)

y_pred = model.predict(x_test_t)


# In[75]:


result = pd.DataFrame()
result["id"] = test["id"]
result["Y18"] = y_pred
result
result.to_csv("C:/Users/study gil/Documents/submission_7.csv", index=False)
# x 대표값, y 034만 제외 = 4.760


# In[25]:


# y34 제외 현지기압 삭제
#학습용 데이터프레임 생성
y_sensor_no34 = ['Y00', 'Y01','Y02','Y05', 'Y06', 'Y07', 'Y08', 
            'Y09', 'Y10', 'Y11', 'Y12', 'Y13', 'Y14', 'Y15', 'Y16', 'Y17']
train_df = pd.DataFrame()

#id값 추가
train_df['id'] = train['id']

#id값을 시간형 데이터로 변경
make_timedata(train_df)

#X 대표값 추가
make_train_x(train_df, 'x_temperature', x_temperature)
make_train_x(train_df, 'x_wind_speed', x_wind_speed)
make_train_x(train_df, 'x_daily_rainfall', x_daily_rainfall)
make_train_x(train_df, 'x_sealevel_pressure', x_sealevel_pressure)
make_train_x(train_df, 'x_daily_sun', x_daily_sun)
make_train_x(train_df, 'x_humidity', x_humidity)
make_train_x(train_df, 'x_wind_direction', x_wind_direction)

#x변수의 개수. 모든 값 이용시 38, 평균값 이용시 9 -> 값 변경 시, train_df부터 model까지 일괄 변경
x_num = 8 

#Y값 추가 (30일의 Y값)
make_train_y(train_df, 'Y', y_sensor_no34)

#Y0~17로 추정한 Y값(0:4320) 밑에 Y18(4320:)을 추가
train_df.iloc[4320: , x_num] = train.iloc[4320: , 56]


# In[ ]:





# In[26]:


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


# In[27]:


model = Sequential()  
model.add(LSTM(20, input_shape=(x_num, 1))) # input_shape(timestep=몇개를 묶어서 볼것인가, feature = 변수가 몇개인가) 
model.add(Dense(1)) # output = 1
model.compile(loss='mean_squared_error', optimizer= 'adam')
model.summary()

#patience=val_loss가 5번의 Epoch동안 줄어들지 않으면, 조기종료. (과적합 방지를 위함)
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)


# In[28]:


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


# In[29]:


#테스트용 데이터프레임 생성
test_df = pd.DataFrame()

#id값 추가
test_df['id'] = test['id']

#id값을 시간형 데이터로 변경
make_timedata(test_df)

#X 대표값 추가
make_test_x(test_df, 'x_temperature', x_temperature)
make_test_x(test_df, 'x_wind_speed', x_wind_speed)
make_test_x(test_df, 'x_daily_rainfall', x_daily_rainfall)
make_test_x(test_df, 'x_sealevel_pressure', x_sealevel_pressure)
make_test_x(test_df, 'x_daily_sun', x_daily_sun)
make_test_x(test_df, 'x_humidity', x_humidity)
make_test_x(test_df, 'x_wind_direction', x_wind_direction)


# In[30]:


x_test = test_df
x_test= x_test.values
x_test_t = x_test.reshape(x_test.shape[0], 8, 1)

y_pred = model.predict(x_test_t)


# In[31]:


result = pd.DataFrame()
result["id"] = test["id"]
result["Y18"] = y_pred
result

result.to_csv("C:/Users/study gil/Documents/submission_9.csv", index=False)
# x 대표값, y 034만 제외 = 4.760


# In[19]:


pressure = ['X01', 'X06', 'X22', 'X27', 'X29','X05', 'X08', 'X09', 'X23', 'X33']
# Y 3,4 만 삭제한거, X 대표값 사용
y_sensor_no34 = ['Y00', 'Y01','Y02','Y05', 'Y06', 'Y07', 'Y08', 
            'Y09', 'Y10', 'Y11', 'Y12', 'Y13', 'Y14', 'Y15', 'Y16', 'Y17']
#학습용 데이터프레임 생성
train_df = pd.DataFrame()

#id값 추가
train_df['id'] = train['id']

#id값을 시간형 데이터로 변경
make_timedata(train_df)

#X 대표값 추가
make_train_x(train_df, 'x_temperature', x_temperature)
make_train_x(train_df, 'pressure', pressure)
make_train_x(train_df, 'x_wind_speed', x_wind_speed)
make_train_x(train_df, 'x_daily_rainfall', x_daily_rainfall)
make_train_x(train_df, 'x_daily_sun', x_daily_sun)
make_train_x(train_df, 'x_humidity', x_humidity)
make_train_x(train_df, 'x_wind_direction', x_wind_direction)

#x변수의 개수. 모든 값 이용시 38, 평균값 이용시 9 -> 값 변경 시, train_df부터 model까지 일괄 변경
x_num = 8 

#Y값 추가 (30일의 Y값)
make_train_y(train_df, 'Y', y_sensor_no34)

#Y0~17로 추정한 Y값(0:4320) 밑에 Y18(4320:)을 추가
train_df.iloc[4320: , x_num] = train.iloc[4320: , 56]


# In[8]:


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


# In[11]:


model = Sequential()  
model.add(LSTM(20, input_shape=(x_num, 1))) # input_shape(timestep=몇개를 묶어서 볼것인가, feature = 변수가 몇개인가) 
model.add(Dense(1)) # output = 1
model.compile(loss='mean_squared_error', optimizer= 'adam')
model.summary()

#patience=val_loss가 5번의 Epoch동안 줄어들지 않으면, 조기종료. (과적합 방지를 위함)
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)


# In[12]:


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


# In[13]:


#9

#테스트용 데이터프레임 생성
test_df = pd.DataFrame()

#id값 추가
test_df['id'] = test['id']

#id값을 시간형 데이터로 변경
make_timedata(test_df)

#X 대표값 추가
make_test_x(test_df, 'x_temperature', x_temperature)
make_test_x(test_df, 'pressure', pressure)
make_test_x(test_df, 'x_wind_speed', x_wind_speed)
make_test_x(test_df, 'x_daily_rainfall', x_daily_rainfall)
make_test_x(test_df, 'x_daily_sun', x_daily_sun)
make_test_x(test_df, 'x_humidity', x_humidity)
make_test_x(test_df, 'x_wind_direction', x_wind_direction)


# In[14]:


x_test = test_df
x_test= x_test.values
x_test_t = x_test.reshape(x_test.shape[0], 8, 1)

y_pred = model.predict(x_test_t)


# In[17]:


result = pd.DataFrame()
result["id"] = test["id"]
result["Y18"] = y_pred
result

result.to_csv("C:/Users/study gil/Documents/submission_10.csv", index=False)
# x 대표값, 해면기압, 현지기압 전체평균 y 034만 제외 = 4.760

