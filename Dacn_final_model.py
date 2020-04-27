#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import tensorflow as tf

np.random.seed(7)
random.seed(7)
tf.random.set_seed(7)


# In[18]:


#1
#csv파일 불러오기
train = pd.read_csv('C:/Users/study gil/Documents/train.csv')
test = pd.read_csv('C:/Users/study gil/Documents//test.csv')
    
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


# In[19]:


# 누적강수량/누적일사량을 10분 단위로 변환
def accumulate_to_10minute(df, col):
    for i in col:
        #누적데이터를 10분간의 데이터로 분류
        tmp = df[i].iloc[0]
        df[i] = df[i] - df[i].shift(1)
        df[i].iloc[0] = tmp
        
        #24시를 넘어서 데이터가 음수가 된다면, 0으로 변경
        for j in np.arange(0,len(df[i])):
            if df[i].iloc[j] < 0:
                df[i].iloc[j] = 0
    print("누적 데이터 변경 완료")


def remove_outlier(df):
    for j in list(df.columns):
        for i in np.arange(0,len(df["X00"])):
            if df.loc[i,j] > 3:
                df.loc[i,j] = df.shift(1).loc[i,j]                

#고장난 센서 X14, X16, X19 삭제
train = train.drop(['X14', 'X16', 'X19'], axis =1)
test = test.drop(['X14', 'X16', 'X19'], axis =1)
print('X14, X16, X19 삭제 완료')

#누적 일사량 변경
accumulate_to_10minute(train, x_daily_sun)
accumulate_to_10minute(test, x_daily_sun)


# In[20]:


#학습용 데이터프레임 생성
train_df = pd.DataFrame(pd.concat([train[x_temperature], 
                                  train[x_daily_sun]], axis=1))

test_df = pd.DataFrame(pd.concat([test[x_temperature], 
                                 test[x_daily_sun]], axis=1))


# standardization을 위해 평균과 표준편차 구하기
MEAN = train_df.mean()
STD = train_df.std()

# 표준편차가 0일 경우 대비하여 1e-07 추가 
train_df = (train_df - MEAN) / (STD + 1e-07)

test_df = (test_df - MEAN) / (STD + 1e-07)


remove_outlier(train_df)


# In[21]:


remove_outlier(test_df)


# In[22]:


#4. RNN 모델에 입력 할 수 있는 시계열 형태로 데이터 변환 
def convert_to_timeseries(df, interval):
    sequence_list = []
    target_list = []
    
    for i in tqdm(range(df.shape[0] - interval)):
        sequence_list.append(np.array(df.iloc[i:i+interval,:-1]))
        target_list.append(df.iloc[i+interval,-1])
    
    sequence = np.array(sequence_list)
    target = np.array(target_list)
    
    return sequence, target


# In[23]:


x_num = 7


#학습에 사용할 Y값 지정
y_columns = ['Y09','Y16','Y15']


#지정한 Y들만큼 데이터의 길이를 늘림
sequence = np.empty((0, 12, x_num))
target = np.empty((0,))

for column in y_columns :
    
    concat = pd.concat([train_df, train[column]], axis = 1)

    _sequence, _target = convert_to_timeseries(concat.head(144*30), interval = 12)

    sequence = np.vstack((sequence, _sequence))
    target = np.hstack((target, _target))


# In[24]:


"""
#6. test_df 생성 (x변수를 평균 사용)
"""

train_df['dummy'] = 0
test_df['dummy'] = 0

X_test, _ = convert_to_timeseries(pd.concat([train_df, test_df], axis = 0), interval=12)
X_test = X_test[-11520:, :, :]

# 만들어 두었던 dummy feature 제거
train_df.drop('dummy', axis = 1, inplace = True)
test_df.drop('dummy', axis = 1, inplace = True)


# In[25]:


sequence.shape


# In[26]:


"""

#7. 학습 모델 생성

"""

simple_lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(128, input_shape=sequence.shape[-2:]),
    tf.keras.layers.Dense(256,activation ='relu'),
    tf.keras.layers.Dense(128,activation ='relu'),
    tf.keras.layers.Dense(1)
])

simple_lstm_model.compile(optimizer='adam', loss='mse',metrics = ['mse'])
simple_lstm_model.summary()


# In[27]:


# loss가 4미만으로 떨어지면 학습 종료 시키는 기능
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = None):
        if(logs.get('loss') < 4):
            print('\n Loss is under 4, cancelling training')
            self.model.stop_training = True
            

callbacks = myCallback()


# In[28]:


"""
#8. 모델 학습
"""
simple_lstm_model.fit(
    sequence, target,
    epochs=60,
    batch_size=64,
    verbose=2,
    shuffle=False,
    callbacks = [callbacks]
)


# In[29]:


"""

#9. 마지막 3일 데이터('Y18')를 사용하여 fine-tuning 

"""

simple_lstm_model.layers[0].trainable = False
finetune_X, finetune_y = convert_to_timeseries(pd.concat([train_df.tail(432), 
                                                          train['Y18'].tail(432)], axis = 1), 
                                               interval=12)


finetune_history = simple_lstm_model.fit(
            finetune_X, finetune_y,
            epochs=33,
            batch_size=64,
            shuffle=False,
            verbose = 2)
history_out= finetune_history.history
history_out.keys()


# In[30]:


import matplotlib.pyplot as plt
loss=history_out['loss']
plt.plot(loss,'bo',label='training loss')
plt.title('Training losss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


# In[31]:


#10. 결과 예측하기 
finetune_pred = simple_lstm_model.predict(X_test)


# 제출 파일 만들기
result = pd.DataFrame()
result['id'] = range(4752, 16272)
result['Y18'] = finetune_pred
result.shape


# In[32]:


result.to_csv('C:/Users/study gil/Documents/0413_2.csv', index = False)

