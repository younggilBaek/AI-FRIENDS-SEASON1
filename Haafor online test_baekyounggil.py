#!/usr/bin/env python
# coding: utf-8

# In[104]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings(action='ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[268]:


#P1
#1 -1 Data loading and EDA
data_1=pd.read_csv('C:/Users/study gil/Documents/P1 regression.csv',header=None)
data_1.columns = list("ABCD")
data_1.shape


# In[269]:


data_1.describe()
# I can see that C and D show very large deviation.
# Also, in C, there seems to be skewed to the righst which showing the large Q3, Max values


# In[270]:


data_1.boxplot();
# a lot of outliers in C, D


# In[271]:


# There are lots of outliers in D
plt.figure(figsize=(10,10))
plt.plot(data_1['D'],label='Y')
plt.legend();


# In[272]:


# In closer look, we can see that D fluctuates a lot around the mean
# seems not to be adequate to fitting D in terms of linear Regression
# it seems to be stationary
plt.figure(figsize=(10,10))
plt.plot(data_1['D'],label='Y')
plt.axis([0,1000,0,100])
plt.legend();


# In[273]:


# Graphs of independent variables (A,B and C)
# A,B,C both fluctuates aroud the mean value. 
# C shows lots of outliers.
# They seems to be stationary time series.
plt.figure(figsize=(10,10))
plt.subplot(3,1,1)
plt.plot(data_1['A'],label='A')
plt.legend();
plt.subplot(3,1,2)
plt.plot(data_1['B'],label='B', color= 'red')
plt.legend();
plt.subplot(3,1,3)
plt.plot(data_1['C'],label='C', color= 'green')
plt.legend();


# In[274]:


cor= data_1.corr()
plt.figure(figsize = (5, 5))
sns.heatmap(cor,  vmin = 0.2, annot = True, vmax = 0.9);
# A,B are highly correlated (0.98)
# C shows the highest correlation with D 


# In[216]:


"""
1. A, B shows very stable values over time. Whereas C has largely fluctating trends.
2. D has a lot of outliers which seems adequate to be removed.
3. A, B shows very high correlations. So it may cause multicollinearity 
when fitting both of them in Regression.
4. All of variables seems to be time-seires. 
5. So it may better to use time_trend attributes 
rather than just fitting linear Regression
"""


# In[275]:


# Before modelling, check for the stationarity.
from statsmodels.tsa.stattools import adfuller
target = data_1['D']
adfuller(target)
#it is stationary


# In[276]:


x_1 = data_1['A']
adfuller(x_1)
#it is stationary


# In[277]:


x_2 = data_1['B']
adfuller(x_2)
#it is stationary


# In[278]:


x_3 = data_1['C']
adfuller(x_3)
#it is stationary
# There is no need to take difference of variables.


# In[279]:


#1-2 Data pre-procesing
from sklearn.ensemble import ExtraTreesRegressor

x = data_1.iloc[:,:3]
y = data_1.iloc[:,3]

et_model = ExtraTreesRegressor()
et_model.fit(x,y)

print(et_model.feature_importances_)
feature_list = pd.concat([pd.Series(x.columns), pd.Series(et_model.feature_importances_)], axis=1)
feature_list.columns = ['features_name', 'importance']
feature_list.sort_values("importance", ascending =False)
# A,B Shows large importance rather than C 


# In[280]:


# standardization을 위해 평균과 표준편차 구하기
MEAN = data_1.mean()
STD = data_1.std()

# add 1e-07 for the case when std is zero 
data_st = (data_1 - MEAN) / (STD + 1e-07)


# In[223]:


# replace outlier to the value right before that (define outlier which has value over 4)
def remove_outlier(df):
    for j in list(df.columns):
        for i in np.arange(0,len(df["A"])):
            if abs(df.loc[i,j]) > 4:
                df.loc[i,j] = df.shift(1).loc[i,j]
remove_outlier(data_st)
data_st = data_st.fillna(0)


# In[281]:


# for Reproducibility, split train and test dataset
train_df = data_st.iloc[:950, :]
test_df = data_st.iloc[950:, :]
print(train_df.shape); print(test_df.shape)
test_df


# In[282]:


# transforming into Timeseries data to fit for RNN MODEL
def convert_to_timeseries(df, interval):
    sequence_list = []
    target_list = []
    
    for i in tqdm(range(df.shape[0] - interval)):
        sequence_list.append(np.array(df.iloc[i:i+interval,:-1]))
        target_list.append(df.iloc[i+interval,-1])
    
    sequence = np.array(sequence_list)
    target = np.array(target_list)
    
    return sequence, target


# In[283]:


x_num = 3


sequence = np.empty((0, 12, x_num))
target = np.empty((0,))

sequence = np.empty((0, 12, x_num))
target = np.empty((0,))


_sequence, _target = convert_to_timeseries(train_df.head(950), interval = 12)

sequence = np.vstack((sequence, _sequence))
target = np.hstack((target, _target))
print(sequence.shape)
print(target.shape)


# In[284]:


# making test set

train_df['dummy'] = 0
test_df['dummy'] = 0

X_test, _ = convert_to_timeseries(pd.concat([train_df.iloc[:,:4], test_df.iloc[:,:4]], axis = 0), interval=12)
X_test = X_test[-50:, :, :]

# 만들어 두었던 dummy feature 제거
train_df.drop('dummy', axis = 1, inplace = True)
test_df.drop('dummy', axis = 1, inplace = True)
print(X_test.shape)


# In[285]:


#1-3 Model training (using LSTM model)

from keras.models import Sequential
from keras.layers import Dense,LSTM
from keras.callbacks import EarlyStopping
model=Sequential()
model.add(LSTM(32, input_shape=sequence.shape[-2:]))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))                
model.compile(optimizer='rmsprop', loss='mse',metrics=['mae'])
model.summary()
early_stop = EarlyStopping(monitor='mae', patience=5, verbose=1)


# In[286]:


model.fit(    
    sequence, target,
    epochs=30,
    batch_size=64,
    verbose=2,
    shuffle=False,
    callbacks=[early_stop]
)


# In[298]:


pred = model.predict(X_test)
pred = pd.DataFrame(pred, index = test_df.index)
error =(test_df['D'].values - pred.values)**2
error.mean()
#MSE OF TEST datasetint


# In[3]:


#P2
# FOR MONTLY Sales, Earnings data to calculate Growth Rate for quarter
# Suppose that they are in a same dataset, which column names are Sales,Earnings each.
# let 4 months to be 1 quarter
def growth(df):
    answer = []
    df['quarter'] = np.repeat([1,2,3,4], int(len(df)//4))
    qua_sale = df.groupby['quarter']['Sales'].mean()
    qua_ear = df.groupby['quarter']['Earnings'].mean()
    growth_s = qua_sale.pct_change()[1:].fillna(0)
    growth_e = qua_ear.pct_change()[1:].fillna(0)
    return(growth_s,growth_e)


# In[155]:


#P12
data_12 = pd.read_csv('C:/Users/study gil/Documents/P12_input001.csv')
data_12 = data_12
data_12.columns = ['time','Y']
data_12['time'] = np.arange(1,data_12.shape[0]+1)
data_12.isnull().sum() # there is no missing value


# In[125]:


plt.figure(figsize=(10,10))
plt.plot(data_12);
# there is an overall trend which is increasing
# it seems not to be stationary


# In[126]:


plt.figure(figsize=(10,10))
plt.plot(data_12)
plt.axis([0,200,0,5]);
#In closer look they fluctuates with similar time interval.


# In[178]:


# to remove the trend effect,
# first take differenciation
y_d1 = data_12['Y'].diff().fillna(0)
plt.figure(figsize=(10,10))
plt.plot(y_d1);


# In[179]:


from statsmodels.tsa.stattools import adfuller
adfuller(y_d1)
# It is stationary


# In[181]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(y_d1)
plot_pacf(y_d1)
plt.show()


# In[ ]:



"""
By looking at the ACF and PACF PLOT,
1. ACF shows the cutoff after order 1.
2. PACF shows the exponentially decreasing pattern.
Hence, we can say that the model follows Moving average model with lag order 1 (ARIMA (0,1,1))
""" 


# In[2]:


#C2
def solution(numbers):
    s1 = []
    s2 = []
    for i in range(len(numbers)):
        if numbers[i]%2 == 0:
            s1.append(numbers[i]) # make even list
        else:
            s2.append(numbers[i]) # make odd list
    return sorted(s1,reverse= True)+sorted(s2,reverse= True) # ordering and make list
ex = [4,3,1,2,6,7,10]
solution(ex)


# In[100]:


#C4
def solution(N):
    if N < 3:
        return 1
    else:
        return solution(N-2) + solution(N-1) # use recursive function
    
solution(4) # Simple example for getting 4th element.


# In[101]:


#C101
# get exp(x) value using taylor's expansion
def solution(x,p):
    n = 1
    temp = 0
    k = 1
    t = x
    while ((1/k) >= 1e-10): # repeating for get asympototic value
        temp += (t/k)
        n += 1
        k *= n 
        t *= x
    return(round(1+temp,p)) # round upto pth decimal points

solution(1,5) # Simple example for getting exp(x) upto 5 decimal points

