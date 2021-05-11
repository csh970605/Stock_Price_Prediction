#!/usr/bin/env python
# coding: utf-8

# In[47]:


import pandas as pd
import pymysql
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import numpy as np
import matplotlib.pyplot as plt

from datetime import timedelta,datetime

def db_query():
    item_name = '삼양홀딩스'
    msql = pymysql.connect(
        host='localhost',
        port=int(3306),
        user='root',
        passwd='4368',
        db='stock_price',
        autocommit=True,
        cursorclass=pymysql.cursors.DictCursor
    )
    cursor = msql.cursor()

    sql = f"select code from com_info where name = '{item_name}'"
    cursor.execute(sql)
    item_code = cursor.fetchall()
    df = pd.DataFrame(item_code)
    item_code = df.iloc[0]['code']

    sql = f"select * from one_day_price where code = '{item_code}'"
    cursor.execute(sql)
    stock_data = cursor.fetchall()
    df = pd.DataFrame(stock_data)
    cursor.close()
    return df





if __name__ == '__main__':
    raw_df = db_query()
    #print(raw_df)
    window_size = 10
    data_size = 5
    
    

raw_df = db_query()




window_size =10
data_size = 5





def MinMaxScaler(data):
    """최솟값과 최댓값을 이용하여 0 ~ 1 값으로 변환"""
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
   
    # 0으로 나누기 에러가 발생하지 않도록 매우 작은 값(1e-7)을 더해서 나눔
    return numerator / (denominator + 1e-7)


    

dfdate=raw_df[['date']]
dfdate2=dfdate



##현재 시간 조회후 90일 추가
time1=datetime.now()
for i in range(90):
    time2=time1+timedelta(days=i)
    date_string=time2.strftime('%Y-%m-%d')
    new_data={
            'date' : date_string
        }
    
   
    dfdate2=dfdate2.append(new_data,ignore_index=True)
    
    
   

print(dfdate2)

dfx = raw_df[['open','high','low','volume', 'close']]
dfy = dfx[['close']]

mini=np.min(dfy,0)
mega=np.max(dfy,0)

dfx = MinMaxScaler(dfx)
dfy = dfx[['close']]



x = dfx.values.tolist()
y = dfy.values.tolist()
da=dfdate.values.tolist()

data_d=[]
data_x = []
data_y = []

_d=da


for i in range(len(y) - window_size):
    _x = x[i : i + window_size] # 다음 날 종가(i+windows_size)는 포함되지 않음
    _y = y[i + window_size]     # 다음 날 종가
    data_x.append(_x)
    data_y.append(_y)
print(_x, "->", _y)



## 90일 예측 조회 할 경우

train_size = int(len(data_y))
train_x = np.array(data_x[0 : train_size])
train_y = np.array(data_y[0 : train_size])

count=int(len(dfdate2))-train_size

test_size = len(data_y)-count
test_x = np.array(data_x[train_size-count : len(data_x)])
test_y = np.array(data_y[train_size-count : len(data_y)])




train_size2=int(len(dfdate)-test_size)
test_size2=len(dfdate2)-train_size2

test_date=np.array(dfdate2[train_size : len(dfdate2)])

##

# 모델 생성
model = Sequential()
model.add(LSTM(units=30, activation='relu', return_sequences=True, input_shape=(window_size, data_size)))
model.add(Dropout(0.1))
model.add(LSTM(units=30, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=1))
model.summary()

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(train_x, train_y, epochs=60, batch_size=30)
pred_y = model.predict(test_x)


print(len(test_date))
print(len(pred_y))

#전처리 데이터 복구
mini=mini.values.tolist()
mega=mega.values.tolist()
pred_y=pred_y*(mega[0]-mini[0])+mini[0]
test_y=test_y*(mega[0]-mini[0])+mini[0]


# Visualising the results
plt.figure(figsize=(30,30))
#plt.plot(test_date,test_y, color='red', label='real SEC stock price')
plt.plot(test_date,pred_y, color='blue', label='predicted SEC stock price')
plt.title('SEC stock price prediction')
plt.xlabel('time')
plt.ylabel('stock price')

plt.legend()
plt.show()



# raw_df.close[-1] : dfy.close[-1] = x : pred_y[-1]
print("Tomorrow's SEC price :", raw_df.close[-1] * pred_y[-1] / dfy.close[-1], 'KRW')


# In[ ]:




