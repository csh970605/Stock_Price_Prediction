#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import requests
import numpy as np
import pymysql


class Get_Stock_From_Naver:  #

    def trans_name_to_code(self):
        pd.set_option('display.max_rows', None)
        self.code_df = pd.read_html('http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13',
                                    header=0)[0]
        self.code_df.종목코드 = self.code_df.종목코드.map('{:06d}'.format)
        self.code_df = self.code_df[['회사명', '종목코드']]
        self.code_df = self.code_df.rename(columns={'회사명': 'name', '종목코드': 'code'})
        self.code_df.sort_values(by=['code'], axis=0).head(5)
        return self.code_df

    def get_url(self, item_name, code_df):
        self.code_df = code_df
        self.code = self.code_df.query("name=='{}'".format(item_name))['code'].to_string(index=False)
        self.url = 'http://finance.naver.com/item/sise_day.nhn?code={code}'.format(code=self.code.strip())
        print('요청 URL = {}'.format(self.url))
        return self.url

    def get_data(self, item_name):
        self.item_name = item_name
        self.code_df = self.trans_name_to_code()
        self.url = self.get_url(self.item_name, self.code_df)
        self.df = pd.DataFrame()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                          '(KHTML, like Gecko) Chrome/88.0.4324.190 Safari/537.36'}
        for self.page in range(1, 100):
            self.pg_url = '{url}&page={page}'.format(url=self.url, page=self.page)
            self.pg_url = requests.get(self.pg_url, headers=self.headers).text
            self.df = self.df.append(pd.read_html(self.pg_url, header=0)[0], ignore_index=True)

        self.df = self.df.dropna()
        self.df = self.df.rename(columns={'날짜': 'date', '종가': 'close', '전일비': 'diff',
                                          '시가': 'open', '고가': 'high', '저가': 'low', '거래량': 'volume'})
        self.df[['close', 'diff', 'open', 'high', 'low', 'volume']] =             self.df[['close', 'diff', 'open', 'high', 'low', 'volume']].astype(int)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values(by=['date'], ascending=False)
        self.df = self.df.query("date >= '2016-08-01' and date <= '2020-03-12'")
        self.df = self.df.reset_index()
        
        print(self.df)
        return self.df

    def trans_data(self, df):
        self.df = df
        self.date = self.df['date']
        self.close = self.df['close']
        self.open = self.df['open']
        self.high = self.df['high']
        self.low = self.df['low']
        self.diff = self.df['diff']
        self.volume = self.df['volume']
        return self.df


if __name__ == '__main__':
    stock_price = Get_Stock_From_Naver()
    item_name = '삼성전자'
    df = stock_price.get_data(item_name)
    df = stock_price.trans_data(df)
    msql = pymysql.connect(
        host='localhost',
        port=int(3306),
        user='root',
        passwd='4368',
        db='p_qksehcpdhkqksehcpwkdql'
    )
    table_name = 'tkatjdwjswk'
    cursor = msql.cursor()
    for i in range(len(df)):
        insert_data = df.iloc[i]
        sql = 'insert ignore into ' + table_name + '(no , date, close, diff, open, high, low, volume) values '                                                    '(%s, %s, %s, %s, %s, %s, %s, %s);'
        data = (i, insert_data['date'], insert_data['close'], insert_data['diff'], insert_data['open'],
                insert_data['high'], insert_data['low'], insert_data['volume'])
        cursor.execute(sql, data)
        msql.commit()

    sql = 'select * from ' + table_name
    cursor.execute(sql)
    rows = cursor.fetchall()
   # print(rows)
    
    msql.close()


# In[3]:


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import numpy as np
import matplotlib.pyplot as plt
#from Investar import Analyzer

mk = Get_Stock_From_Naver()
item_name = '삼성전자'
df = mk.get_data(item_name)
raw_df = stock_price.trans_data(df)




window_size = 10 
data_size = 5

def MinMaxScaler(data):
    """최솟값과 최댓값을 이용하여 0 ~ 1 값으로 변환"""
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # 0으로 나누기 에러가 발생하지 않도록 매우 작은 값(1e-7)을 더해서 나눔
    return numerator / (denominator + 1e-7)

dfx = raw_df[['open','high','low','volume', 'close']]
dfx = MinMaxScaler(dfx)
dfy = dfx[['close']]

x = dfx.values.tolist()
y = dfy.values.tolist()

data_x = []
data_y = []
for i in range(len(y) - window_size):
    _x = x[i : i + window_size] # 다음 날 종가(i+windows_size)는 포함되지 않음
    _y = y[i + window_size]     # 다음 날 종가
    data_x.append(_x)
    data_y.append(_y)
print(_x, "->", _y)

train_size = int(len(data_y) * 0.7)
train_x = np.array(data_x[0 : train_size])
train_y = np.array(data_y[0 : train_size])

test_size = len(data_y) - train_size
test_x = np.array(data_x[train_size : len(data_x)])
test_y = np.array(data_y[train_size : len(data_y)])

# 모델 생성
model = Sequential()
model.add(LSTM(units=10, activation='relu', return_sequences=True, input_shape=(window_size, data_size)))
model.add(Dropout(0.1))
model.add(LSTM(units=10, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=1))
model.summary()

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(train_x, train_y, epochs=60, batch_size=30)
pred_y = model.predict(test_x)

# Visualising the results
plt.figure()
plt.plot(test_y, color='red', label='real SEC stock price')
plt.plot(pred_y, color='blue', label='predicted SEC stock price')
plt.title('SEC stock price prediction')
plt.xlabel('time')
plt.ylabel('stock price')
plt.legend()
plt.show()

# raw_df.close[-1] : dfy.close[-1] = x : pred_y[-1]
print("Tomorrow's SEC price :", raw_df.close[-1] * pred_y[-1] / dfy.close[-1], 'KRW')


# In[ ]:




