import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import pymysql
from prediction import Rnn
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from datetime import timedelta,datetime


# 볼린저밴드 : 주가는 상한선과 하한선 사이에서 대부분 움직인다. 캔들이 상한선을 뚫으면 저항선이 뚫렸기 때문에 올라갈 확률이높고
#            캔들이 하한선을 뚫으면 받쳐주는 지지대가 없기 때문에 내려갈 확률이 높다

rnn = Rnn('삼성전자', '2020-05-08', '2020-08-08')
rnn.run()
raw_df = rnn.get_pred_df()
print(raw_df)

# df = pd.read_csv('동화약품.csv')

raw_df = raw_df.set_index(pd.DatetimeIndex(raw_df['date'].values)) # 날짜 우선으로 데이터 전처리

period = 20
raw_df['MA'] = raw_df['close'].rolling(window=period).mean() # 이동평균선 구하기
raw_df['STD'] = raw_df['close'].rolling(window=period).std() # 표준편차
raw_df['Upper'] = raw_df['MA'] + (raw_df['STD'] * 2) # 상한밴드
raw_df['Lower'] = raw_df['MA'] - (raw_df['STD'] * 2) # 하한밴드

column_list = ['close', 'MA', 'Upper', 'Lower']

new_df = raw_df[period-1:] # 새로운 데이터 프레임 생성

fig = plt.figure(figsize=(12.2, 6.4))
ax = fig.add_subplot(1,1,1)
x_axis = new_df.index

# 그래프 나타내기
ax.fill_between(x_axis, new_df["Upper"], new_df['Lower'], color = 'grey')
ax.plot(x_axis, new_df['close'], color = 'gold', lw =3, label = 'close price', alpha = 0.5)
ax.plot(x_axis, new_df['MA'], color = 'blue', lw =3, label = 'Moving Average', alpha = 0.5)

ax.set_xlabel('date')
ax.set_ylabel('price')
plt.xticks(rotation = 45)
ax.legend()
plt.show()






