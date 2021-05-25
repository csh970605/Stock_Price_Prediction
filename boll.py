import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import pymysql
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from datetime import timedelta,datetime


# 볼린저밴드 : 주가는 상한선과 하한선 사이에서 대부분 움직인다. 캔들이 상한선을 뚫으면 저항선이 뚫렸기 때문에 올라갈 확률이높고
#            캔들이 하한선을 뚫으면 받쳐주는 지지대가 없기 때문에 내려갈 확률이 높다


def db_query():
    item_name = 'CJ대한통운'
    msql = pymysql.connect(
        host='localhost',
        port=int(3306),
        user='root',
        passwd='1234',
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

raw_df = db_query()

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












