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

class Bollinger:
    def __init__(self, item_name, start_date, end_date):
        self.rnn = Rnn(item_name, start_date, end_date)
        self.rnn.run()
        self.raw_df = self.rnn.get_pred_df()
        print(self.raw_df)

        # df = pd.read_csv('동화약품.csv')

        self.raw_df = self.raw_df.set_index(pd.DatetimeIndex(self.raw_df['date'].values))  # 날짜 우선으로 데이터 전처리

        period = 20
        self.raw_df['MA'] = self.raw_df['close'].rolling(window=period).mean()  # 이동평균선 구하기
        self.raw_df['STD'] = self.raw_df['close'].rolling(window=period).std()  # 표준편차
        self.raw_df['Upper'] = self.raw_df['MA'] + (self.raw_df['STD'] * 2)  # 상한밴드
        self.raw_df['Lower'] = self.raw_df['MA'] - (self.raw_df['STD'] * 2)  # 하한밴드

        column_list = ['close', 'MA', 'Upper', 'Lower']

        self.new_df = self.raw_df[period - 1:]  # 새로운 데이터 프레임 생성

        self.fig = plt.figure(figsize=(12.2, 6.4))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.x_axis = self.new_df.index

    def get_dataframe(self):
        return self.raw_df


    def draw_plot(self):
        # 그래프 나타내기
        self.ax.fill_between(self.x_axis, self.new_df["Upper"], self.new_df['Lower'], color='grey')
        self.ax.plot(self.x_axis, self.new_df['close'], color='gold', lw=3, label='close price', alpha=0.5)
        self.ax.plot(self.x_axis, self.new_df['MA'], color='blue', lw=3, label='Moving Average', alpha=0.5)

        self.ax.set_xlabel('date')
        self.ax.set_ylabel('price')
        plt.xticks(rotation=45)
        self.ax.legend()
        plt.show()








