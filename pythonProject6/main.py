import pandas as pd
import pymysql
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import numpy as np
import matplotlib.pyplot as plt
import count

from datetime import timedelta, datetime

class Rnn:
    def __init__(self, item_name):
        self.item_name = item_name

        self.msql = pymysql.connect(
            host='localhost',
            port=int(3306),
            user='root',
            passwd='1234',
            db='stock_price',
            autocommit=True,
            cursorclass=pymysql.cursors.DictCursor
        )
        self.cursor = self.msql.cursor()

    def __del__(self):
        self.msql.close()

    def db_query(self):
        sql = f"select code from com_info where name = '{self.item_name}'"
        self.cursor.execute(sql)
        self.item_code = self.cursor.fetchall()
        self.df = pd.DataFrame(self.item_code)
        self.item_code = self.df.iloc[0]['code']

        sql = f"select * from one_day_price where code = '{self.item_code}'"
        self.cursor.execute(sql)
        stock_data = self.cursor.fetchall()
        self.df = pd.DataFrame(stock_data)
        return self.df

####################여기 떼내기#############
    def define_window_size(self) :
        self.raw_df = self.db_query()
        # print(raw_df)
        self.window_size = 10
        self.data_size = 5
        print(self.window_size)
        print(self.data_size)
        return self.raw_df



    def minMaxScaler(self, data):
        """최솟값과 최댓값을 이용하여 0 ~ 1 값으로 변환"""
        numerator = data - np.min(data, 0)
        denominator = np.max(data, 0) - np.min(data, 0)

        # 0으로 나누기 에러가 발생하지 않도록 매우 작은 값(1e-7)을 더해서 나눔
        return numerator / (denominator + 1e-7)

    def save_date(self):

        ##학습하는 날짜데이터 저장
        self.dfdate = self.raw_df[['date']]
        # 에측해야되는 날짜데이터
        self.dfdate2 = self.dfdate

        ##현재 시간 조회후 90일 추가
        self.time1 = datetime.now()
        for i in range(90):
            self.time2 = self.time1 + timedelta(days=i)
            date_string = self.time2.strftime('%Y-%m-%d')
            new_data = {
                'date': date_string
            }

            self.dfdate2 = self.dfdate2.append(new_data, ignore_index=True)

        return self.dfdate2

    def train_dataset(self):
        dfx = self.raw_df[['open', 'high', 'low', 'volume', 'close']]
        dfy = dfx[['close']]

        self.mini = np.min(dfy, 0)
        self.mega = np.max(dfy, 0)

        dfx = self.minMaxScaler(dfx)
        dfy = dfx[['close']]

        x = dfx.values.tolist()
        y = dfy.values.tolist()
        da = self.dfdate.values.tolist()

        data_d = []
        data_x = []
        data_y = []

        _d = da

        for i in range(len(y) - self.window_size):
            _x = x[i: i + self.window_size]  # 다음 날 종가(i+windows_size)는 포함되지 않음
            _y = y[i + self.window_size]  # 다음 날 종가
            data_x.append(_x)
            data_y.append(_y)
        print(_x, "->", _y)

        return data_x, data_y


    def create_trainset(self, data_x, data_y):
        train_size = int(len(data_y))
        self.train_x = np.array(data_x[0: train_size])
        self.train_y = np.array(data_y[0: train_size])

        count = int(len(self.dfdate2)) - train_size

        test_size = len(data_y) - count
        self.test_x = np.array(data_x[train_size - count: len(data_x)])
        self.test_y = np.array(data_y[train_size - count: len(data_y)])

        self.train_size2 = int(len(self.dfdate) - test_size)
        self.test_size2 = len(self.dfdate2) - self.train_size2

        self.test_date = np.array(self.dfdate2[train_size: len(self.dfdate2)])

        ##
        print(train_size)
        print(len(data_x))

        return


## 90일 예측 조회 할 경우

    def create_Pmodel(self):
        # 모델 생성
        model = Sequential()
        model.add(LSTM(units=30, activation='relu', return_sequences=True, input_shape=(self.window_size, self.data_size)))
        model.add(Dropout(0.1))
        model.add(LSTM(units=30, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(units=1))
        model.summary()

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(self.train_x, self.train_y, epochs=60, batch_size=30)
        pred_y = model.predict(self.test_x)

        print(len(self.test_date))
        print(len(pred_y))

        # 전처리 데이터 복구
        self.mini = self.mini.values.tolist()
        self.mega = self.mega.values.tolist()
        self.pred_y = pred_y * (self.mega[0] - self.mini[0]) + self.mini[0]
        self.test_y = self.test_y * (self.mega[0] - self.mini[0]) + self.mini[0]

    def draw_model(self):

        # Visualising the results
        plt.figure(figsize=(10, 10))
        # plt.plot(test_date,test_y, color='red', label='real SEC stock price')
        plt.plot(self.test_date, self.pred_y, color='blue', label='predicted SEC stock price')
        plt.title('SEC stock price prediction')
        plt.xlabel('time')
        plt.ylabel('stock price')

        plt.legend()
        plt.show()

    def run(self):

        self.define_window_size()
        self.save_date()
        data_x, data_y = self.train_dataset()
        self.create_trainset(data_x, data_y)
        self.create_Pmodel()
        self.draw_model()


if __name__ == '__main__':
    asdf=Rnn('삼성전자')
    asdf.run()

