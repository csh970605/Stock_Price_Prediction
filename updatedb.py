from threading import Timer
import pandas as pd
from urllib.request import urlopen
import urllib, pymysql, calendar, time, json
import numpy as np
import requests
import pymysql
from datetime import datetime

class Get_Stock_From_Naver:
    # 객체가 생성될 때# 접속하고, 소멸될 때 접속을 해제한다
    def __init__(self):
        self.conn = pymysql.connect(host='localhost',
                                    user='root',
                                    password='1234',
                                    db='stock_price',
                                    charset='utf8')

        with self.conn.cursor() as curs:
            sql = """
            CREATE TABLE IF NOT EXISTS com_info (
                code VARCHAR(50),
                name VARCHAR(50),
                last_update DATE,
                PRIMARY KEY (code))
            """
            curs.execute(sql)
            sql = """
            CREATE TABLE IF NOT EXISTS one_day_price(
                code VARCHAR(50),
                date date,
                open bigint(50),
                high bigint(50),
                low bigint(50),
                close bigint(50),
                diff bigint(50),
                volume bigint(50),
                primary key (code, date))
            """

            curs.execute(sql)
        self.conn.commit()

        self.codes = dict()
        self.update_com_info()  # update_comp_info() 메서드로 KRX 주식 코드를읽어 com_info 테이블에 업데이트



    def __del__(self):
        self.conn.close()

    def read_krx_code(self):  # krx로부터 상장기업 목록 파일을 읽어와서 데이터프레임으로 변환
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                          '(KHTML, like Gecko) Chrome/88.0.4324.190 Safari/537.36'}
        url = 'http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13'
        krx = pd.read_html(url, header=0)[0]   # 상장법인목록을  read_html() 함수로 읽는다
        krx = krx[['종목코드', '회사명']]  # 종목코드 칼럼과 회사명만 남긴다 [[]] : 특정칼럼 뽑아 순서대로 재구성
        krx = krx.rename(columns={'종목코드':'code','회사명':'name'})  # 한글 칼럼 -> 영어로
        krx.code = krx.code.map('{:06d}'.format) # 종목코드 형식을 {:06d}형식의 문자열로 변경
        krx.sort_values(by=['code'], axis=0)
        return krx

    def update_com_info(self): # 종목코드를 comp_jnfo 테이블에 업데이트한 후 딕셔너리에 저장
        sql = "SELECT * FROM com_info order by code desc"
        df = pd.read_sql(sql, self.conn) # comp_info 테이블을 read_sql() 함수로 읽는다.
        for idx in range(len(df)):
            self.codes[df['code'].values[idx]]=df['name'].values[idx]
            # 위에서 읽은 데이터프레임을 이용해서 종목코드와 회사명으로 codes 딕셔너리를 만든다.

        with self.conn.cursor() as curs:
            sql = "SELECT max(last_update) FROM com_info"
            curs.execute(sql)
            rs = curs.fetchone()  # SELECT max() 〜 구문을 이용해서 DB에서 가장 최근 업데이트 날짜를 가져온다.
            today = datetime.today().strftime('%Y-%m-%d')

            if rs[0] == None or rs[0].strftime('%Y-%m-%d') < today:
                # 위에서 구한날짜가 존재하지않거나 오늘보다 오래된경우에만 업데이트한다.
                krx = self.read_krx_code()  # KRX 상장기업 목록 파일을 읽어서 krx 데이터프레임에 저장한다.
                for idx in range(len(krx)):
                    code = krx.code.values[idx]
                    name = krx.name.values[idx]
                    sql = f"REPLACE INTO com_info (code, name, last_update) VALUES ('{code}', '{name}', '{today}')"
                    curs.execute(sql)  # REPLACE INTO 구문을 이용해서 종목코드，회사명，오늘날짜，행을 DB에 저장한다.
                    self.codes[code] = name  # codes 딕셔너리에 키-값으로 종목코드와 회사명을 추가한다.
                    tmnow = datetime.now().strftime('%Y-%m-%d %H:%M')
                    print(f"[{tmnow}] {idx:04d} REPLACE INTO com_infoVALUES ({code}, {name}, {today})")
                    self.conn.commit()

    def read_naver(self, code):
        # 네이버에서 주식 시세를 읽어서 데이터프레임으로 변환
        url = 'http://finance.naver.com/item/sise_day.nhn?code={code}'.format(code=code.strip())
        df = pd.DataFrame()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                          '(KHTML, like Gecko) Chrome/88.0.4324.190 Safari/537.36'}
        for page in range(1, 100):
            pg_url = '{url}&page={page}'.format(url=url, page=page)
            pg_url = requests.get(pg_url, headers=headers).text
            df = df.append(pd.read_html(pg_url, header=0)[0], ignore_index=True)

        df = df.dropna()
        df = df.rename(columns={'날짜': 'date', '종가': 'close', '전일비': 'diff',
                                          '시가': 'open', '고가': 'high', '저가': 'low', '거래량': 'volume'})
        df[['close', 'diff', 'open', 'high', 'low', 'volume']] = \
            df[['close', 'diff', 'open', 'high', 'low', 'volume']].astype(int)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by=['date'], ascending=True)
        df = df.query("date >= '2016-08-01'")
        df = df.reset_index().drop(['index'], axis=1)
        return df

    def replace_into_db(self, df, num, code, name):
        # 네이버에서 읽어온 주식 시세를 DB에 Replace
        with self.conn.cursor() as curs:
            for r in df.itertuples():
                # 인수로 넘겨받은 데이터프레임을 튜플로 순회처리한다.
                sql = f"REPLACE INTO one_day_price VALUES ('{code}', "\
                      f"'{r.date}', {r.open}, {r.high}, {r.low}, {r.close}, "\
                      f"{r.diff}, {r.volume})"
                curs.execute(sql) # REPLACE INTO 구문으로 one_day_price 테이블을 업데이트한다.
            self.conn.commit() # commit() 함수를 호출해 디비에 반영한다.
            print('[{}] #{:04d} {} ({}) : {} rows > REPLACE INTO one_day_'\
                  'price [OK]'.format(datetime.now().strftime('%Y-%m-%d'\
                  ' %H:%M'), num+1, name, code, len(df)))

    def update_daily_price(self):
        # KRX 상장법인의 주식 시세를 네이버로부터 읽어서 DB에 업데이트
        print(self.codes)
        for idx, code in enumerate(self.codes): # self.codes 딕셔너리에 저장된 모든 종목코드에 대해 순회처리한다.
            print(idx, code + '\n')
            df = self.read_naver(code)
            if df is None:
                continue
            self.replace_into_db(df, idx, code, self.codes[code])
            #  일별시세 데이터프레임이 구해지면 replace_into_db() 메서드로 DB에 저장한다.




if __name__ == '__main__':

    dbu = Get_Stock_From_Naver()
    dbu.update_daily_price()
    dbu.conn.close()