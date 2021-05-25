import pymysql
from datetime import datetime


class Count:
    def __init__(self):
        self.conn = pymysql.connect(host='localhost',
                                    user='root',
                                    password='1234',
                                    db='stock_price',
                                    charset='utf8')

    def __del__(self):
        self.conn.close()

    def counting(self):
        now = datetime.now()
        datestr = []
        start_date = input("시작날짜 : ")
        end_date = input("종료 날짜 : ")

        with self.conn.cursor() as curs:
            sql = f"SELECT date FROM opened_date WHERE date BETWEEN '{start_date}' AND '{end_date}'"
            length = curs.execute(sql)
            date = curs.fetchall()
            date = list(date)

        for datedate in date:
            datestr.append(datedate[0].strftime("%Y-%m-%d"))

        return datestr, length