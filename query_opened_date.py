import pymysql
from datetime import datetime

conn = pymysql.connect(host='localhost',
                       user='root',
                       password='1234',
                       db='stock_price',
                       charset='utf8')

now = datetime.now()
datestr = []
start_date = input("시작날짜 : ")
end_date = input("종료 날짜 : ")

with conn.cursor() as curs:
    sql = f"SELECT date FROM opened_date WHERE date BETWEEN '{start_date}' AND '{end_date}'"
    length = curs.execute(sql)
    date = curs.fetchall()
    date = list(date)
    for datetime in date:
        datestr.append(now.strftime("%Y-%m-%d"))
    print(length)
    print(datestr)
