#!/usr/bin/env python
# coding: utf-8

# In[21]:


import datetime as dt
from datetime import timedelta,datetime

#####저장할 배열
a=[]
#####

####코드를 돌린날 기준 3년후 까지의 휴장일 배열 데이터 출력
d=dt.datetime(2019,1,1)
d2=d
d3=d
we=d.weekday()

#######주말 제외한 날짜 배열에 저장
for i in range(1825):
    
    if we==5 or we==6 :
        d2=d+timedelta(days=i)
        d3=d2+timedelta(days=1)
        we=d3.weekday()
    else:
        
        d2=d+timedelta(days=i)
        d3=d2+timedelta(days=1)
        we=d3.weekday()
        date_string=d2.strftime('%Y-%m-%d')
        a.append(date_string)
#########

#print(a)

######주말 제외한 공휴일 
a.remove('2019-01-01')
a.remove('2019-02-04')
a.remove('2019-02-05')
a.remove('2019-02-06')
a.remove('2019-03-01')
#a.remove('2019-05-05')
#a.remove('2019-05-12')
a.remove('2019-06-06')
a.remove('2019-08-15')
a.remove('2019-09-12')
a.remove('2019-09-13')
#a.remove('2019-09-14')
a.remove('2019-10-03')
a.remove('2019-10-09')
a.remove('2019-12-25')


a.remove('2020-01-01')
a.remove('2020-01-24')
#a.remove('2020-01-25')
#a.remove('2020-01-26')
#a.remove('2020-03-01')
a.remove('2020-04-30')
a.remove('2020-05-05')
#a.remove('2020-06-06')
#.remove('2020-08-15')
a.remove('2020-08-17')
a.remove('2020-09-30')
a.remove('2020-10-01')
a.remove('2020-10-02')
#a.remove('2020-10-03')
a.remove('2020-10-09')
a.remove('2020-12-25')

a.remove('2021-05-19')
a.remove('2021-09-20')
a.remove('2021-09-21')
#a.remove('2021-05-22')
#a.remove('2021-10-09')
a.remove('2021-12-31')

#a.remove('2022-01-01')
a.remove('2022-02-01')
a.remove('2022-02-02')
a.remove('2022-02-03')
a.remove('2022-03-01')
a.remove('2022-05-05')
a.remove('2022-06-06')
a.remove('2022-08-15')
a.remove('2022-09-09')
#a.remove('2022-09-10')
#a.remove('2022-09-11')
a.remove('2022-10-03')
#a.remove('2022-10-09')
#a.remove('2022-12-25')

#a.remove('2023-01-01')
#a.remove('2023-01-21')
#a.remove('2023-01-22')
a.remove('2023-01-23')
a.remove('2023-03-01')
a.remove('2023-05-05')
a.remove('2023-05-26')
a.remove('2023-06-06')
a.remove('2023-08-15')
a.remove('2023-09-28')
a.remove('2023-09-29')
#a.remove('2023-09-30')
a.remove('2023-10-03')
a.remove('2023-10-09')
a.remove('2023-12-25')



#a.remove('2024-01-01')
#a.remove('2024-02-09')
#a.remove('2024-02-10')
#a.remove('2024-02-11')
#a.remove('2024-03-01')
#a.remove('2024-05-05')
#a.remove('2024-05-15')
#a.remove('2024-06-06')
#a.remove('2024-08-15')
#a.remove('2024-09-16')
#a.remove('2024-09-17')
#a.remove('2024-09-18')
#a.remove('2024-10-03')
#a.remove('2024-10-09')
#a.remove('2024-12-25')


print(a)
   





# In[ ]:




