
import pandas as pd
import numpy as np

temp = pd.DataFrame({'날짜_일반': ['2021/01/01', '2021/01/02', '2021/01/03', '2021/01/04', '2021/01/05'],
                     '날짜_시간': ['2021-01-01 1:12:10', '2021-01-02 1:13:45', '2021-01-03 2:50:10', 
                                   '2021-01-04 3:12:30', '2021-01-05 5:40:20'],
                     '날짜_특수': ['21-01-01', '21-01-02', '21-01-03', '21-01-04', '21-01-05'],
                     '범주': ['금', '토', '일', '월', '화']})

td_series = pd.Series(pd.to_timedelta(['1 days 02:30:45', '3 days 10:15:30']))
components = td_series.dt.components
print(components)
print(type(components))
print(components.days)
print(components.hours)
print(components.minutes)
print(components.seconds)