
# 작업형 제1유형
import pandas as pd
pd.set_option('display.width', 120)
path1 = "https://raw.githubusercontent.com/Soyoung-Yoon/bigdata/main/"
path2 = "https://raw.githubusercontent.com/Soyoung-Yoon/data_01/main/"

# 6-1) event_log_with_time.csv 데이터를 사용한다.
# 지속시간(duration)을 구하고, 지속시간 중 가장 큰 5개 의 평균을 구한다.
# duration = end_datetime - start_datetime
# 해당 평균을 구하여, 반올림하여 초 단위 정수로 출력한다.
df = pd.read_csv(path2 + "event_log_with_time.csv")
# print(df.head(3))
df['start_datetime'] = df['start_datetime'].astype('datetime64[ns]')
df['end_datetime'] = df['end_datetime'].astype('datetime64[ns]')
duration = df['end_datetime'] - df['start_datetime']
value = duration.sort_values(ascending=False).head(5).mean()
# print(round(value.total_seconds())) # 594

# 6-2) event_log_with_time.csv 데이터를 사용한다.
# 'Run' 이벤트 중 가장 오래 지속된 이벤트의 시간을 반올림하여 분 단위 정수로 구하시오.
# 지속시간(duration) = end_datetime - start_datetime
components = duration.max().components
result = components.minutes + (components.seconds / 60)
# print(round(result)) # 10

# 6-3) event_log_with_time.csv 데이터를 사용한다.
# start_datetime이 2024년 4월 1일부터 5일까지의 데이터를 대상으로, 발생 빈도가 가장 높은 event를 찾고, 
# 해당 event에 대한 value 평균을 구하시오.
# 결과는 반올림하여 정수로 출력한다.
cond = (df['start_datetime'].dt.year==2024) & (df['start_datetime'].dt.month==4) & \
       (df['start_datetime'].dt.day >= 1) & (df['start_datetime'].dt.day <= 5)
df = df[cond].copy()
max_event = df['event'].value_counts().idxmax()
# print(round(df[df['event']==max_event]['value'].mean())) # 54

# 6-4) event_log_02.csv 파일을 사용하여 다음을 구하시오.
# start_datetime 기준 2020년 1월에 발생한 이벤트(event) 중 가장 많이 발생한 이벤트 종류를 A,
# 가장 적게 발생한 이벤트 종류를 B라고 한다.
# 2020년 1월 발생한 A이벤트의 end_datetime 기준 가장 나중에 발생한 이벤트의 일(day)을 Aday,
# B이벤트의 end_datetime 기준 가장 먼저 발생한 이벤트의 일(day)을 Bday라고 할 때 두 수의 합을 정수로 출력하라.
# 2020-01-25 의 경우 일(day)은 25이다.
df = pd.read_csv(path2 + "event_log_02.csv")
# print(df.head(3))
cond = df['start_datetime'].str[:7] == '2020-01'
df = df[cond].copy()
A, B = df['event'].value_counts().agg(['idxmax','idxmin'])
print(A,B) # Run Start
df['end_datetime'] = df['end_datetime'].astype('datetime64[ns]')
Aday = df.loc[df['event']==A, 'end_datetime'].sort_values().iloc[-1].day
Bday = df.loc[df['event']==B, 'end_datetime'].sort_values().iloc[0].day
# print(Aday, Bday) # 30 10
print(int(Aday) + int(Bday)) # 32

# 6-5) event_log_02.csv 파일을 사용하여 다음을 구하시오.
# 월별 지속시간(duration)의 합계 중에서 가장 큰 값에서 가장 작은 값을 뺀 값을 구하고, 
# 그 값의 시간, 분, 초를 모두 더한 값을 정수로 출력하시오.
# 지속시간(duration) = end_datetime - start_datetime
# 월(month) : start_datetime의 월 정보 사용
df['start_datetime'] = df['start_datetime'].astype('datetime64[ns]')
df['end_datetime'] = df['end_datetime'].astype('datetime64[ns]')
df['duration'] = df['end_datetime'] - df['start_datetime']
df['month'] = df['start_datetime'].dt.month
max_time = df.groupby('month')['duration'].sum().max()
min_time = df.groupby('month')['duration'].sum().min()
components = (max_time - min_time).components
result = components.hours + components.minutes + components.seconds
# print(result) # 62

# 작업형 제3유형
# 다음과 같은 다중선형회귀 모형을 사용한 회귀모델을 만들고 결과를 확인합니다.
# tips.csv 데이터를 사용합니다.
# 모델 생성시 상수항(=절편)을 포함하도록 합니다.
# 종속변수 : total_bill
# 독립변수 : tip, size, time, smoker
df = pd.read_csv(path1 + "tips.csv")
# 6-1) 위의 조건에 맞게 선형 회귀 모델을 생성하고 summary를 출력한다.
from statsmodels.api import OLS
formula = "total_bill ~ tip + size + time + smoker"
model = OLS.from_formula(formula, df).fit()
# print(model.summary())

#6-2) size가 2증가하면 total_bill이 몇 증가하는가? 결과는 반올림하여 소수점아래 4자리까지 출력한다.
import numpy as np
# print(round(model.params['size'] * 2, 4)) # 6.9312

#6-3) 다른 변수들을 통제한 상태에서, Dinner와 Lunch 중 total_bill이 낮은 시간은?
result = "Lunch" if model.params['time[T.Lunch]'] < 0 else "Dinner"
# print(result) # Lunch

#6-4) 다른 변수들을 통제한 상태에서, Dinner와 Lunch의 total_bill 차이는 평균적으로 얼마인가?
# 결과는 반올림하여 소수점아래 4자리까지 출력한다.
result = abs(model.params['time[T.Lunch]'])
# print(round(result, 4)) # 1.5637

#6-5) 다른 변수들을 통제한 상태에서, Lunch가 Dinner에 비해 평균적으로 total_bill이 얼마나 변화하는가?
# 결과는 반올림하여 소수점아래 4자리까지 출력한다.
# print(round(model.params['time[T.Lunch]'], 4)) # -1.5637

#6-6) 회귀분석 결과에서 잔차의 자유도를 정수로 출력한다.
# print(int(model.df_resid)) # 239

#6-7) 다음의 표본에 대해 예측값을 구해, 반올림하여 소수점아래 3자리까지 출력한다.
# tip=5, size=4, time='Dinner', smoker='Yes'
data = pd.DataFrame({'tip': [5], 'size': [4], 'time': ['Dinner'], 'smoker': ['Yes']})
# print(round(model.predict(data)[0], 3)) # 32.862

#6-8) 아래의 표본 데이터를 사용하여 예측값의 99% 신뢰구간의 하한과 상한의 차이에 대한 절댓값을 출력한다.
# 결과는 반올림하여 소수점아래 4자리까지 출력한다.
# tip=3, size=2, time='Lunch', smoker='No'
data =  pd.DataFrame({'tip': [3], 'size': [2], 'time': ['Lunch'], 'smoker': ['No']})
result = model.get_prediction(data)
lower, upper = result.conf_int(alpha=0.01)[0]
# print(round(upper - lower, 4)) # 4.1967

# 6-9) 통계적으로 가장 유의미한 변수와 해당 변수의 회귀계수에 대한 신뢰구간 중 하한을 출력한다.
# 신뢰구간의 하한은 반올림하여 소수점아래 3자리까지 출력한다.'
# 신뢰수준 : 99%
# 출력 예) size 10.123
temp = model.pvalues[1:].idxmin()
result = model.conf_int(alpha=0.01).loc[temp, 0]
# print(temp, round(result, 3)) # tip 2.302

# 6-10) 원본 데이터를 사용하여 mae를 구해, 반올림하여 소수점 아래 4자리까지 출력한다.
from sklearn.metrics import mean_absolute_error as mae
y_true = df['total_bill']
y_pred = model.predict(df)
# print(round(mae(y_true, y_pred), 4)) # 4.2008

# 6-11) 회귀분석의 잔차가 정규성을 만족하는지 Shapiro-Wilk 검정을 수행하시오.
# 검정 후, 검정 결과의 검정통계량을 반올림하여 소수점 아래 3자리까지 출력하시오.
residual = y_true - y_pred
import scipy.stats
s = pd.Series(dir(scipy.stats))
# print(s[s.str.contains('sha')])
from scipy.stats import shapiro
statistic, _ = shapiro(residual)
# print(round(statistic, 3)) # 0.936

# 6-12) 회귀모형의 잔차에 대해 독립성(자기상관)을 알아보기 위해,
# Durbin-Watson 검정을 사용한 통계량을 반올림하여 소수점 아래 3자리까지 출력하고,
# 자기상관이 없다고 판단되는 경우 "자기상관 없음",
# 자기상관이 있다고 판단되는 경우 "자기상관 있음"으로 답하세요.
# 예) 123.1234 자기상관 없음
from statsmodels.stats.stattools import durbin_watson
dw = durbin_watson(residual)
# if round(dw, 3) >= 1.5 and round(dw, 3) <=2.5:
#     print(f"{dw:.3f} 자기상관 없음")
# else:
#     print(f"{dw:.3f} 자기상관 있음")
# 2.152 자기상관 없음

#6-13) tip을 종속변수로 하고,
# total_bill과 smoker에 대한 교호작용 분석을 위한 모델을 생성하고, summary()를 출력해 확인합니다.
formula2 = "tip ~ total_bill + C(smoker) + total_bill:C(smoker)"
model2 = OLS.from_formula(formula2, df).fit()
# print(model2.summary())

# 6-14) 위의 교호작용항을 고려할 때,
# 흡연자일 경우, 총 금액(total_bill)이 팁에 미치는 영향이 증가하는가? 감소하는가?
# 증가 또는 감소를 표기하고,
# 그 증가 또는 감소량을 절댓값을 취해 반올림하여 소수점 아래 4자리까지 출력하세요.
# 두 값의 사이에 공백을 1개 넣어 한 줄로 작성하세요.
# print("감소", round(abs(model2.params['total_bill:C(smoker)[T.Yes]']), 4))
# 감소 0.0676