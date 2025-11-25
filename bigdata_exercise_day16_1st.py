
# 공통: 데이터 경로
import pandas as pd
path1 = "https://raw.githubusercontent.com/Soyoung-Yoon/bigdata/main/"
path2 = "https://raw.githubusercontent.com/Soyoung-Yoon/data_01/main/"

# 작업형 제1유형
# 8회-1번) 맥주 소비량 구하기
# 다음 절차에 따라 최종 결과를 구해 정수로 작성하시오.
df = pd.read_csv(path1 + "drinks_09011.csv")
# print(df.head(3))
# 1) 대륙별 맥주의 평균 소비량이 가장 많은 곳을 구하시오.
region = df.groupby('대륙')['맥주'].mean().idxmax()
# 2) 1번에서 구한 대륙에서 다섯번째로 맥주 소비량이 많은 국가를 구하시오.
df = df[df['대륙']==region].copy()
country = df.groupby('국가')['맥주'].sum().sort_values(ascending=False).nlargest(5).idxmin()
# 3) 2번에서 구한 국가의 맥주 소비량을 정수로 작성하시오.
# print(int(df.loc[df['국가'] == country, '맥주'])) # 313

# 8회-2번) 관광객
# 다음의 결과를 구하시오.
df = pd.read_csv(path1 + "tourist_08012.csv")
# print(df.head(3))
# - 1) 관광객비율이 두 번째로 높은 나라의 '관광'수를 a라고 정의하시오.
# > 관광객비율 = 관광 / (관광 + 공무)
df['관광객비율'] = df['관광'] / (df['관광'] + df['공무'])
# country = df.groupby('국적')['관광객비율'].sum().nlargest(2).idxmin()
# a = df.loc[df['국적'] == country, '관광'].iloc[0]
A = df.sort_values('관광객비율',ascending=False).iloc[1]['관광']
# - 2) 관광객 수가 두 번째로 높은 나라의 '공무'수를 b라고 정의하시오.
# b = df.loc[df['국적'] == country, '공무'].iloc[0]
B = df.sort_values('관광', ascending=False).iloc[1]['공무']
# - 3) a + b의 값을 구하여 정수로 출력한다.
# print(int(A+B)) # 239

# 8회-3번) 스케일링
# 다음의 결과를 구하시오.
df = pd.read_csv(path1 + "environment_08013.csv")
# print(df.head(3))
# 1) co컬럼과 nmch컬럼을 사용하여 Min-Max Scale을 시행하시오.
s1 = df['co']
s2 = df['nmch']
co = (s1 - s1.min()) / (s1.max() - s1.min())
nmch = (s2 - s2.min()) / (s2.max() - s2.min())
# print(co, nmch)
#   > Min-Max Scale = (Xn - Xmin) / (Xmax - Xmin)
# 2) Min-Max Scale이 적용된 각 칼럼의 표준편차를 구하시오.
#   > co칼럼의 표준편차 = a, nmch 칼럼의 표준편차 = b
a = co.std(ddof=1)
b = nmch.std(ddof=1)
# 3) 2번에서 구한 a, b를 사용하여 두 값의 차이 값인 a - b의 값을 구해, 반올림하여 소수점 아래 3자리까지 출력한다.
# print(round(a - b, 3)) # -0.026

# 16-4) 연속 평균 이하 유병률 국가
# 국가별 유병률 데이터를 분석 건강 관리가 잘 되어온 국가일 가능성이 높은 국가를 알아보자.
# 다음 절차에 따라 문제를 해결하시오.
df = pd.read_csv(path1 + "worlddata.csv")
# print(df.head(3))
# 1) 1999년, 2000년, 2001년, 2002년 각 연도별로,국가 전체 유병률 평균값보다 작은 유병률을 기록한 국가를 각각 구하시오.
# 2) 그리고, 이 네 해 모두에서 평균 이하 유병률을 기록한 국가의 개수를 구하시오
#    단, 유병률이 0인 것이 1개라도 포함된 국가는 제외하고 구한다.
import numpy as np
df = df.set_index('year').T
df = df.replace(0, np.nan).dropna()
result = ((df < df.mean()).sum(axis=1) == 4).sum()
# print(result) # 54

# 16-5) 점포별 매출 변화
# 점포(Store_ID)별로 2022년과 2023년의 월별 매출(Monthly_ Sales) 데이터를 이용하여 결측치를 처리하고, 
# 전년 대비 매출 증가율을 분석하고자 한다.
# 다음 절차에 따라 문제를 해결하시오.
df = pd.read_csv(path1 + "store_sales_data_v2.csv")
# print(df.head(3))
# 1) 2022년과, 2023년에 대해 각각 점포별로 월별 매출을 구하고, 매출 결측치는 바로 전월 매출로 대체한다.
#    단, 첫 번째 월(2022년 1월)의 결측치는 2022년 2월의 데이터로 대체하시오.
# 2) 점포별 1년 전 대비 매출 증가율을 계산하시오.
#   > 2023년 매출 증가율 = (2023년총매출 - 2022년총매출)/ 2022년총매출
# 3) 2023년 매출 증가율이 가장 높은 점포의 2022년총매출과 2023년의 총매출의 합을 정수로 출력한다.
d2022 = df[df['Year']==2022].pivot(index='Store_ID', columns='Month', values='Monthly_Sales')
d2023 = df[df['Year']==2023].pivot(index='Store_ID', columns='Month', values='Monthly_Sales')
d2022 = d2022.ffill(axis=1).bfill(axis=1).sum(axis=1)
d2023 = d2023.ffill(axis=1).bfill(axis=1).sum(axis=1)
# print(d2022.isna().sum())
# print(d2023.isna().sum())
increase = (d2023 - d2022) / d2022
store = increase.idxmax()
result = int(d2022[store] + d2023[store])
# print(result) # 803184