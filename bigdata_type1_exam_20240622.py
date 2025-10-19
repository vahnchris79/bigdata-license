
# 기출문제 제8회 2024년 6월 22일 시행

import pandas as pd
import numpy as np
path = "https://raw.githubusercontent.com/YoungjinBD/data/main/exam/"
pd.set_option('display.width', 120)
pd.set_option('display.max_row', 100)

# 작업형 제1유형
# 1. 다음의 데이터는 대륙별 국가의 맥주소비량을 조사한 것이다.
# 1) 평균 맥주소비량이 가장 많은 대륙을 구하시오.
# 2) 이전 문제에서 구한 대륙에서 5번째로 맥주소비량이 많은 나라를 구하시오.
# 3) 이전 문제에서 구한 나라의 평균 맥주소비량을 구하시오.(소수점 첫째 자리에서 반올림)

# 풀이
# 1) 평균 맥주소비량이 가장 많은 대륙을 구하시오.
df = pd.read_csv(path + "8_1_1.csv")
# print(df.head())
# print(df.isna().sum().to_frame().T)
# print(df['대륙'].unique())
result1 = df.groupby(['대륙'])['맥주소비량'].mean().idxmax()
# print(result1) # SA

# 2) 이전 문제에서 구한 대륙에서 5번째로 맥주소비량이 많은 나라를 구하시오.
df2 = df[df['대륙'] == result1].copy()
# result2 = df2.sort_values('맥주소비량', ascending=False)['국가'].values[5]
result2 = df2.groupby('국가')['맥주소비량'].sum().sort_values(ascending=False).index[4]
# print(result2) # Colombia -> Venezuela

# 3) 이전 문제에서 구한 나라의 평균 맥주소비량을 구하시오.(소수점 첫째 자리에서 반올림)
# result3 = df[df['국가'] == result2].groupby(['국가'])['맥주소비량'].mean().reset_index()['맥주소비량'][0]
result3 = df.loc[df['국가'] == result2, '맥주소비량'].mean().round()
# print(round(result3, 3)) # 251.491 -> 253.0

# 2. 다음의 데이터는 국가별로 방문객 유형을 조사한 것이다.
# 1) 관광객비율이 두 번째로 높은 나라의 '관광' 수를 구하시오.
#    관광객비율 = 관광/합계(소수점 넷째자리에서 반올림)
#    합계 = 관광 + 사무 + 공무 + 유학 + 기타
# 2) 관광 수가 두 번째로 높은 나라의 '공무' 수의 평균을 구하시오(소수점 첫째 자리에서 반올림)
# 3) 이전에 구한 관광 수와 공무 수의 합계를 구하시오.

# 풀이
# 1) 관광객비율이 두 번째로 높은 나라의 '관광' 수를 구하시오.
df = pd.read_csv(path + "8_1_2.csv")
# print(df.shape) # (100, 6)
# print(df.head())

# print(df['국가'].value_counts()) # 국가별 중복존재
# 국가별 관광, 사무, 공무, 유학, 기타의 합계 계산
df2 = df.groupby('국가')[['관광','사무','공무','유학','기타']].sum()
# print(df2.shape) # (48, 5)
df2['합계'] = df2.loc[:, '관광':'기타'].sum(axis=1)
# print(df2.head())
df2['관광객비율'] =  np.round(df2['관광'] / df2['합계'], 3)
result1 = df2.sort_values('관광객비율', ascending=False)['관광'][1]
# result1 = df2.sort_values(by='관광객비율', ascending=False).iloc[1, 1] 
# print(result1) # 7831 -> 9039

# 2) 관광 수가 두 번째로 높은 나라의 '공무' 수의 평균을 구하시오(소수점 첫째 자리에서 반올림)
second = df2.sort_values(by='관광', ascending=False).index[1]
result2 = round(df.loc[df['국가'] == second, '공무'].mean(), 0)
print(result2) # 647.3 -> 494.0

# 3) 이전에 구한 관광 수와 공무 수의 합계를 구하시오.
result3 = result1 + result2
# print(result3) # 8478.3 -> 9533.0

# 3. CO(GT), NMHC(GT) 컬럼에 대해 Min-Max 스케일러는 실행하고, 스케일링된 CO(GT), NMHC(GT) 컬럼의 표준편차를 구하시오.
# (소수점 셋째 자리에서 반올림)
from sklearn.preprocessing import MinMaxScaler
df = pd.read_csv(path + "8_1_3.csv")
# print(df.head())

temp = MinMaxScaler().fit_transform(df[['CO(GT)', 'NMHC(GT)']])
df3 = pd.DataFrame(temp, columns=['CO(GT)', 'NMHC(GT)'])
res_std1 = round(df3['CO(GT)'].std(ddof=1), 2)
res_std2 = round(df3['NMHC(GT)'].std(ddof=1), 2)
# print(res_std1, res_std2) # 0.367 0.146 -> 0.37, 0.15
