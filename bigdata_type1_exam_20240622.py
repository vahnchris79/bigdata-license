
# 기출문제 제8회 2024년 6월 22일 시행

import pandas as pd
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
result2 = df2.sort_values('맥주소비량', ascending=False)['국가'].values[5]
# print(result2) # Colombia

# 3) 이전 문제에서 구한 나라의 평균 맥주소비량을 구하시오.(소수점 첫째 자리에서 반올림)
result3 = df[df['국가'] == result2].groupby(['국가'])['맥주소비량'].mean().reset_index()['맥주소비량'][0]
# print(round(result3, 3)) # 251.491

# 2. 다음의 데이터는 국가별로 방문객 유형을 조사한 것이다.
# 1) 관광객비율이 두 번째로 높은 나라의 '관광' 수를 구하시오.
#    관광객비율 = 관광/합계(소수점 넷째자리에서 반올림)
#    합계 = 관광 + 사무 + 공무 + 유학 + 기타
# 2) 관광 수가 두 번째로 높은 나라의 '공무' 수의 평균을 구하시오(소수점 첫째 자리에서 반올림)
# 3) 이전에 구한 관광 수와 공무 수의 합계를 구하시오.

# 풀이
# 1) 관광객비율이 두 번째로 높은 나라의 '관광' 수를 구하시오.
df = pd.read_csv(path + "8_1_2.csv")
# print(df.head())
df['합계'] = df['관광'] + df['사무'] + df['공무'] + df['유학'] + df['기타']
df['관광객비율'] = round(df['관광'] / df['합계'], 4)
country = df.loc[df['관광객비율'] == sorted(df['관광객비율'], reverse=True)[1], '국가'].values[0]
result1 = df.loc[df['국가'] == country, '관광'].values[1]
# print(result1) # 7831

# 2) 관광 수가 두 번째로 높은 나라의 '공무' 수의 평균을 구하시오(소수점 첫째 자리에서 반올림)
result2 = round(df.loc[df['국가'] == '이탈리아', '공무'].mean(), 1)
# print(result2) # 647.3

# 3) 이전에 구한 관광 수와 공무 수의 합계를 구하시오.
result3 = result1 + result2
# print(result3) # 8478.3

# 3. CO(GT), NMHC(GT) 컬럼에 대해 Min-Max 스케일러는 실행하고, 스케일링된 CO(GT), NMHC(GT) 컬럼의 표준편차를 구하시오.
# (소수점 셋째 자리에서 반올림)
from sklearn.preprocessing import MinMaxScaler
df = pd.read_csv(path + "8_1_3.csv")
# print(df.head())

temp = MinMaxScaler().fit_transform(df[['CO(GT)', 'NMHC(GT)']])
df3 = pd.DataFrame(temp, columns=['CO(GT)', 'NMHC(GT)'])
res_std1 = round(df3['CO(GT)'].std(ddof=1), 3)
res_std2 = round(df3['NMHC(GT)'].std(ddof=1), 3)
# print(res_std1, res_std2) # 0.367 0.146
