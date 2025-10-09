
# 빅데이터분석기사 9회(2024.11.30) 기출문제

import pandas as pd
import numpy as np

path = "https://raw.githubusercontent.com/YoungjinBD/data/main/exam/"

# 작업형 제1유형
# 1. 데이터에서 (연도, 성별, 지역코드)별 대출액의 합계를 구하시오.
# 이후, 각 (연도, 지역코드)별로 남성과 여성의 총 대출액 차이의 절댓값을 계산하고, 성별 간
# 총 대출액 차이가 가장 큰 지역코드를 구하시오.
# (총 대출액 = 금액1 + 금액2)
df1 = pd.read_csv(path + "9_1_1.csv", dtype={'year': str, '지역코드': str})
df1['총대출액'] = df1['금액1'] + df1['금액2']
# group = df1.groupby(['year','gender','지역코드'])['총대출액'].sum().reset_index()
group = df1.pivot_table(values='총대출액', columns='gender', 
                        index=['year','지역코드'], aggfunc=np.sum).fillna(0).reset_index()
# print(group.head())
group.columns = ['year', '지역코드', '총대출액_남성', '총대출액_여성']
group['차이'] = sorted(abs(group['총대출액_남성'] - group['총대출액_여성']), reverse=True)
result = group.loc[group['차이'] == group['차이'].max(), '지역코드']
# print(result) 

# 2. 연도별 최대 검거율을 가진 범죄유형을 찾아서 해당연도 및 유형의 검거건수들의 총합을 구하시오.
# (검거율 = 검거건수 / 발생건수)
df2 = pd.read_csv(path + "9_1_2.csv", dtype={'연도': str})
# print(df2.head())
# print(df2.info())
df2_검거 = df2.loc[df2['구분'] == '검거건수'].copy()
df2_발생 = df2.loc[df2['구분'] == '발생건수'].copy()
df2_검거_melted = df2_검거.melt(id_vars='연도', 
                                value_vars=[col for col in df2_검거.columns if '범죄유형' in col],
                                var_name='범죄유형', value_name='검거건수')
df2_발생_melted = df2_발생.melt(id_vars='연도', 
                                value_vars=[col for col in df2_검거.columns if '범죄유형' in col],
                                var_name='범죄유형', value_name='발생건수')
df2_all = df2_검거_melted.merge(df2_발생_melted, how="left", on=['연도', '범죄유형'])
df2_all['검거율'] = sorted(df2_all['검거건수'] / df2_all['발생건수'], reverse=True)
result = df2_all.loc[df2_all['검거율'] == df2_all['검거율'].max(), '검거건수'].sum()
# print(result)

# 3. 제시된 문제를 순서대로 풀고, 해답을 제시하시오.
df3 = pd.read_csv(path + "9_1_3.csv")
# print(df3)
# print(df3.info())
# (1) 평균만족도: 결측지는 평균만족도 컬럼의 전체 평균으로 채우시오.
df3['평균만족도'] = df3['평균만족도'].fillna('mean')
# print(df3.isna().sum())

# (2) 근속연수 : 결측치는 각 부서와 등급별 평균 근속연수로 채우시오(평균값의 소수점은 버림)
mean_work = round(df3.groupby(['부서','등급'])['근속연수'].transform('mean'),0)
df3['근속연수'] = df3['근속연수'].fillna(mean_work)
# print(df3.isna().sum())

# (3) A: 부서가 'HR'이고 등급이 'A'인 사람들의 평균 근속연수를 계산하시오
A = df3.loc[(df3['부서'] == 'HR') & (df3['등급'] == 'A'), '근속연수'].mean()
# print(A)

# (4) B: 부서가 'Sales'이고 등급이 'B'인 사람들의 평균 교육참가횟수를 계산하시오
B = df3.loc[(df3['부서'] == 'Sales') & (df3['등급'] == 'B'), '교육참가횟수'].mean()
# print(B)
# (5) A와 B를 더한 값을 구하시오.
result = A + B
# print(result)