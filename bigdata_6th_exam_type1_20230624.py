
# 기출문제 제6회
import pandas as pd
pd.set_option('display.max_row', None)
pd.set_option('display.max_column', None)
import warnings
warnings.filterwarnings('ignore')

path = "https://raw.githubusercontent.com/YoungjinBD/data/main/exam/"

# 작업형 제1유형
# 1. 다음 데이터에서 ProductA 가격과 ProductB 가격이 모두 0원이 아닌 데이터를 필터링하고,
#    ProductA와 ProductB의 가격 차이를 정의하시오. 각 도시별 가격 차이의 평균 중 가장 큰 값을
#    구하시오.(소수점 첫째 자리까지 반올림)

# 데이터 확인
df1 = pd.read_csv(path + "6_1_1.csv")
# print(df1.head(), df1.info(), sep="\n")

#  ProductA 가격과 ProductB 가격이 모두 0원이 아닌 데이터를 필터링
df1 = df1[(df1['ProductA가격'] > 0) & (df1['ProductB가격'] > 0)].copy()
# print(df1.info()) # 66

# ProductA와 ProductB의 가격 차이를 정의
df1['가격차이'] = df1['ProductA가격'] - df1['ProductB가격']

# 각 도시별 가격 차이의 평균 중 가장 큰 값
result = round(df1.groupby('도시명')['가격차이'].mean().max(), 1)
# print(result) # 16250.0 -> 16333.3

# 2. 100명의 키와 몸무게를 조사하여 적정 체중인지 판단할 수 있는 BMI를 산출하려 한다. 아래 표를 참고하여
#    BMI를 기준으로 저체중, 정상, 과체중, 비만을 구분하고, 저체중인 사람과 비만인 사람의 총 합을 구하시오.

# 데이터 확인
df2 = pd.read_csv(path + "6_1_2.csv")
# print(df2.head(), df2.info(), sep="\n")

# 키 단위 변경(Height_cm -> Height_m)
df2['Height_m'] = df2['Height_cm'] / 100

# BMI 계산
df2['BMI'] = df2['Weight_kg'] / (df2['Height_m'] * df2['Height_m'])

# BMI 기준 비만정도 구분
df2.loc[df2['BMI'] >= 25, 'Class'] = '비만'
df2.loc[(df2['BMI'] >= 23) & (df2['BMI'] < 25), 'Class'] = '과체중'
df2.loc[(df2['BMI'] >= 18.5) & (df2['BMI'] < 23), 'Class'] = '적정'
df2.loc[df2['BMI'] < 18.5, 'Class'] = '저체중'
# print(df2.head())

# 저체중인 사람과 비만인 사람의 총 합
# print(df2['Class'].value_counts())
result = df2['Class'].value_counts().values[0] + df2['Class'].value_counts().values[1]
# print(result) # 74

# 3. 다음 데이터에서 연도별로 가장 큰 순생산량(생산된 제품 수 - 판매된 제품 수)을 가진 공장을 찾고,
#    순생산량의 합을 계산하시오.

# 데이터 확인
df3 = pd.read_csv(path + "6_1_3.csv")
# print(df3.head(), df3.info(), sep="\n")

# 연도별로 가장 큰 순생산량(생산된 제품 수 - 판매된 제품 수)을 가진 공장
# df3['products_cnt'] = (df3.iloc[:, [1, 2]].sum(axis=1)) - (df3.iloc[:, [2, 3]].sum(axis=1))
df3['products_cnt'] = (df3['products_made_domestic'] + df3['products_made_international']) - (df3['products_sold_domestic'] + df3['products_sold_international'])
# df3['products_cnt2'] = (df3.iloc[:, 1:2].sum(axis=1)) - (df3.iloc[:, 3:4].sum(axis=1))
# print(df3.head(3))
# print(df3.info())
group = df3.groupby(['year','factory'])['products_cnt'].sum()
result = group.sort_values(ascending=False).idxmax()
# print(result) # Factory D

# 해당공장의 순생산량의 합을 계산
result = df3.loc[df3['factory'] == result, 'products_cnt'].sum()
print(result) # 3167
