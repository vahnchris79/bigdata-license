
# 작업형 제1유형
import pandas as pd
pd.set_option('display.width', 150)
path1 = "https://raw.githubusercontent.com/Soyoung-Yoon/bigdata/main/"
path2 = "https://raw.githubusercontent.com/Soyoung-Yoon/data_01/main/"

# 2회-1번) 보스턴 주택 가격
# 보스턴 주택 가격 데이터셋(Boston)을 사용해 극단적인 값이 분석에 미치는 영향을 줄이기 위한 전처리 과정을 수행한 뒤, 
# 특정 조건에 해당하는 평균 범죄율을 확인하려 한다.
# 다음 절차에 따라 문제를 해결하시오.
df = pd.read_csv(path1 + "boston.csv")
# print(df.head())
# 1) 범죄율(CRIM)컬럼에서 TOP10(즉, 상위 10위)을 찾는다.
s = df['CRIM'].sort_values(ascending=False).head(10)
# print(s)
# 2) 이 중 10번째 값으로, 상위 10개의 값을 모두 대체한다.
# print(s.iloc[-1])
df.loc[s.index, 'CRIM'] = s.iloc[-1]
# print(df)
# 3) 값을 변경한 데이터에서 AGE 변수가 80이상인 것을 대상으로, 범죄율의 평균을 산출한다.
result = df[df['AGE'] >= 80]['CRIM'].mean()
# 4) 산출된 평균을 반올림하여 소수점 아래 4자리까지 출력한다.
# print(round(result, 4))

# 2회-2번) 캘리포니아 주택 정보 - 결측값
# 캘리포니아 주택 정보 데이터셋(Housing)의 앞부분만 사용하여 결측값 처리 전후의 변화를 비교해보는 연습을 진행한다.
# 다음 절차에 따라 문제를 해결하시오.
df = pd.read_csv(path1 + "housing.csv")
# print(df.head(3))
# 1) 전체 데이터의 첫 번째 행부터 순서대로 80%에 해당하는 데이터만 추출하여 사용한다.
rows = int(len(df) * 0.8)
df = df.loc[:rows, :]
# 2) 추출된 데이터에서 total_bedrooms 변수에 결측값이 있다면, 해당 결측값을 total_bedrooms 변수의 중앙값으로 대체한다.
# print(df.isna().sum())
df2 = df.copy()
df2['total_bedrooms'] = df2['total_bedrooms'].fillna(df2['total_bedrooms'].median())
# print(df.isna().sum())
# 3) 결측값을 대체하기 전과 후의 total_bedrooms 변수의 표준편차를 각각 계산한다.
before_std = df['total_bedrooms'].std(ddof=1)
after_std = df2['total_bedrooms'].std(ddof=1)
# 4) 마지막으로, 대체 전 표준편차 - 대체 후 표준편차의 값을 구하고, 반올림하여 소수점 아래 3자리까지 출력한다.
# print(round(before_std - after_std, 3)) # 1.975

# 2회 3번) 캘리포니아 주택 정보 - 이상값
# 캘리포니아 주택 정보 데이터셋(Housing)을 사용하여 latitude 컬럼의 이상값을 찾아 그 합을 계산한다.
# 다음 절차에 따라 문제를 해결하시오.
df = pd.read_csv(path1 + "housing.csv")
# - 1) 'latitude' 컬럼의 이상값를 찾아, 이상치들의 합을 산출하시오.
# - 2) 이상치 기준은 다음과 같다.
# > 평균 - (표준편차 * 1.5), 평균 + (표준편차 * 1.5)
# - 3) 계산 결과를 반올림하여 정수형으로 출력한다.
lower = df['latitude'].mean() - (df['latitude'].std() * 1.5)
upper = df['latitude'].mean() + (df['latitude'].std() * 1.5)
# print(round(df[(df['latitude'] < lower) | (df['latitude'] > upper)]['latitude'].sum())) # 45816

# 10-4) 최다 발생/검거 범죄유형의 빈도 분석**
# 연도별로 가장 많이 발생하거나 검거된 범죄유형을 조사하여, 자주 1위를 기록한 유형은 우선적으로 대응하려합니다.
# 다음 절차에 따라 문제를 해결하시오.
df = pd.read_csv(path2 + "crime_data02.csv")
# print(df.head(3))
# - 1) 연도별로 '발생건수'에 대해 가장 많이 발생하는 범죄유형을 찾고, 가장 많이 1위를 차지한 범죄유형의 번호를 A라고한다.
# - 2) 연도별로 '검거건수'에 대해 가장 많이 발생하는 범죄유형을 찾고, 가장 많이 1위를 차지한 범죄유형의 번호를 B라고한다.
# - 3) A, B의 범죄유형 번호를 정수로 사용하여 두 번호의 합을 구해 출력한다.
A = df[df['구분']=="발생건수"].drop(columns=['구분']).set_index('연도').idxmax(axis=1)
A = A.value_counts().idxmax()[4:]
B = df[df['구분']=="검거건수"].drop(columns=['구분']).set_index('연도').idxmax(axis=1)
B = B.value_counts().idxmax()[4:]
# print(int(A) + int(B))

# 10-5) 연도별 발생 대비 검거 차이 분석
# 범죄 발생과 검거 간의 차이는 치안의 효율성을 나타낸다. 어느 해에 그 차이가 가장 컸는지 확인해 본다.
# 다음 절차에 따라 문제를 해결하시오.
df = pd.read_csv(path2 + "crime_data02.csv")
# 1) 연도별로 '발생건수' 합계와 '검거건수' 합계를 구한 뒤, 그 차이를 계산한다.
#    > 차이 = '발생건수' 합계 - '검거건수' 합계
# 2) 차이가 가장 큰 연도의 연도값을 정수로 출력한다.
df = df.set_index('연도')
A = df[df['구분']=="발생건수"].drop(columns=['구분']).sum(axis=1)
B = df[df['구분']=="검거건수"].drop(columns=['구분']).sum(axis=1)
# print((A-B).idxmax())

df2 = pd.read_csv(path2 + "crime_data02.csv")
df2 = df2.set_index(['연도', '구분'])
df3 = df2.sum(axis=1).unstack()
df3['차이'] = df3['발생건수'] - df3['검거건수']
print(df3['차이'].idxmax())