
# 공통: 데이터 경로
import pandas as pd
path1 = "https://raw.githubusercontent.com/Soyoung-Yoon/bigdata/main/"
path2 = "https://raw.githubusercontent.com/Soyoung-Yoon/data_01/main/"

# 작업형 제1유형
# 7회-1번) 결측치 처리 및 과목별 점수 표준화
# 학생들의 수업 과목별 점수 데이터를 변환하고자 한다.
# 다음 절차에 따라 문제를 해결하시오.
df = pd.read_csv(path1 + "student_scores01.csv")
# print(df.head(3))
# 1) 'score' 컬럼에서 결측치가 발생한 행을 제거한 뒤, 가장 많은 학생이 수강한 과목(subject)을 찾으시오.
# print(df.info())
df = df.dropna(subset=['score'])
subject = df['subject'].value_counts().idxmax()
# 2) 해당 과목의 점수(score)를 표준화(standardization)하여, 표준화된 점수 중 가장 높은 값을 구하시오.
#    > 표준화 공식: (값 - 평균) / 표준편차
# df = df[df['subject']==subject].copy()
s = df.loc[df['subject']==subject, 'score']
s = (s - s.mean()) / s.std(ddof=1)
result = round(s.max(), 3)
# 3) 구한 값을 반올림하여 소수점 아래 3자리까지 출력한다.
# print(round(result, 3)) # 1.713

# 7회-2번) 종속변수와의 상관관계 분석
# 주어진 데이터에서 종속변수와 가장 높은 상관관계를 갖는 독립변수를 찾고자 한다.
# 다음 절차에 따라 문제를 해결하시오.
df = pd.read_csv(path1 + "close_features.csv")
# print(df.head(3))
# - 1) 해당 데이터에는 77개의 독립변수와 1개의 종속변수('CLOSE')가 존재한다.
# - 2) 종속변수 'CLOSE'와 가장 높은 상관관계를 갖는 독립변수를 찾는다.
# - 3) 찾은 독립변수의 평균값을 구하고, 반올림하여 소수점 아래 3자리까지 출력한다.
max_feature = df.corr()['CLOSE'].drop('CLOSE').abs().idxmax()
result = round(df[max_feature].mean(), 3)
# print(result) # 4.807

# **7회-3번) 이상치 개수 구하기**
# 주어진 데이터에서 특정 기준에 따라 이상치 개수를 구하고자 한다.
# 다음 절차에 따라 문제를 해결하시오.
df = pd.read_csv(path1 + "outlier_data.csv")
# print(df.head(3))
# 1) 'feature_2' 컬럼의 이상치 개수를 구하여 정수로 출력한다.
# 2) 이상치 기준은 다음과 같다.
#    > 이상치 < 평균 - IQR * 1.5, 이상치 > 평균 + IQR * 1.5
s = df['feature_2']
# print(len(s)) # 2000
mean = s.mean()
Q1, Q3 = s.quantile([0.25, 0.75])
IQR = Q3 - Q1
lower = mean - (IQR * 1.5)
upper = mean + (IQR *  1.5)
# print(len(s[(s.values < lower) | (s.values > upper)])) # 224

# 15-4) 이상치가 가장 많은 독립변수 찾기 및 이상치 개수 구하기
# 주어진 데이터는 아파트 가격 관련 데이터를 포함하고 있다. 아파트 가격(Price)은 종속변수이며,
# 나머지 변수들은 독립변수이다.
# 다음 절차에 따라 문제를 해결하시오.
df = pd.read_csv(path1 + "apartment_prices.csv")
# print(df.head(3))
# 1) 아파트 가격을 결정하는 독립변수들을 대상으로 분석을 수행한다.
# 2) 이상치 기준은 다음과 같다.
#    > 이상치 < 평균 - IQR * 1.5, 이상치 > 평균 + IQR * 1.5
# 3) 모든 독립변수에 대해 이상치 개수를 구한다.
# 4) 이상치 개수가 가장 많은 독립변수를 찾아, 이상치 개수를 정수로 출력합니다.
res = []
for col in df.columns[:-1]:
    s = df[col]
    mean = s.mean()
    Q1, Q3 = s.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower = mean - (IQR * 1.5)
    upper = mean + (IQR * 1.5)
    cnt = len(s[(s.values < lower) | (s.values > upper)])
    res.append({'col': col, 'cnt': cnt})
result = pd.DataFrame(res).sort_values('cnt', ascending=False)['cnt'].max()
# print(result) # 16

# 15-5) 상관관계 분석
# 주어진 데이터에서 독립변수들 간의 상관관계를 분석하고자 한다.
# 다음 절차에 따라 문제를 해결하시오.
df = pd.read_csv(path1 + "penguins01.csv")
# print(df.head(3))
# 1) 결측치가 포함된 모든 행을 제거합니다.
df = df.dropna()
# 2) 1)의 결과를 사용해 모든 숫자형 변수 간 상관계수를 'spearman'으로 계산하시오.
# print(df.info())
corr = df.corr(method='spearman', numeric_only=True)
# 3) 숫자형 변수들 중 상관관계가 가장 큰 두 변수를 찾으시오.
#    단, 같은 변수 간의 상관계수(자기 자신과의 상관계수)는 제외한다.
corr = corr.replace(1, 0)
result = corr.abs().max().max()
# 4) 해당 두 변수의 상관계수 값을 반올림하여 소수점 아래 2자리까지 출력한다.
# print(round(result, 2)) # 0.84


# 15-6) 상관관계 분석 2
# 주어진 데이터에서 독립변수들 간의 상관관계를 분석하고자 한다.
# 다음 절차에 따라 문제를 해결하시오.
df = pd.read_csv(path1 + "mpg.csv")
# print(df.head(3))
# 1) 모든 "숫자형 변수" 간 상관계수(Pearson correlation coefficient)를 계산하시오.
#   단, 같은 변수 간의 상관계수(자기 자신과의 상관계수)는 제외한다.
corr = df.corr(numeric_only=True)
# print(corr)
corr = corr.replace(1, 0)
# 2) 상관계수의 절댓값이 0.8 이상인 변수쌍의 개수를 구하여 정수로 출력하시오.
# print(int((corr.abs() >= 0.8).sum().sum() / 2)) # 8