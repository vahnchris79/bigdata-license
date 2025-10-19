
# 기출문제 제8회 2024년 6월 22일 시행

import pandas as pd
path = "https://raw.githubusercontent.com/YoungjinBD/data/main/exam/"
pd.set_option('display.width', 120)
pd.set_option('display.max_row', 100)

# 작업형 제3유형
# 문제1
# 어느 회사에서 직원들의 업무 효율성을 높이기 위한 새로운 소프트웨어를 도입하였다.
# 도입 전과 도입 후의 업무 처리 시간을 각각 측정하여 새로운 소프트웨어의 효과를 검증하고자 한다.
# 1) 도입 전과 도입 후의 업무처리 시간의 평균과 표준편차를 구하시오.(소수점 둘째 자리까지 반올림)
# 2) 도입 전후의 업무처리 시간 차이가 유의미한 지 부호 순위 검정을 실시하고, 검정통계량을 계산하시오.
# (소수점 둘째 자리까지 반올림)
# 3) p-value를 바탕으로 유의수준 5%에서 귀무가설의 기각/채택여부를 결정하시오.(p-value는 소수점 둘째 자리까지 반올림)

# 풀이
# 1) 도입 전과 도입 후의 업무처리 시간의 평균과 표준편차를 구하시오.(소수점 둘째 자리까지 반올림)
df = pd.read_csv(path + "8_3_1.csv")
# print(df.head())
before_mean = df['before'].mean()
before_std = df['before'].std()
after_mean = df['after'].mean()
after_std = df['after'].std()
# print(round(before_mean, 2), round(before_std, 2), round(after_mean, 2), round(after_std, 2)) # 8.21 1.71 7.23 1.96

# 2) 도입 전후의 업무처리 시간 차이가 유의미한 지 부호 순위 검정을 실시하고, 검정통계량을 계산하시오.
# (소수점 둘째 자리까지 반올림)
from scipy.stats import wilcoxon
statistic, pvalue = wilcoxon(df['before'], df['after'])
# print(round(statistic, 2), round(pvalue, 2)) # 72.0 0.0

# 3) p-value를 바탕으로 유의수준 5%에서 귀무가설의 기각/채택여부를 결정하시오.(p-value는 소수점 둘째 자리까지 반올림)
# print("채택" if round(pvalue, 2) > 0.05 else "기각") # 기각

# 문제2
# 어느 회사에서 직원들의 생산성에 영향을 미치는 요인이 무엇인지 확인하고자 한다. 100명의 직원들을 대상으로 생산성 점수,
# 근무 시간, 연령, 그리고 경력을 조사하였다.
train = pd.read_csv(path + "8_3_2_train.csv")
test = pd.read_csv(path + "8_3_2_test.csv")
# 1) 훈련데이터를 기준으로 생산성 점수(productivity)를 종속변수로 하고 근무 시간, 연령 그리고 경력을 독립변수로 하는 
# 다중회귀 분석을 수행한 후, 회귀계수가 가장 높은 변수를 구하시오.(다중회귀모형 적합 시 절편 포함)
# 2) 유의수준 5% 하에서 각 독립변수가 생산성에 미치는 영향이 통계적으로 유의미한 지 판단하고, 유의미한 변수 개수를 구하시오.
# (p-value는 소수점 넷째 자리까지 반올림)
# 3) 테스트 데이터로 모델의 성능을 평가하시오 (R^2 산출)

# 풀이
# 1) 훈련데이터를 기준으로 생산성 점수(productivity)를 종속변수로 하고 근무 시간, 연령 그리고 경력을 독립변수로 하는 
# 다중회귀 분석을 수행한 후, 회귀계수가 가장 높은 변수를 구하시오.(다중회귀모형 적합 시 절편 포함)
# print(train.shape, test.shape) # (80, 4) (20, 4)
# print(train.head(), test.head(), sep="\n")
import statsmodels.api as sm

X = train[['hours', 'age', 'experience']]
y = train['productivity']

X = sm.add_constant(X) # 절편 포함

model = sm.OLS(y, X).fit()
print(model.summary())
print(model.params)

# 회귀계수(coef) 확인(가장 높은 변수): hours(근무 시간)
coefficients = model.params[1:]
# print(coefficients.idxmax()) # hours

# 2) 유의수준 5% 하에서 각 독립변수가 생산성에 미치는 영향이 통계적으로 유의미한 지 판단하고, 유의미한 변수 개수를 구하시오.
# (p-value는 소수점 넷째 자리까지 반올림)
pvalue = round(model.pvalues, 4)
# print((pvalue < 0.05)[1:].sum()) # 3

# 3) 테스트 데이터로 모델의 성능을 평가하시오 (R^2 산출)
from sklearn.metrics import r2_score

X_test = test[['hours', 'age', 'experience']]
y_test = test['productivity']

X_test = sm.add_constant(X_test)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
# print(r2) # 0.8036428205191006