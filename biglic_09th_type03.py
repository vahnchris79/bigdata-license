
# 작업형 제3유형
# 문제 1.
# 한 제조 회사에서 생산성을 높이고자 직원들의 주요 생산성 요인을 분석하기로 결정하였다.
# 이를 위해 200명의 직원 데이터를 수집했으며, 직원들의 근무 기간, 특성정보, 그리고 개인적인 속성을 조사하였다.
# Data Description
# id: 데이터의 고유 식별자, tenure: 사용기간, 
# f2: 고객의 두 번째 특성, f3: 고객의 세 번째 특성, f4: 고객의 네 번째 특성, f5: 고객의 다섯 번째 특성
# design: 생산성 점수

# (1) design을 예측하는 다중회귀분석을 시행한 후 유의하지 않은 설명변수 개수를 구하시오.
# (단, 불필요한 컬럼은 제외하며, 모델의 절편항은 포함)
import pandas as pd
path = "https://raw.githubusercontent.com/YoungjinBD/data/main/exam/"
df1 = pd.read_csv(path + "9_3_1.csv")
# print(df1.head())
# print(df1.info())
# print(df1.columns) # ['id', 'tenure', 'f2', 'f3', 'f4', 'f5', 'design']

import statsmodels.api as sm

x = df1.drop(columns=['design'], axis=1)
y = df1['design']

# 상수항 추가
x = sm.add_constant(x)

# 회귀모형 적합
model = sm.OLS(y, x).fit()
# print(model.summary())

#유의하지 않은 변수의 개수 <- p-value > 0.05
pvalues = model.pvalues
not_sig_vars = pvalues[pvalues > 0.05].drop('const')
result = len(not_sig_vars)
# print(result)

# (2) 훈련데이터의 예측값과 실제값의 피어슨 상관계수를 구하시오.
# (소수점 셋째 자리에서 반올림)

# 예측값
y_pred = model.predict(x)

# 피어슨 상관계수
from scipy.stats import pearsonr
corr, p_value = pearsonr(y, y_pred)
# print(round(corr, 3)) # 0.925
# print(round(p_value, 3)) # 0.000

# (3) 적합한 모델을 활용하여 테스트 데이터에서의 RMSE를 구하시오.
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error as RMSE

temp = train_test_split(x, y, test_size=0.3, random_state=42)
x_train, x_test, y_train, y_test = temp

x_train_const = sm.add_constant(x_train)
x_test_const = sm.add_constant(x_test)

model = sm.OLS(y_train, x_train_const).fit()
y_pred = model.predict(x_test_const)

result = RMSE(y_test, y_pred)
# print(round(result, 3)) # 4.734

# 문제2
# 한 통신회사에서는 고객이탈을 줄이고자 주요 요인들은 분석하기로 결정하였다.
# 이를 위해 500명의 고객 데이터를 수집했으며, 고객의 서비스 이용 및 가입 정보,
# 그리고 일부 개인적인 속성을 조사하였다.
# Data description
# col1: 고객의 첫 번째 특성, col2 = 고객의 두 번째 특성
# Phone_Service: 폰서비스 가입 여부
# Tech_insurance: 기술보험 가입 여부
# churn: 이탈 여부
df3 = pd.read_csv(path + "9_3_2.csv")
# (1) 고객이탈을 예측하는 로지스틱 회귀를 시행한 후 col1 컬럼의 p-value를 구하시오.
# print(df3.head())
# print(df3.shape) # (500, 5)
# print(df3['churn'].unique())
from statsmodels.api import GLM, add_constant, families
X_all = df3[['col1', 'col2', 'Phone_Service', 'Tech_Insurance']]
Y_all = df3['churn']
X_all = add_constant(X_all)

x_train = X_all.iloc[:350, :]
x_test = X_all.iloc[350:, :]
y_train = Y_all.iloc[:350]
y_test = Y_all.iloc[350:]
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (350, 5) (150, 5) (350,) (150,)
model = GLM(y_train, x_train, family=families.Binomial()).fit()
# print(round(model.llf, 3))
# print(model.summary())
result = round(model.pvalues[1], 4)
# print(result) # 0.001

# (2) 폰 서비스를 받지 않는 고객 대비 받는 고객의 이탈 확률 오즈비를 구하시오
# print(df3['Phone_Service'].value_counts()) # 1: 307, 2: 193
# print(df3['churn'].value_counts()) # 1: 251, 2: 249i
import statsmodels.api as sm
import numpy as np

y = df3['churn']
X = df3[['Phone_Service']]
X = sm.add_constant(X)
model = sm.Logit(y, X).fit()
# print(model.summary())
coef = model.params
odds_ratio = np.exp(coef)
# print(odds_ratio) # 1.776

# (3) 이탈할 확률이 0.3이상인 고객 수를 구하시오
df3['pred_prob'] = model.predict(X)
count_high = sum(df3['pred_prob'] >= 0.3)
# print(count_high)