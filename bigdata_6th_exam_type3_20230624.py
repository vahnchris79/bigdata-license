
# 기출문제 제6회
import pandas as pd
pd.set_option('display.max_row', None)
pd.set_option('display.max_column', None)
import warnings
warnings.filterwarnings('ignore')

path = "https://raw.githubusercontent.com/YoungjinBD/data/main/exam/"

# 작업형 제3유형
# 문제1
# 어느 회시에서 100명의 직원들을 대상으로 하루 업무 수행 시간을 조사하였다. K-S검정을 통해 업무 수행 시간이
# 정규분포를 따르는지 검정하고자 한다.
df1 = pd.read_csv(path + "6_3_1.csv")
# print(df1.head(3))

# 1. 직원들의 업무 수행 시간의 평균과 표준편차를 구하시오.(소수점 셋째 자리까지 반올림)
mean_work_hours = df1['work_hours'].mean()
std_work_hours = df1['work_hours'].std()
# print(f"평균: {mean_work_hours:.3f}, 표준편차: {std_work_hours:.3f}")
# 평균: 8.090, 표준편차: 1.519

# 2. 직원들의 업무 수행 시간이 정규분포를 따르는 지 K-S검정을 실시하고 검정통계량을 계산하시오.
# (소수점 셋째 자리까지 반올림)
from scipy.stats import kstest
# wH = df1['work_hours'].to_numpy()
statistic, pvalue = kstest(df1['work_hours'], "norm", args=(mean_work_hours, std_work_hours))
# print(f"검정통계량: {statistic:.3f}, p-value: {pvalue.std():.3f}")
# 검정통계량: 1.000, p-value: 0.000

# 3. p-value를 바탕으로 유의수준 5%에서 귀무가설의 기각/채택여부를 결정하시오(p-value는 소수점 셋째 자리까지 반올림)
# print('기각' if pvalue < 0.05 else "채택") 
# # 기각

# 문제2
# 다음의 데이터는 주택들의 가격(price), 면적(area), 방의 개수(rooms), 연식(age)을 조사하여 기록한 것이다.
df2 = pd.read_csv(path + "6_3_2.csv")
# print(df2.head(3), df2.shape, sep="\n") # (100, 4)

# 1. 주택 가격을 종속변수로 하고 면적, 방의 개수, 연식을 독립변수로 하는 다중회귀 분석을 수행하여 회귀계수가 가장 높은 변수를
#    구하시오.(다중회귀모형 적합 시 절편 포함)

import statsmodels.api as sm
from statsmodels.api import OLS, add_constant
from sklearn.model_selection import train_test_split

x = df2.drop(columns=['age'])
y = df2['age']
x = sm.add_constant(x)

# temp = train_test_split(x, y, test_size=0.3, random_state=42)
# x_train, x_test, y_train, y_test = temp
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (70, 4) (30, 4) (70,) (30,)

model = sm.OLS(y, x).fit()
# print(model.summary())
# print(model.params[1:].abs().idxmax()) # rooms

# 2. 유의수준 5% 하에서 각 독립 변수가 주택가격에 미치는 영향을 통계적으로 유의미한 지 판단하고, 유의미한 변수개수를 구하시오.
print(f"{model.params[1]}: 기각" if abs(model.pvalues[1]) < 0.05 else f"{model.params[1]}: 채택")
print(f"{model.params[2]}: 기각" if abs(model.pvalues[2]) < 0.05 else f"{model.params[2]}: 채택")
print(f"{model.params[3]}: 기각" if abs(model.pvalues[3]) < 0.05 else f"{model.params[3]}: 채택")
# 유의미한 변수개수: 0