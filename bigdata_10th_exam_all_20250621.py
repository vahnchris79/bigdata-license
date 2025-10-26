
# 빅데이터분석기사 10회(2025. 06. 21) 기출문제

path = "https://raw.githubusercontent.com/YoungjinBD/data/main/exam/"
# 작업형 제1유형
# 1. 소주제별로 정답률(정답여부가 1인 응답수/해당 소주제 전체 응답수)를 구하고 
# 3번째로 높은 정답률을 구하시오.
# Data description
# 학생 ID : 학생 고유 번호
# 문제 ID : 문제 고유 번호
# 대주제  : 문제 대분류
# 소주제  : 문제 소분류
# 정답여부: 1=정답, 0=오답
# 정답률은 내림차순으로 정렬하였을 때, 동일한 정답률은 하나의 순위로 간주
# (공동 1등이 2명이 있으면 그 다음 순위는 2등으로 간주)
import pandas as pd
import numpy as np
df1 = pd.read_csv(path + "10_1_1.csv")

# 데이터 확인
# print(df1.head(3))
# print(df1.info())

# 소주제별 정답여부의 합과 수량으로 정답률을 구함
nums = df1.groupby('소주제')['정답여부'].sum()
cnts = df1.groupby('소주제')['정답여부'].count()
ratio = np.round((nums / cnts), 2)
uni_ratio = sorted(ratio.unique(), reverse=True)
# print(uni_ratio)

# 2. 제시된 문제를 순서대로 풀고, 해답을 제시하시오.

# 데이터 확인
df2 = pd.read_csv(path + "10_1_2.csv")
# print(df2.head(3))
# print(df2.info())

# 1) date를 연도, 월로 분리하여 연도-월별 price의 합계를 구하시오.
# 그 중 두번 째로 큰 매출액(합계)을 구하시오
df2['date'] = pd.to_datetime(df2['date'])
df2['year'] = df2['date'].dt.year
df2['month'] = df2['date'].dt.month
result = df2.groupby(['year', 'month'])['price'].sum().reset_index().sort_values('price', ascending=False)
# print(result.head(10))
# print(result['price'].values[1])

# 2) 이전 문제에서 네 번째로 큰 price합계에 해당하는 연도-월을 찾으시오.
# 해당 연도-월에서 카테고리별 price합계를 구하시오
# 그 중 가장 높은 price 합계(정수)를 제출하시오
year = int(result.iloc[3]['year'])
month = int(result.iloc[3]['month'])
# print(year, month)
filtered = df2.loc[(df2['year'] == year) & (df2['month'] == month)]
grouped = filtered.groupby(['year','month','category'])['price'].sum().reset_index().sort_values('price', ascending=False)
result = grouped['price'][2]
# print(int(result))

# 3. 제시된 문제를 순서대로 풀고, 해답을 제시하시오.
# Data description
# label: 'spam' 또는 'ham'
# message: 영어문장(특수문자/숫자 등 포함)
df3 = pd.read_csv(path + "10_1_3.csv")
# 1) 각 메시지의 단어 수를 공백(" ")을 기준으로 세는 새로운 컬럼을 만드시오
df3['cnt'] = [len(df3.loc[i, 'message'].split(" ")) for i in df3.index]
# print(df3.head(5))

# 2) 'spam'과 'ham' 각각의 평균 단어 수를 계산하시오
ham_mean = df3.groupby('label')['cnt'].mean().values[0]
spam_mean = df3.groupby('label')['cnt'].mean().values[1]
# print(ham_mean, spam_mean)

# 3) 두 평균 차이의 절댓값을 소수점 셋째자리까지 반올림하여 제출하시오.
result = round(abs(spam_mean - ham_mean), 3)
# print(result)

# 작업형 제2유형
# 제공된 학습용 데이터는 여러 상권 내 건물의 특성(상권유형, 건물면적, 건물연식, 세대수 등)과
# 연간 총 가스 소비량 정보를 담고 있다.
# 학습용 데이터를 활용하여 건물의 연간 총 가스 소비량(gas_totl)을 예측하는 모델을 개발하고,
# 이 중 가장 우수한 모델을 평가용 데이터에 적용하여 예측 결과를 제출하시오.
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error as RMSE

pd.set_option('display.width', 100)

# 평가지표 사용자 함수
def get_scores(model, x_train, x_test, y_train, y_test):
    A = model.score(x_train, y_train)
    B = model.score(x_test, y_test)
    y_pred1 = model.predict(x_train)
    y_pred2 = model.predict(x_test)
    C = RMSE(y_train, y_pred1)
    D = RMSE(y_test, y_pred2)
    return f"r2: {A:.4f}, {B:.4f}, RMSE: {C:.4f}, {D:.4f}"

# 데이터 불러오기(시험장에서는 제공됨)
train = pd.read_csv(path + "10_2_train.csv")
test = pd.read_csv(path + "10_2_test.csv")
# print(train.head())
# print(test.head())
# print(train.shape, test.shape) # (160, 5) (40, 5)

# 데이터 전처리
X = train.drop(columns=['gas_totl'])
Y = train['gas_totl']

X_all = pd.concat([X, test]).drop(columns=['gas_totl'])
# print(X_all.info())
# print(X_all.shape) # (200, 4)

# biz_typy 컬럼에 대한 LabelEncoder 실시
X_all['biz_type'] = LabelEncoder().fit_transform(X_all['biz_type'])
# print(X_all.head())

# 전체 컬럼에 스케일링을 적용
temp = StandardScaler().fit_transform(X_all)
X_all = pd.DataFrame(temp, columns=X_all.columns)
# print(X_all.head())

X = X_all.iloc[:160]
test = X_all.iloc[160:]
# print(X.shape, Y.shape, test.shape) # (160, 4) (160,) (40, 4)

# 데이터 분할
temp = train_test_split(X, Y, test_size=0.3, random_state=42)
x_train, x_test, y_train, y_test = temp
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (112, 4) (48, 4) (112,) (48,)

# 모델생성
# LinearRegression
model1 = LinearRegression().fit(x_train, y_train)
# print(get_scores(model1, x_train, x_test, y_train, y_test)) # r2: 0.6748, 0.6903, RMSE: 289.0117, 308.9916

# DecisionTreeRegressor
# model2 = DecisionTreeRegressor(random_state=0).fit(x_train, y_train)
# print(get_scores(model2, x_train, x_test, y_train, y_test)) # r2: 1.0000, 0.6463, RMSE: 0.0000, 330.2289
# print(model2.get_depth()) # 13
# for d in range(3, 13):
#     model2 = DecisionTreeRegressor(max_depth=d, random_state=1234).fit(x_train, y_train)
#     print(d, get_scores(model2, x_train, x_test, y_train, y_test))
model2 = DecisionTreeRegressor(max_depth=3, random_state=42).fit(x_train, y_train)
# print(get_scores(model2, x_train, x_test, y_train, y_test)) # r2: 0.7486, 0.6827, RMSE: 254.1105, 312.7822

# model3 = RandomForestRegressor(random_state=1234).fit(x_train, y_train)
# print(get_scores(model3, x_train, x_test, y_train, y_test)) # r2: 0.9421, 0.6265, RMSE: 121.9645, 339.3329
# for d in range(6, 13):
#     model3 = RandomForestRegressor(max_depth=d, random_state=1234).fit(x_train, y_train)
#     print(d, get_scores(model3, x_train, x_test, y_train, y_test)) # 8 r2: 0.9409, 0.6271, RMSE: 123.1892, 339.0576
model3 = RandomForestRegressor(max_depth=8, random_state=1234).fit(x_train, y_train)
# print(get_scores(model3, x_train, x_test, y_train, y_test)) # r2: 0.9409, 0.6271, RMSE: 123.1892, 339.0576

model4 = AdaBoostRegressor(random_state=0).fit(x_train, y_train)
# print(get_scores(model4, x_train, x_test, y_train, y_test)) # r2: 0.7780, 0.5982, RMSE: 238.7908, 351.9557

model5 = GradientBoostingRegressor(random_state=0).fit(x_train, y_train)
# print(get_scores(model5, x_train, x_test, y_train, y_test)) # r2: 0.9838, 0.5379, RMSE: 64.4557, 377.4400

final_model = model1
y_pred = final_model.predict(test)
pd.DataFrame({'pred': y_pred}).to_csv('type2_result.csv', index=False)
temp = pd.read_csv('type2_result.csv')
# print(temp['pred'].describe())
# print(test.describe())
# print(Y.describe())

# 작업형 제3유형
# 1. 한 기업이 인사관리 데이터를 가지고 이직여부 예측모델을 개발하려고 한다. 다음 문제를 풀이하시오.
# Data description
# attrition: 이직여부(0=잔류, 1=이직)
# age: 나이
# income: 연봉
# overtime: 야근상태(0=해당없음, 1=보통, 2=상시 등)
# 1) 이직여부를 예측하는 로지스틱 회귀모형을 적합하고, 유의한 변수(유의확률 0.05 미만)의 회귀계수를 
# 소수점 셋째 자리까지 반올림하여 제출하시오.(단, 절편 제외)
import statsmodels.formula.api as smf

df31 = pd.read_csv(path + "10_3_1.csv")
# print(df31.head())
formula = "attrition ~ age + income + C(overtime)"
model = smf.logit(formula, data=df31).fit()
# print(model.summary())
pvalues = model.pvalues[1:]
params = model.params[1:]
# print(pvalues)
# 유의확률 0.05 미만, 절편 제외
sig_vars = params[pvalues < 0.05]
rounded_coefs = np.round(sig_vars, 3)
# print(rounded_coefs) # incomes -0.05

# 2) age가 1 증가할 때 이직(또는 잔류) 오즈비(odds ratio)를 소수점 셋째자리까지 반올림하여 제출하시오
age_coef = model.params["age"]
odds_ratio_age = round(np.exp(age_coef), 3)
# print(odds_ratio_age) # 0.894

# 3) age=20, income=3000, overtime=2값을 가진 데이터의 이직확률을 모델로 예측하여 소수점 셋째자리까지 반올림하여 제출하시오/
data = pd.DataFrame({'age': [20], 'income': [3000], 'overtime': [2]})
pred_prob = round(model.predict(data), 3)
# print(pred_prob) # 0.480

# 2. 어느 지역의 주택들의 정보를 수집하여 주택 가격을 예측하는 모델을 개발하려고 한다.
# 다음 문제를 풀이하시오.
# Data description
# price: 주택 가격, area: 주택 면적, height: 집 높이, wall: 벽 유무(0=없음, 1=있음)
df32 = pd.read_csv(path + "10_3_2.csv")
# print(df32.head())
# 1) 주택 가격을 예측하는 다중선형회귀모형을 적합하고, 유의한 변수(유의확률 0.05 미만)의 회귀계수 합
#  (절편 제외)을 소수점 셋제 자리까지 반올림하여 제출하시오
formula = "price ~ area + height + wall"
model = smf.ols(formula, data=df32).fit()
# print(model.summary())
params = model.params[1:]
pvalues = model.pvalues[1:]
# 유의한 변수(유의확률 0.05 미만)
sig_vars = params[pvalues < 0.05]
coef_sum = round(sig_vars.sum(), 3)
# print(coef_sum) # 10.289

# 2) 유의한 변수만으로 다중선형회귀모형을 다시 적합하고, 결정계수를 소수점 셋째 자리까지 반올림하여 제출하시오/
sig_vars_names = sig_vars.index.tolist()
formula2 = "price ~ " + "+".join(sig_vars_names)
model2 = smf.ols(formula2, data=df32).fit()
# print(model2.summary())
r2_round = round(model2.rsquared, 3)
# print(r2_round) # 0.859

# 3) area=100, height=10, wall=1 값을 가진 데이터의 예측 주택 가격을 모델로 예측하여 소수점 셋째 자리까지 반올림하여 제출하시오./
# (단, 이전 문제에서 뽑은 통계적으로 유의미한 변수만 선택할 것)
data2 = pd.DataFrame({'area': [100], 'height': [10], 'wall': [1]})
test2 = data2[sig_vars_names]
pred_prob = model2.predict(test2)
result = round(pred_prob, 3)
# print(result) # 329.036
