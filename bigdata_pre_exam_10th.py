
# 기출문제 제10회(2025-06-21)
# 작업형 제1유형
import pandas as pd
path = "https://raw.githubusercontent.com/YoungjinBD/data/main/exam/"

# 1. 소주제별로 정답률(정답여부가 1인 응답수/해당 소주제 전체 응답수)를 구하고 3번째로 높은 정답률을 구하시오.
df1 = pd.read_csv(path + "10_1_1.csv")
# print(df)
# 정답여부가 1인 응답수
df_y1 = df1[df1['정답여부']==1].groupby('소주제')['정답여부'].count()
df_all = df1.groupby('소주제')['정답여부'].count()
df_ratio = pd.merge(df_y1, df_all, "left", "소주제")
df_ratio['정답률'] = df_ratio['정답여부_x'] / df_ratio['정답여부_y']
# print(df_ratio.sort_values('정답률', ascending=False)[:4].min().values[2]) # 0.68

# 2. 제시된 문제를 순서대로 풀고 해답을 제시하시오.
# ① date를 연도(year), 월(month)로 분리하여 연도-월별 price의 합계를 구하시오. 그 중 두번째로 큰 매출액(합계)를 구하시오.
df2 = pd.read_csv(path + "10_1_2.csv")
# print(df2.head())
df2['date'] = df2['date'].astype('datetime64[ns]')
df2['year'] = df2['date'].dt.year
df2['month'] = df2['date'].dt.month
# print(df2.head(3))
result = df2.groupby(['year', 'month'], as_index=False)['price'].sum().sort_values('price', ascending=False).head(2)['price'].min()
# print(result) # 1777389

# ② 이전 문제에서 네 번째로 큰 price합계에 해당하는 연도-월을 찾으시오. 해당 연도-월에서 카테고리별 price합계를 구하시오.
#    그 중 가장 높은 price합계(정수)를 제출하시오.
year = df2.groupby(['year', 'month'], as_index=False)['price'].sum().sort_values('price', ascending=False).head(4)['year'].max()
month = df2.groupby(['year', 'month'], as_index=False)['price'].sum().sort_values('price', ascending=False).head(4)['month'].max()
result = df2[(df2['year']==year) & (df2['month']==month)].groupby('category', as_index=False)['price'].sum()['price'].max()
# print(int(result)) # 1012500

# 3. 제신된 문제를 순서대로 풀고, 해답을 제시하시오.
df3 = pd.read_csv(path + "10_1_3.csv")
# print(df3.head(3))

# ① 각 메시지의 단어 수를 공백(' ')을 기준으로 세는 새로운 컬럼을 만드시오.
# df3['cnt'] = 0
# for i in df3.index:
#     df3.loc[i, 'cnt'] = len(df3.loc[i, 'message'].split(' '))
df3['cnt'] = df3['message'].str.split(' ').apply(len)
# print(df3.head())

# ② 'spam'과 'ham' 각각의 평균 단어 수를 계산하시오.
# print(df3.head())
spam_mean = df3[df3['label']=='spam'].groupby('label', as_index=False)['cnt'].mean()['cnt'].values[0]
ham_mean = df3[df3['label']=='ham'].groupby('label', as_index=False)['cnt'].mean()['cnt'].values[0]
# print(spam_mean)
# print(ham_mean)

# ③ 두 평균의 차이의 절댓값을 소수점 셋째자리까지 반올림하여 제출하시오.
# pd.options.display.float_format = ':.3f'.format()
# print(round(abs(spam_mean - ham_mean), 3)) # 0.300

# 작업형 제2유형
# 제공된 학습용 데이터(10_2_train.csv)는 여러 상권 내 건물의 특성(상권 유형, 건물 면적, 건물 연식, 세대 수 등)과
# 연간 총 가스 소비량 정보를 담고 있다. 학습용 데이터를 활용하여 건물의 연간 총 가스 소비량(gas_totl)을 예측하는
# 모델을 개발하고, 이 중 가장 우수한 모델을 평가용 데이터(10_2_test.csv)에 적용하여 예측 결과를 제출하시오.
# 평가지표: RMSE

# 라이브러리
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error as RMSE

# 데이터 불러오기
train = pd.read_csv(path + "10_2_train.csv")
test = pd.read_csv(path + "10_2_test.csv")
# print(train.head(3), train.shape, sep="\n") # (160, 5)
# print(test.head(3), test.shape, sep="\n")   #  (40, 5)

# 데이터 전처리
train = train[train['gas_totl'] != 0]
X = train.drop(columns=['gas_totl'])
Y = train['gas_totl']
X_submission = test.drop(columns=['gas_totl'])
X_all = pd.concat([X, X_submission])
# print(X_all.head(3))
X_all['biz_type'] = LabelEncoder().fit_transform(X_all['biz_type'])
X_all = pd.get_dummies(X_all, drop_first=True, dtype='int32')
# print(X_all.head())

# 데이터 분할
X = X_all.iloc[:len(X), :]
X_submission = X_all.iloc[len(X):, :]
# print(X.shape, X_submission.shape) # (160, 4) (40, 4)
temp = train_test_split(X, Y, test_size=0.3, random_state=1)
x_train, x_test, y_train, y_test = temp
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (112, 4) (48, 4) (112,) (48,)

# 파이프라인 모델사전
models = {
    "Linear": Pipeline([
        ("scaler", MinMaxScaler()), ("model", LinearRegression())
    ]),
    "DecisionTree": Pipeline([
        ("model", DecisionTreeRegressor(max_depth=3, random_state=1))
    ]),
    "RandomForest": Pipeline([
        ("model", RandomForestRegressor(max_depth=3, random_state=1))
    ]),
    "AdaBoost": Pipeline([
        ("model", AdaBoostRegressor(n_estimators=50, random_state=1))
    ]),
    "GradientBoosting": Pipeline([
        ("model", GradientBoostingRegressor(random_state=1))
    ])
}

# 성능평가 함수
def get_scores(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    y_pred1 = model.predict(x_train)
    y_pred2 = model.predict(x_test)
    RMSE_train = RMSE(y_train, y_pred1)
    RMSE_test = RMSE(y_test, y_pred2)
    return model, RMSE_train, RMSE_test

# 모델별 성능평가
results = []
for name, model in models.items():
    model, RMSE_train, RMSE_test = get_scores(model, x_train, x_test, y_train, y_test)
    results.append({
        "Model": name,"RMSE_train": f"{RMSE_train:.4f}", "RMSE_test": f"{RMSE_test:.4f}"
    })
res = pd.DataFrame(results).sort_values("RMSE_test").reset_index(drop=True)
# print(res)

# 모델적용
model = models[res.loc[0, "Model"]]
y_pred = model.predict(X_submission)

# 제출파일 생성
# pd.DataFrame({'pred': y_pred}).to_csv("result_10th.csv", index=False)

# 결과확인
# temp = pd.read_csv("result_10th.csv")
# print(temp['pred'].describe())
# print("=" * 35)
# print(Y[:len(X_submission)].describe())

# 작업형 제3유형
# 문제1
# 한 기업이 인사관리 데이터를 가지고 이직 여부 예측 모델을 개발하려고 한다. 다음 문제를 풀이하시오.
# Data description
# attrition: 이직여부(0: 잔류, 1: 이직), age: 나이, income: 연봉, overtime: 야근상태(0: 해당없음, 1: 보통, 2: 상시)
# ① 이직 여부를 예측하는 로지스틱 회귀모형을 적합하고, 유의한 변수(유의확률 0.05미만)의 회귀계수를 
# 소수점 셋째 자리까지 반올림하여 제출하시오.(단, 절편 제외)
from statsmodels.api import GLM, families
import pandas as pd
import numpy as np
df = pd.read_csv(path + "10_3_1.csv")
formula = "attrition ~ age + income + C(overtime)"
model = GLM.from_formula(formula, df, family = families.Binomial()).fit()
# print(model.summary())
# print(model.pvalues[1:] < 0.05) # income, 0.06 (X)
pvalues = model.pvalues[1:]
params = model.params[1:]
# print(round(params[pvalues < 0.05].values[0], 3)) # -0.005 

# ② age가 1 증가할 때 이직(또는 잔류) 오즈비(odds_ratio)를 소수점 셋째 자리까지 반올림하여 제출하시오
# print(round(np.exp(model.params['age'] * 2), 3)) # 0.819 (X)
# print(round(np.exp(model.params['age']), 3)) # 0.894

# ③ age=20, income=3000, overtime=2 값을 가진 데이터의 이직확률을 모델로 예측하여 소수점 셋째 자리까지 반올림하여 제출하시오.
data = pd.DataFrame({'age': [20], 'income': [3000], 'overtime': [2]})
result = model.predict(data)[0]
print(round(result, 3)) # 0.431 -> 0.480

# 문제2
# 어느 지역의 주택들의 정보를 수집하여 주택 가격을 예측하는 모델을 개발하려고 한다. 다음 문제를 풀이하시오.
# Data Description
# price: 주택 가격, area: 주택 면적, height: 집 높이, wall: 벽 유무(0: 없음, 1: 있음)
# ① 주택 가격을 예측하는 다중선형회귀모형을 적합하고, 유의한 변수(유의확률 0.05 미만)의 회귀계수 합(절편 제외)을 소수점 셋째 자리까지 반올림하여 제출하시오.
from statsmodels.api import OLS
import pandas as pd
import numpy as np
df = pd.read_csv(path + "10_3_2.csv")

formula = "price ~ area + height + wall"
model = OLS.from_formula(formula, df).fit()
# print(model.summary())
# print(round(model.params[1] + model.params[2], 3)) # 10.289

# ② 유의한 변수만으로 다중선형회귀모형을 다시 적합하고, 결정계수를 소수점 셋째 자리까지 반올림하여 제출하시오.
# formula2 = "price ~ area + height"
# model2 = OLS.from_formula(formula2, df).fit()
# print(round(model2.rsquared, 3)) # 0.859

# ③ area=100, height=10, wall=1 값을 가진 데이터의 예측 주택 가격을 모델로 예측하여 소수점 셋째 자리까지 반올림하여 제출하시오.
#    (단, 이전 문제에서 뽑은 통계적으로 유의미한 변수만 선택할 것)
# data = pd.DataFrame({'area': [100], 'height': [10], 'wall': [1]})
# result = model2.predict(data)[0]
# print(round(result, 3)) # 329.036