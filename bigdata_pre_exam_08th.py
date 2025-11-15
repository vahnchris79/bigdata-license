
# 기출문제 제8회 2024-06-22
# 작업형 제1유형
import pandas as pd
pd.set_option('display.width', 120)
path = "https://raw.githubusercontent.com/YoungjinBD/data/main/exam/"

# 1. 다음의 데이터는 대륙별 국가의 맥주소비량을 조사한 것이다.
# ① 평균 맥주소비량이 가장 많은 대륙을 구하시오.
df1 = pd.read_csv(path + "8_1_1.csv")
# print(df1.head())
result1 = df1.groupby('대륙')['맥주소비량'].mean().idxmax()
# print(result1) # SA

# ② 이전 문제에서 구한 대륙에서 5번째로 맥주소비량이 많은 나라를 구하시오.
# result2 = df1[df1['대륙']==result1].groupby('국가')['맥주소비량'].nlargest(4).idxmin()[0]
df1 = df1[df1['대륙'] == result1]
result2 = df1.groupby('국가')['맥주소비량'].sum().nlargest(5).idxmin()
# print(result2) # Peru

# ③ 이전 문제에서 구한 나라의 평균 맥주소비량을 구하시오.(소수점 첫째 자리에서 반올림)
# print(round(df1[df1['국가']==result2]['맥주소비량'].mean())) # 249 -> 253

# 2. 다음의 데이터는 국가별로 방문객 유형을 조사한 것이다.
df2 = pd.read_csv(path + "8_1_2.csv")
# print(df2.head(3))
# ① 관광객비율이 두 번째로 높은 나라의 관광 수를 구하시오.
#   - 관광객비율 =  관광 / 합계(관광+사무+공무+유학+기타), 소수점 넷째 자리에서 반올림
df2['합계'] = df2.iloc[:, 1:].sum(axis=1)
df2['관광객비율'] = (df2['관광'] / df2['합계']).round(3)
# print(df2.head(3))
# df2 = df2.set_index('국가')
# A = df2[df2.index == df2['관광객비율'].nlargest(2).idxmin()]['관광'].nlargest(2).min()
# print(df2.sort_values('관광객비율', ascending=False))

# print(df2[df2['관광객비율'].sort_values(ascending=False).nlargest(2)])
# print(A) # 6911

# ② 관광 수가 두 번째로 높은 나라의 공무수의 평균을 구하시오.(소수점 첫째 자리에서 반올림)
B = df2[df2.index == df2['관광'].nlargest(2).idxmin()]['공무'].mean().round(0)
# print(B) # 494.0

# ③ 이전에 구한 관광 수와 공무 수의 합계를 구하시오.
# print(A + B) # 7405

# 3. CO(GT), NMHC(GT)칼럼에 대해서 Min-Max스케일러를 실행하고, 스케일링된 CO(GT), NMHC(GT)칼럼의
# 표준편차를 구하시오.(소수점 셋째 자리에서 반올림)
df3 = pd.read_csv(path + "8_1_3.csv")
# print(df3.head(3))

from sklearn.preprocessing import MinMaxScaler
df3 = df3.loc[:, 'CO(GT)':'NMHC(GT)'].copy()
temp = MinMaxScaler().fit_transform(df3)
df3_scaled = pd.DataFrame(temp, columns=df3.columns)
co_std = round(df3_scaled['CO(GT)'].std(ddof=1), 2)
nmhc_std = round(df3_scaled['NMHC(GT)'].std(ddof=1), 2)
# print(co_std, nmhc_std, sep=", ") # 0.37, 0.15

# 작업형 제2유형
# 제공된 학습용 데이터(8_2_train.csv)는 자전거 대여와 관련된 날짜별 정보와 해당 날짜의 총 대여 건수(count)를 포함하고 있다.
# 학습용 데이터를 활용하여 자전거 총 대여 건수(count)를 예측하는 회귀 모델을 개발하고, 성능이 가장 우수한 모델을 
# 평가용 데이터(8_2_test.csv)에 적용하여 예측 결과를 제출하시오.
# 모델 성능 지표:  MAE(Mean Absolute Error)
# Data Description
# ID: 고유 식별자, holiday: 공휴일 여부, workingday: 평일여부, weather: 날씨 상황, temp: 실제 기온, atemp: 체감 기온, .humidity: 습도,
# windspeed: 풍속, count:. 자전거 총 대여 건수
# 제출 형식
# 파일명: result.csv, 제출 칼럼: ID, pred, pred: 예측된 자전거 대여 건수(정수 또는 소수 가능), 행 수: 테스트 데이터의 ID수와 동일

# 라이브러리
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error as MAE

# 데이터 확인
train = pd.read_csv(path + "8_2_train.csv")
test = pd.read_csv(path + "8_2_test.csv")
# print(train.head(3), train.info(), sep="\n") # 378, # 결측치 없음
# print(test.head(3), test.info(), sep="\n")   # 166, # 결측치 없음

# 데이터 전처리
X = train.drop(columns=['ID', 'count'])
Y = train['count']
X_submission = test.drop(columns=['ID', 'count'])
X_all = pd.concat([X, X_submission])
for col in ['holiday', 'workingday', 'weather']:
    X_all[col] = LabelEncoder().fit_transform(X_all[col])
# X_all = pd.get_dummies(X_all, drop_first=True, dtype='int32')
# print(X_all.head(3))

# 데이터 분할
X = X_all.iloc[:len(X), :]
X_submission = X_all.iloc[len(X):, :]
# print(X.shape, X_submission.shape) # (378, 7) (166, 7)
temp = train_test_split(X, Y, test_size=0.3, random_state=1)
x_train, x_test, y_train, y_test = temp

# 파이프라인 모델사전
models = {
    "Linear": Pipeline([
        ("scaler", MinMaxScaler()), ("model", LinearRegression())
    ]),
    "DecisionTree": Pipeline([
        ("model", DecisionTreeRegressor(max_depth=3, random_state=1))
    ]),
    "RandomForest": Pipeline([
        ("model", RandomForestRegressor(n_estimators=100, max_depth=3, random_state=1))
    ]),
    "AdaBoost": Pipeline([
        ("model", AdaBoostRegressor(n_estimators=900, random_state=1))
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
    MAE_train = MAE(y_train, y_pred1)
    MAE_test = MAE(y_test, y_pred2)
    return model, MAE_train, MAE_test

# 모델별 성능평가
results = []
for name, model in models.items():
    model, MAE_train, MAE_test = get_scores(model, x_train, x_test, y_train, y_test)
    results.append({
        "Model": name, "MAE_train": f"{MAE_train:.4f}", "MAE_test": f"{MAE_test:.4f}"
    })
res = pd.DataFrame(results).sort_values("MAE_test").reset_index(drop=True)
# print(res)

# 모델선택, 예측결과 생성
model = models[res.loc[0, 'Model']]
y_pred = model.predict(X_submission)

# 제출파일 생성
# pd.DataFrame({'ID': test['ID'], 'pred': y_pred}).to_csv("result_08th_type2.csv", index=False)

# 결과확인
# temp = pd.read_csv("result_08th_type2.csv")
# print(Y[:len(X_submission)].describe())
# print("=" * 35)
# print(temp['pred'].describe())

# 작업형 제3유형
# 문제1
# 어느 회사에서 직원들의 업무 효율성을 높이기 위한 새로운 소프트웨어를 도입하였다. 도입 전과 도입 후의 업무 처리 시간을 각각 측정하여
# 새로운 소프트웨어의 효과를 검증하고자 한다.
df3 = pd.read_csv(path + "8_3_1.csv")
# print(df3.head(3))

# ① 도입 전과 도입 후의 업무처리 시간의 평균과 표준편차를 구하시오.(소수점 둘째 자리까지 반올림)
before_mean = df3['before'].mean().round(2)
before_std = round(df3['before'].std(ddof=1), 2)
after_mean = df3['after'].mean().round(2)
after_std = round(df3['after'].std(ddof=1), 2)
# print(f"도입 전 평균: {before_mean}, 도입 후 평균: {after_mean}")       # 도입 전 평균: 8.21, 도입 후 평균: 7.23
# print(f"도입 전 표준편차: {before_std}, 도입 후 표준편차: {after_std}") # 도입 전 표준편차: 1.71, 도입 후 표준편차: 1.96

# ② 도입 전후의 업무처리 시간 차이가 유의미한 지 부호 순위 검정을 실시하고, 검정통계량을 계산하시오.(소수점 둘째 자리까지 반올림)
from scipy.stats import wilcoxon
statistic, p_value = wilcoxon(df3['before'], df3['after'])
# print(statistic, p_value)
# print(round(statistic, 3)) # 72.000

# ③ p-value를 바탕으로 유의수준 5%에서 귀무가설의 기각/채택 여부를 결정하시오.(p-value는 소수점 둘째 자리까지 반올림)
# print("기각" if round(p_value, 2) < 0.05 else "채택") # 기각

# 문제2
# 어느 회사에서 직원들의 생산성에 영향을 미치는 요인이 무엇인지 확인하고자 한다. 100명의 직원들을 대상으로 생산성 점수, 근무 시간, 연령, 그리고 경력을 조사하였다.
train = pd.read_csv(path + "8_3_2_train.csv")
test = pd.read_csv(path + "8_3_2_test.csv")
# print(train.head(3))

# ① 훈련데이터를 기준으로 생산성 점수(productivity)를 종속변수로 하고, 근무시간, 연령, 그리고 경력을 독립변수 하는 다중회귀 분석을 수행한 후 
#    회귀계수가 가장 높은 변수를 구하시오.(다중회귀모형 적합 시 절편 포함)
from statsmodels.api import OLS, add_constant
formula = "productivity ~ " + " + ".join(train.columns[1:])
model = OLS.from_formula(formula, train).fit()
# print(model.summary())
# print(model.params[1:].idxmax()) # hours

# ② 유의수준 5%하에서 각 독립변수가 생산성에 미치는 영향이 통계적으로 유의미한 지 판단하고, 유의미한 변수 개수를 구하시오(p-value는 소수점 넷째 자리까지 반올림)
# print(sum(round(model.pvalues[1:],4) <= 0.05)) # 3

# ③ 테스트 데이터로 모델의 성능을 평가하시오 (R^2 산출)
from sklearn.metrics import r2_score
x_test = test[['hours','age','experience']]
y_test = test['productivity']
x_test = add_constant(x_test)
y_pred = model.predict(x_test)
# print(round(r2_score(y_test, y_pred), 3)) # 0.804
