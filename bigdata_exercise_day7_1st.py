
# 작업형 제1유형
import pandas as pd
path1 = "https://raw.githubusercontent.com/Soyoung-Yoon/bigdata/main/"
path2 = "https://raw.githubusercontent.com/Soyoung-Yoon/data_01/main/"
# 7-1) 지역별 공간점수
# 다음 절차에 따라 문제를 해결하시오.
# - bed와 bath의 값이 모두 0이 아닌 행만 선택한다.
# - 각 행(row)별로 ```bed * 1.5 + bath * 2``` 의 값을 '공간점수'라고 정의한다.
# - 지역(zip_code)별로 공간점수의 평균을 계산한 후, 지역별 평균 공간점수가 
#   가장 높은 7개 지역의 가격(price) 평균을 반올림하여 정수로 구하라.
df = pd.read_csv(path1 + "housing01.csv")
# print(df.head(3), df.shape, sep="\n") # (104667, 12)
# bed와 bath의 값이 모두 0이 아닌 행만 선택
df = df[(df['bed'] > 0) & (df['bath'] > 0)]
# 각 행(row)별로 ```bed * 1.5 + bath * 2``` 의 값을 '공간점수'라고 정의
df['공간점수'] = (df['bed'] * 1.5) + (df['bath'] * 2)
# print(df.head(3))
# 지역(zip_code)별로 공간점수의 평균을 계산한 후, 지역별로 평균 공간점수가 가장 높은 7개 지역 추출
high_zip = df.groupby('zip_code')[['공간점수']].mean().sort_values('공간점수', ascending=False)[:7].index.values
# 장 높은 7개 지역의 가격(price) 평균을 반올림하여 정수로 구하라.
# print(round(df[df['zip_code'].isin(high_zip)]['price'].mean())) # 11146024

# 7-2) 이벤트 발생 빈도
# 다음 절차에 따라 문제를 해결하시오.
# (년, 월)별로 발생한 이벤트의 개수를 확인하여, 이벤트 발생 개수 TOP3의 이벤트 수 합을 구하여 정수로 출력하시오.
# TOP3는 가장 큰 값 3개를 의미한다.
# start_datetime : 발생 날짜/시간
# end_datetime : 종료 날짜/시간
df = pd.read_csv(path2 + "event_log_03.csv")
# print(df.head(3))
# (년, 월)별로 발생한 이벤트의 개수를 확인
df['start_datetime'] = df['start_datetime'].astype('datetime64[ns]').astype(str)
df['year'] = df['start_datetime'].str[:4]
df['month'] = df['start_datetime'].str[5:7]
# df2 = df.groupby(['year','month','event'])['event'].count().reset_index(name='count').sort_values('count', ascending=False).head(3)
# print(df.groupby(['year','month'])['event'].size().sort_values(ascending=False).head(3).sum()) # 111
# 이벤트 발생 개수 TOP3의 이벤트 수 합을 구하여 정수로 출력
# print(int(df2['count'].sum())) # 32

# 7-3) 이벤트 발생 시간분석
# 다음 절차에 따라 문제를 해결하시오.
# 이벤트가 시작된 시각(start_datetime)을 기준으로 이벤트가 가장 많이 발생한 시(hour)를 구하시오.
# 해당 시각에 발생한 모든 이벤트의 value 값을 합산하고, 그 합계를 반올림하여 정수로 출력하시오.
df = pd.read_csv(path2 + "event_log_03.csv")
# print(df.head(3))
# 이벤트가 시작된 시각(start_datetime)을 기준으로 이벤트가 가장 많이 발생한 시(hour)를 구하시오
df['start_datetime'] = df['start_datetime'].astype('datetime64[ns]')
df['hour'] = df['start_datetime'].dt.hour
hour = df.groupby('hour')['event'].count().idxmax()
# 해당 시각에 발생한 모든 이벤트의 value 값을 합산하고, 그 합계를 반올림하여 정수로 출력
# print(round(df[df['start_datetime'].dt.hour == hour]['value'].sum())) # 2714

# 7-4) 월별 이벤트 발생 빈도
# 다음 절차에 따라 문제를 해결하시오.
# (년, 월)별로 발생한 이벤트의 개수를 확인하여, 년별로 가장 많은 이벤트가 발생한 월에 해당하는 
# 데이터 개수의 합을 구하여 정수로 출력하시오.
df = pd.read_csv(path2 + "event_log_03.csv")
# print(df.head(3))
# (년, 월)별로 발생한 이벤트의 개수를 확인하여, 년별로 가장 많은 이벤트가 발생한 월 추출
df['start_datetime'] = df['start_datetime'].astype('datetime64[ns]')
df['year'] = df['start_datetime'].dt.year
df['month'] = df['start_datetime'].dt.month
# df2 = df.groupby(['year','month','event'])['event'].count().reset_index(name='count').sort_values('count', ascending=False)
# max_years = df2.groupby('year')['count'].max().index.values
# print(df.groupby(['year','month']).size().unstack().max(axis=1).sum()) # 108
# 데이터 개수의 합을 구하여 정수로 출력하시오.
# print(len(df2[df2['year'].isin(max_years)]['month'])) # 215

# 7-5a) 지속시간의 분위수 분석
# 다음 절차에 따라 문제를 해결하시오.
# 지속시간 기준 75분위수(=제3사분위수) 이상에 해당하는 데이터의 지속시간(duration) 합계를 초단위 정수로 출력한다.
# 지속시간(duration) = end_datetime - start_datetime
df = pd.read_csv(path2 + "event_log_03.csv")
# print(df.head(3))
df['start_datetime'] = df['start_datetime'].astype('datetime64[ns]')
df['end_datetime'] = df['end_datetime'].astype('datetime64[ns]')
df['duration'] = df['end_datetime'] - df['start_datetime']
dur = df['duration']
components = dur[dur >= dur.quantile(0.75)].dt.components.sum()
seconds = dur[dur >= dur.quantile(0.75)].sum().total_seconds()
# print(int((components.minutes * 60) + components.seconds)) # 133254
# print(int(seconds)) # 133254

# 작업형 제2유형
# Seoul Bike Sharing Dataset 데이터셋
# 서울시의 시간대별 자전거 대여 데이터를 포함하고 있으며, 날씨와 계절 요인이 자전거 대여량에 어떤 영향을 미치는지를 
# 분석할 수 있는 시계열형 예측 데이터셋입니다.
# 제공된 학습용 데이터(bike_train.csv)를 이용하여 대여 자전거 대수(Rented_Bike_Count)를 예측하는 모델을 개발하고, 
# 개발한 모델에 기반하여 평가용 데이터(bike_test.csv)에 적용하여 얻은 대여 자전거 대수 예측 값을 
# 아래 [제출형식]에 따라 csv 파일로 생성하여 제출하시오.
# 예측 결과는 RMSE(Root Mean Squared Error) 평가지표에 따라 평가함
# 성능이 우수한 예측 모델을 구축하기 위해서는 데이터 정제, Feature Engineering, 
# 하이퍼 파라미터(hyper parameter) 최적화, 모델 비교 등이 필요할 수 있음. 다만, 과적합에 유의하여야 함
# [[제출 형식]]
# 가. CSV 파일명: result.csv(파일명에 디렉토리/폴더 지정불가
# 나. 예측 칼럼명 : pred
# 다. 제출 칼럼 개수 : pred 칼럼 1개
# 라. 평가용 데이터 개수와 예측 결과 데이터 개수 일치 : 2,716개
# 라이브러리
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error as MSE

# 데이터 확인
train = pd.read_csv(path1 + "bike_train.csv")
test = pd.read_csv(path1 + "bike_test.csv")
# print(train.head(3), train.shape, sep="\n") # (6044, 14)
# print(test.head(3), test.shape, sep="\n")   # (2716, 13)

# 데이터 전처리
X = train.drop(columns=['Rented_Bike_Count'])
Y = train['Rented_Bike_Count']
X_all = pd.concat([X, test])
X_all['Date'] = X_all['Date'].astype('datetime64[ns]')
X_all['Year'] = X_all['Date'].dt.year
X_all['Month'] = X_all['Date'].dt.month
X_all['Day'] = X_all['Date'].dt.day
X_all['Weekday'] = X_all['Date'].dt.day_name('ko_KR')
X_all = X_all.drop(columns=['Date'])
cols_obj = X_all.select_dtypes(include='object').columns
for col in cols_obj:
    X_all[col] = LabelEncoder().fit_transform(X_all[col])
# print(X_all.head(3))

# 데이터 재분할
X = X_all.iloc[:len(X), :]
X_submission = X_all.iloc[len(X):, :]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1234)
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (4230, 16) (1814, 16) (4230,) (1814,)

# 파이프라인 모델사전
models = {
    "Linear": Pipeline([
        ("scaler", MinMaxScaler()), ("model", LinearRegression())
    ]),
    "DecisionTree": Pipeline([
        ("model", DecisionTreeRegressor(max_depth=10, random_state=1234))
    ]),
    "RandomForest": Pipeline([
        ("model", RandomForestRegressor(n_estimators=1000, max_depth=10, random_state=1234))
    ]),
    "AdaBoost": Pipeline([
        ("model", AdaBoostRegressor(n_estimators=1000, random_state=1234))
    ]),
    "GradientBoosting": Pipeline([
        ("model", GradientBoostingRegressor(random_state=1234))
    ])
}

# 성능평가 함수
def get_scores(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    y_pred1 = abs(model.predict(x_train))
    y_pred2 = abs(model.predict(x_test))
    RMSE_train = MSE(y_train, y_pred1) ** 0.5
    RMSE_test = MSE(y_test, y_pred2) ** 0.5
    return model, RMSE_train, RMSE_test

# 모델별 성능평가
results = []
for name, model in models.items():
    model, RMSE_train, RMSE_test = get_scores(model, x_train, x_test, y_train, y_test)
    results.append({
        "Model": name, "RMSE_train": f"{RMSE_train:.4f}", "RMSE_test": f"{RMSE_test:.4f}"
    })
res = pd.DataFrame(results).sort_values("RMSE_test").reset_index(drop=True)
# print(res)

# 모델선택, 예측산출
model = models[res.loc[0, "Model"]]
y_pred = model.predict(X_submission)

# 제출파일 생성
# pd.DataFrame({'pred': y_pred}).to_csv("result_day7_1st.csv", index=False)

# 결과확인
# temp = pd.read_csv("result_day7_1st.csv")
# print(Y[:len(test)].describe())
# print("=" * 35)
# print(temp['pred'].describe())

# 작업형 제3유형
# 6회 2번. 다중 선형 회귀
# 다음은 연령(age), 몸무게(weight), 콜레스테롤 수치(cholesterol)에 대한 일부 표본 데이터를 사용하여 
# 선형 회귀 분석을 수행하여 다음 물음에 답하시오.
df = pd.read_csv(path1 + "cholesterol.csv")
# print(df.head(3))

# 종속변수 : weight
# 독립변수 : age, cholesterol
# 위의 조건에 따라 모형을 생성, 학습하여 다음 물음에 답하시오.
from statsmodels.api import OLS
formula = "weight ~ age + cholesterol"
model = OLS.from_formula(formula, df).fit()
# print(model.summary())
# 1. 위에서 생성된 모델에 대해 age의 회귀계수를 구하고, 반올림하여 소수점 아래 3자리까지 출력하시오.
# print(round(model.params['age'], 3)) # -0.036

# 2. age가 고정된 상태에서 cholesterol과 weight 사이에 선형관계가 존재한다는 가설을 세운다. 
#    이 가설을 유의수준 0.05 하에에 검정하고, 통계적으로 유의미한 관계가 있는지 여부를 
#    "있음" 또는 "없음"으로 표시하시오.
# print("채택" if model.pvalues['cholesterol':'weight'] <= 0.05 else "기각")
result2 = model.pvalues['cholesterol']
# print("없음" if result2 > 0.05 else "있음") # 있음

# 3. 위 모델을 기반으로 age = 55, cholesterol = 72.6일 때의 weight 값을 예측하고, 
#    반올림하여 소수점 아래 4자리까지 출력하시오.
data = pd.DataFrame({'age': [55], 'cholesterol': [72.6]})
# print(round(model.predict(data)[0], 4)) # 78.6219

# 7회 1번. 로지스틱 회귀
# 다음은 생물학적 특성(age, diameter, height, weight)과 성별(gender)에 대한 데이터이다.
# 총 300개 샘플로 구성되어 있으며, 다음과 같이 학습용과 평가용으로 분할하여 사용한다.
# 학습용: 1 ~ 210번 샘플 (학습 모델 생성에 사용)
# 평가용: 211 ~ 300번 샘플
# 종속변수: gender, 독립변수: age, diameter, height, weight
# 상수항(절편)을 포함하고, 규제는 적용하지 않는다.
# 단, gender는 이진 변수로 처리되어 있으며, 분석에 적합한 형태로 변환되어 있다.

# 위 데이터를 바탕으로 **로지스틱 회귀 분석을 수행하여** 다음 물음에 답하시오.
df = pd.read_csv(path1 + "gender_classification.csv")
# print(df.head(3))
train = df.iloc[:211, :]
test = df.iloc[211:, :]
# print(train.shape, test.shape) # (211, 5) (89, 5)
from statsmodels.api import GLM, families
formula = "gender ~ age + diameter + height + weight"
model = GLM.from_formula(formula, train, family=families.Binomial()).fit()
# print(model.summary())
# 1. 로지스틱 회귀 모형에서 'weight' 변수를 설명변수로 사용할 때의 오즈비(odds ratio)를 소수점 아래 3자리까지 반올림하여 구하시오.
# y_1 = model.params['weight']
# odds_ratio = y_1 / (1 - y_1)
import numpy as np
result1 = np.exp(model.params['weight'])
# print(round(result1, 3)) # 0.997

# 2. 로지스틱 회귀 모형의 잔차이탈도(residual deviance)를 반올림하여 소수점 아래 4자리까지 구하시오.
# print(round(model.deviance, 4)) # 57.3795

# 3. 로지스틱 회귀 모형에 평가용 데이터를 적용해, gender를 예측하고, 
# 예측값과 실제값 간의 오차율(error rate)을 반올림하여 소수점 아래 3자리까지 구하시오.
y_proba = model.predict(test)
y_pred = round(y_proba)
result3 = (test['gender'] != y_pred).mean()
# print(round(result3, 3)) # 0.034

# 7회 2번. 다중 선형 회귀
# 여러 개의 독립변수를 기반으로 target 값을 예측하기 위해 다중 선형 회귀모형을 구축하시오.
# 종속변수: target
# 독립변수: target을 제외한 모든 변수
df = pd.read_csv(path1 + "mlr_noisy.csv")
# print(df.head(3))
from statsmodels.api import OLS
formula = "target ~ " + " + ".join(df.columns[:-1])
# print(formula)
model = OLS.from_formula(formula, df).fit()
# print(model.summary())
# 위의 구축된 회귀모형을 사용하여 다음 물음에 답하시오.
# 1. 가장 높은 회귀계수를 구해, 반올림하여 소수점 아래 3자리까지 출력하시오.
# print(round(model.params[1:].max(), 3)) # 84.177
# 2. 적합된 선형 회귀 모형의 결정계수를 구해, 반올림하여 소수점 아래 4자리까지 출력하시오.
# print(round(model.rsquared, 4)) # 0.9847
# 3. 독립변수 중 가장 높은 p-value를 구해, 반올림하여 소수점 아래 3자리까지 출력하시오.
# print(round(model.pvalues[1:].max(), 3)) # 0.996