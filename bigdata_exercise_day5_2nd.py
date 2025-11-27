# 작업형 제1유형
import pandas as pd
pd.set_option('display.width', 120)
path1 = "https://raw.githubusercontent.com/Soyoung-Yoon/bigdata/main/"
path2 = "https://raw.githubusercontent.com/Soyoung-Yoon/data_01/main/"

# 5-1) tips 데이터셋을 이용한다.
# tip / total_bill 비율이 0.15를 초과하고, size가 3 이상인 경우만 필터링한다.
# 이 조건을 만족하는 손님의 총 인원 수(size의 합)를 구하시오.
df = pd.read_csv(path1 + "tips.csv")
df = df[((df['tip'] / df['total_bill']) > 0.15) & (df['size'] >= 3)].copy()
# print(df['size'].sum()) # 143

# 5-2) 데이터에서 month가 'Jan'부터 'Jun'까지인 경우를 상반기, 'Jul'부터 'Dec'까지인 경우를 하반기로 나누어 분석한다.
# 상반기와 하반기 각각에 대해, 연도(year)별 총 승객 수(passengers)를 계산한다.
# 각 데이터에서, 연도별 승객 수 증가량을 계산하라. 단, 첫 번째 연도는 증가량을 0으로 둔다.
# 상반기와 하반기 각각의 증가량 중, 가장 많이 증가한 연도를 구한다.
# flights 전체 데이터에서, 최대 승객 수를 구한다.
# (상반기 최대 증가 연도) + (하반기 최대 증가 연도) + (최대 승객 수)의 결과를 정수로 출력하시오.
df = pd.read_csv(path1 + "flights.csv")
# print(df.head(3))
# print(df['month'].unique())
s1 = df[df['month'].isin(['Jan','Feb','Mar','Apr','May','Jun'])].groupby('year')['passengers'].sum()
s2 = df[df['month'].isin(['Jul','Aug','Sep','Oct','Nov','Dec'])].groupby('year')['passengers'].sum()
year1 = s1.diff().fillna(0).idxmax()
year2 = s2.diff().fillna(0).idxmax()
max_passengers = df['passengers'].max()
# print(int(year1 + year2 + max_passengers)) # 4541

# 5-3) tips 데이터셋에서 다음 조건을 만족하는 손님의 total_bill 평균을 반올림하여 소수점 아래 3자리까지 구하시오.
# total_bill이 자신이 속한 요일(day)의 평균 이상인 손님만 선택한다.
# 위 조건을 만족하는 손님 중 sex가 Female이고, tip이 중앙값 이상인 데이터만 선택한다.
df = pd.read_csv(path1 + "tips.csv")
# print(df.head(3))
cond1 = df.groupby('day')['total_bill'].transform('mean')
df = df[df['total_bill'] >= cond1].copy()
result = df[(df['sex']=='Female') & (df['tip'] >= df['tip'].median())]['total_bill'].mean()
# print(round(result, 3)) # 29.335

# 5-4) titanic_dataq.csv를 사용하여 다음 조건을 만족하는 승객 수를 정수로 구하시오.
# Cabin 컬럼의 결측치를 Cabin 컬럼의 첫 글자 중 최빈값으로 채우기 한 뒤, Cabin 컬럼의 값을 첫 글자로 변경한다. (예) C12 => C
# Cabin 컬럼의 최빈값에 해당하는 데이터만 추출하여, 나이(Age) 컬럼의 결측치를 Embarked와 Pclass 조합별 평균 나이로 채운다.
# Age 컬럼의 결측치를 채운 후, Age가 Embarked와 Pclass 조합별 중앙값 이상이고, Fare가 중앙값보다 높은 승객만 남긴다.
df = pd.read_csv(path1 + "titanic_dataq.csv")
# print(df.head(3))
df['Cabin'] = df['Cabin'].str[0]
Cabin_mode = df['Cabin'].mode()[0]
# print(Cabin_mode)
df['Cabin'] = df['Cabin'].fillna(Cabin_mode)
df = df[df['Cabin'] == Cabin_mode]
s = df.groupby(['Embarked','Pclass'])['Age'].transform('mean')
df['Age'] = df['Age'].fillna(s)
s2 = df.groupby(['Embarked', 'Pclass'])['Age'].transform('median')
result = ((df['Age'] >= s2) & (df['Fare'] > df['Fare'].median())).sum()
# print(result) # 194

# 5-5) California housing 데이터셋을 활용하여 다음 조건을 모두 만족하는 계산을 수행하시오.
# Latitude 값을 기준으로 데이터를 5개 구간으로 분위수 분할하여 ocean_proximity 컬럼을 생성한다.
# (pd.qcut(..., q=5, labels=False)를 사용할 것)
# 다음의 필터링을 수행하기 전, 후의 데이터에 대해 ocean_proximity별 HouseAge의 중앙값(median)을 계산한다.
# 필터링 전 데이터에 대한 계산 결과를 S1, 필터링 후 데이터에 대한 계산 결과를 S2로 저장한다.
# HouseAge가 자신이 속한 ocean_proximity 그룹의 중앙값보다 큰 경우만 필터링한다.
# 최종적으로 S2에서 S1의 값을 뺀(S2 - S1) 값 중 최소값을 정수로 구하시오.
df = pd.read_csv(path1 + "california.csv")
df['ocean_proximity'] = pd.qcut(df['Latitude'], q=5, labels=False)
# print(df.head())
cond = df.groupby('ocean_proximity')['HouseAge'].transform('median')
S1 = df.groupby('ocean_proximity')['HouseAge'].median()
df2 = df[df['HouseAge'] > cond].copy()
S2 = df2.groupby('ocean_proximity')['HouseAge'].median()
# print(int(min(S2-S1))) # 7

# 작업형 제2유형
# **Seoul Bike Sharing Dataset 데이터셋**
# 서울시의 시간대별 자전거 대여 데이터를 포함하고 있으며, 날씨와 계절 요인이 자전거 대여량에 어떤 영향을 미치는지를 
# 분석할 수 있는 시계열형 예측 데이터셋입니다.
# 제공된 학습용 데이터(bike_train.csv)를 이용하여 대여 자전거 대수(Rented_Bike_Count)를 예측하는 모델을 개발하고, 
# 개발한 모델에 기반하여 평가용 데이터(bike_test.csv)에 적용하여 얻은 대여 자전거 대수 예측 값을 
# 아래 [제출형식]에 따라 csv 파일로 생성하여 제출하시오.
# - 예측 결과는 RMSE(Root Mean Squared Error) 평가지표에 따라 평가함
# - 성능이 우수한 예측 모델을 구축하기 위해서는 데이터 정제, Feature Engineering, 
# 하이퍼 파라미터(hyper parameter) 최적화, 모델 비교 등이 필요할 수 있음. 다만, 과적합에 유의하여야 함
# [[제출 형식]]
# - 가. CSV 파일명: result.csv(파일명에 디렉토리/폴더 지정불가
# - 나. 예측 칼럼명 : pred
# - 다. 제출 칼럼 개수 : pred 칼럼 1개
# - 라. 평가용 데이터 개수와 예측 결과 데이터 개수 일치 : 2,716개
# 라이브러리
import pandas as pd
import numpy as np
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
# print(train.head(3), train.info(), sep="\n") # 6044
# print(test.head(3), test.info(), sep="\n")   # 2716

# 데이터 전처리
X = train.drop(columns=['Rented_Bike_Count'])
Y = train['Rented_Bike_Count']
X_all = pd.concat([X, test])
# print(X_all.head(3))
X_all['Date'] = X_all['Date'].astype('datetime64[ns]')
X_all['Year'] = X_all['Date'].dt.year
X_all['Month'] = X_all['Date'].dt.month
X_all['Day'] = X_all['Date'].dt.day
X_all['Weekday'] = X_all['Date'].dt.weekday
X_all = X_all.drop(columns=['Date'])
# print(X_all.info())
X_all['Seasons'] = LabelEncoder().fit_transform(X_all['Seasons'])
X_all = pd.get_dummies(X_all, drop_first=True, dtype='int32')
# print(X_all.head(3))

# 데이터 재분할
X = X_all.iloc[:len(X), :]
X_submission = X_all.iloc[len(X):, :]
# print(X.shape, X_submission.shape) # (6044, 16) (2716, 16)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=300)
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (4230, 16) (1814, 16) (4230,) (1814,)

# 모델사전 생성
models = {
    "Linear": Pipeline([
        ("scaler", MinMaxScaler()), ("model", LinearRegression())
    ]),
    "DecisionTree": Pipeline([
        ("model", DecisionTreeRegressor(max_depth=10, random_state=300))
    ]),
    "RandomForest": Pipeline([
        ("model", RandomForestRegressor(n_estimators=300, max_depth=10, random_state=300))
    ]),
    "AdaBoost": Pipeline([
        ("model", AdaBoostRegressor(n_estimators=300, random_state=300))
    ]),
    "Gradient": Pipeline([
        ("model", GradientBoostingRegressor(random_state=300))
    ])
}

# 성능평가 함수
def get_scores(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    y_pred1 = np.where(model.predict(x_train) < 0, abs(model.predict(x_train)), model.predict(x_train))
    y_pred2 = np.where(model.predict(x_test) < 0,  abs(model.predict(x_test)), model.predict(x_test))
    RMSE_train = np.sqrt(MSE(y_train, y_pred1))
    RMSE_test = np.sqrt(MSE(y_test, y_pred2))
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

# 모델적용, 예측산출
model = models[res.loc[0, "Model"]]
y_pred = model.predict(X_submission)

# 제출파일 생성
# pd.DataFrame({'pred': y_pred}).to_csv("result_day5_2nd.csv", index=False)

# 결과확인
# temp = pd.read_csv("result_day5_2nd.csv")
# print(Y[:len(test)].describe())
# print("=" * 35)
# print(temp['pred'].describe())

# 작업형 제3유형
# 다음과 같은 다중선형회귀 모형을 사용한 회귀모델을 만들고 결과를 확인합니다.
# Advertising.csv 데이터를 사용합니다.
# 모델 생성시 상수항(=절편)을 포함하도록 합니다.
# 종속변수 : sales
# 독립변수 : sales를 제외한 모든 변수
# 처음부터 순서대로 150개 데이터를 train, 나머지 50개를 test로 사용합니다.
# 모델 생성시 train 데이터를 사용합니다.
df = pd.read_csv('https://www.statlearning.com/s/Advertising.csv')
train = df.iloc[: 150, :]
test = df.iloc[150:, :]
# print(train.shape, test.shape) # (150, 5) (50, 5)
#5-1) 위의 조건에 맞게 OLS / OLS.from_formula 모델을 생성하고 summary를 출력한다.
from statsmodels.api import OLS
formula = "sales ~ TV + radio + newspaper"
model = OLS.from_formula(formula, train).fit()
# print(model.summary())

#5-2) 위에서 생성한 모델에서 적합된 모형 결정계수를 구해 반올림하여 소수점 아래 3자리까지 출력한다.
# print(round(model.rsquared, 3)) # 0.896 

#5-3) 위에서 생성한 모델에서 통계적으로 유의미한 변수는 몇 개인가? 유의수준 : 5%
# print(sum(model.pvalues[1:] <= 0.05)) # 2

#5-4) 위의 모델에서 통계적으로 유의한 변수들과 TV와 Radio의 교호작용항을 사용하여 새롭게 모델링하여 model2를 생성한다.
formula2 = "sales ~ TV + radio + TV:radio "
model2 = OLS.from_formula(formula2, train).fit()
# print(model2.summary())

#5-5) 다음의 값을 사용하여 예측값을 구해, 반올림하여 소수점아래 4자리까지 출력한다.
#     TV = 170.5, radio=13.2, newspaper=43.2
data = pd.DataFrame({'TV': [170.5], 'radio': [13.2], 'newspaper': [43.2]})
result = model2.predict(data)[0]
# print(round(result, 4)) # 12.8629

#5-6) 다음의 모델은 통계적 유의미해석에 사용하는 가설에서 모델은 귀무가설을 기각하는가 채택하는가?
# 귀무가설 : 모든 독립변수가 종속변수에 영향을 주지 않는다.
# 대립가설 : 적어도 하나의 독립변수가 종속변수에 영향을 준다.
# 신뢰수준 : 95%
# print("기각" if model2.f_pvalue <= 0.05 else "채택") # 채택

# 5-7) 모델에서 가장 영향력 있는 변수의 t-value를 구해, 반올림하여 소수점 아래 3자리까지 출력한다.
temp = model2.params[1:].abs().idxmax()
# print(round(model2.tvalues[temp], 3)) # 2.125

# 5-8) 모델에서 가장 유의미한 변수의 회귀계수를 구해, 반올림하여 소수점 아래 4자리까지 출력한다.
temp = model2.pvalues[1:].idxmin()
# print(round(model2.params[temp], 4)) # 0.0011

# 5-9) train 데이터를 사용하여 해당 모델의 예측값과 실제값의 피어슨(pearson) 상관계수를 구하여라.
# 결과는 반올림하여 소수점 아래 3자리까지 출력한다.
temp = pd.DataFrame({'y_true': train['sales'],
                     'y_pred': model2.predict(train)})
result = temp.corr().loc['y_true', 'y_pred']
# print(round(result, 3)) # 0.983

# 5-10) train 데이터를 사용하여 해당 모델의 예측값과 실제값의 스피어만(spearman) 상관계수를 구하여라.
# 결과는 반올림하여 소수점 아래 3자리까지 출력한다.
temp = pd.DataFrame({'y_true': train['sales'],
                     'y_pred': model2.predict(train)})
result = temp.corr('spearman').loc['y_true', 'y_pred']
# print(round(result, 3)) # 0.994

# 5-11) test 데이터를 사용하여 rmse를 구해, 반올림하여 소수점 아래 4자리까지 출력한다.
from sklearn.metrics import root_mean_squared_error as rmse
# print(round(rmse(test['sales'], model2.predict(test)), 4)) # 0.8665

# 5-12) train 데이터를 사용하여 잔차를 구하고, 잔차의 IQR을 구해, 반올림하여 소수점 아래 4자리까지 출력한다.
residual = train['sales'] - model2.predict(train)
Q1, Q3 = residual.quantile([0.25, 0.75])
# print(round(Q3-Q1, 4)) # 0.9658

# 5-13) 통계적으로 가장 유의하지 않은 변수의 표준오차(Standard Error)를 소수점 아래 4자리까지 출력한다.
temp = model2.pvalues[1:].idxmax()
# print(round(model2.bse[temp], 4)) # 0.0104