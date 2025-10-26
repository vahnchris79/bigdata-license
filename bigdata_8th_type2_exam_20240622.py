
# 기출문제 제8회 2024년 6월 22일 시행

import pandas as pd
path = "https://raw.githubusercontent.com/YoungjinBD/data/main/exam/"
pd.set_option('display.width', 120)
pd.set_option('display.max_row', 100)

# 작업형 제2유형
# 제공된 학습용 데이터(8_2_train.csv)는 자전거 대여와 관련된 날짜별 정보와 해당 날짜의 총 대여 건수(count)를 포함하고 있다.
# 학습용 데이터를 활용하여 자전거 총 대여 건수(count)를 예측하는 회귀모델을 개발하고, 성능이 가장 우수한 모델을 평가용 데이터
# (8_2_test.csv)에 적용하여 예측 결과를 제출하시오.
# 모델 성능 지표: MAE(mean_absolute Error)
# Data description
# ID: 고유식별자, holiday: 공휴일 여부, workingday: 평일 여부, weather: 날씨 상황, temp: 실제 기온, atemp: 체감 온도
# humidity: 습도, windspeed: 풍속, count: 자전거 총 대여 건수
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as MAE

# 모델 성능지표 함수
def get_scores(model, x_train, x_test, y_train, y_test):
    A = model.score(x_train, y_train)
    B = model.score(x_test, y_test)
    y_pred1 = model.predict(x_train)
    y_pred2 = model.predict(x_test)
    C = MAE(y_train, y_pred1)
    D = MAE(y_test, y_pred2)
    return f"r2: {A:.4f}, {B:.4f}, MAE: {C:.4f}, {D:.4f}"

# 데이터 불러오기
train = pd.read_csv(path + "8_2_train.csv")
test = pd.read_csv(path + "8_2_test.csv")
# print(train.shape, test.shape) # (378, 9) (166, 9)
# print(train.head(), test.head(), sep="\n")

# 데이터 전처리
X = train.drop(columns=['count'])
Y = train['count']
X_submission = test.drop(columns=['count'])
X_all = pd.concat([X, X_submission])
# print(X_all.shape) # (544, 7)
# print(X_all.head())
cols_obj = X_all.select_dtypes(include='object').columns
for col in cols_obj:
    X_all[col] = LabelEncoder().fit_transform(X_all[col])
# print(X_all.head())

# 데이터 나누기
X = X_all.iloc[:378,]
X_submission = X_all.iloc[378:,]
# print(X.shape, Y.shape, X_submission.shape)
temp = train_test_split(X, Y, test_size=0.3, random_state=1234)
x_train, x_test, y_train, y_test = temp
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (264, 7) (114, 7) (264,) (114,)

# 모델 적합
model1 = LinearRegression().fit(x_train, y_train)
# print(get_scores(model1, x_train, x_test, y_train, y_test)) # r2: 0.3534, 0.1101, MAE: 110.0202, 125.9141

# model2 = DecisionTreeRegressor(random_state=0).fit(x_train, y_train)
# print(get_scores(model2, x_train, x_test, y_train, y_test)) # r2: 1.0000, -0.4348, MAE: 0.0000, 146.2982
# print(model2.get_depth()) # 16
# for d in range(3, 16):
#     model2 = DecisionTreeRegressor(max_depth=d, random_state=0).fit(x_train, y_train)
#     print(d, get_scores(model2, x_train, x_test, y_train, y_test)) # 7 r2: 0.8409, -0.6970, MAE: 45.7624, 164.5734
model2 = DecisionTreeRegressor(max_depth=7, random_state=0).fit(x_train, y_train)
# print(get_scores(model2, x_train, x_test, y_train, y_test)) # r2: 0.8409, -0.6970, MAE: 45.7624, 164.5734

# model3 = RandomForestRegressor(random_state=1234).fit(x_train, y_train)
# print(get_scores(model3, x_train, x_test, y_train, y_test)) # r2: 0.9129, 0.1070, MAE: 40.1432, 121.1999
# for d in range(3, 16):
#     model3 = RandomForestRegressor(max_depth=d, random_state=1234).fit(x_train, y_train)
#     print(d, get_scores(model3, x_train, x_test, y_train, y_test)) # 8 r2: 0.8846, 0.1174, MAE: 47.5393, 120.5448
model3 = RandomForestRegressor(max_depth=8, random_state=1234).fit(x_train, y_train)
# print(get_scores(model3, x_train, x_test, y_train, y_test)) # r2: 0.8846, 0.1174, MAE: 47.5393, 120.5448

model4 = AdaBoostRegressor(random_state=0).fit(x_train, y_train)
# print(get_scores(model4, x_train, x_test, y_train, y_test)) # r2: 0.5436, 0.0606, MAE: 105.7077, 130.1185

model5 = GradientBoostingRegressor(random_state=0).fit(x_train, y_train)
# print(get_scores(model5, x_train, x_test, y_train, y_test)) # r2: 0.8255, 0.0937, MAE: 55.3631, 124.8760

# 예측값 구하기
final_model = model3
y_pred = final_model.predict(X_submission)
pd.DataFrame({'ID': X_submission['ID'], 'pred': y_pred}).to_csv("result_type2_20240622.csv", index=False)

# 예측결과 비율확인
temp = pd.read_csv("result_type2_20240622.csv")
print("=" * 10)
print(temp['pred'].describe())
