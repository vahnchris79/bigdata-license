
# Big Mart Sales Prediction
# URL: https://www.kaggle.com/api/v1/datasets/download/akashdeepkuila/big-mart-sales

# 목표: 각 상품(Item)이 특정 매장(Outlet)에서 기록한 **판매액(Item_Outlet_Sales)**을 예측하는 것입니다.

# 데이터 구성
# Train.csv (8523행): 상품, 매장, 판매액 정보 포함
# Test.csv (5681행): 상품과 매장 정보는 같지만 판매액(Item_Outlet_Sales)이 없음
# 요구사항: 테스트 데이터의 판매액을 예측하는 회귀(Regression) 모델 생성
# 독립변수: ProductID, Weight, FatContent, ProductVisibility, ProductType, MRP, OutletID,
#           EstablishmentYear, OutletSize, LocationType, OutletType
# 타깃변수: OutletSales

# 라이브러리 import 
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error as RMSE

pd.set_option("display.max_row", 100)

# 회귀모델 평가함수
def get_scores(model, x_train, x_test, y_train, y_test):
    A = model.score(x_train, y_train)
    B = model.score(x_test, y_test)
    y_pred1 = model.predict(x_train)
    y_pred2 = model.predict(x_test)
    C = RMSE(y_train, y_pred1)
    D = RMSE(y_test, y_pred2)
    return f"r2: {A:.4f}, {B:.4f}, RMSE: {C:.4f}, {D:.4f}"

# 데이터 불러오기
path = "dataset/bigmart_sales/"
XY = pd.read_csv(path + "Train-Set.csv")
X_submission = pd.read_csv(path + "Test-Set.csv")
# print(XY.shape, X_submission.shape) # (8523, 12) (5681, 11)
#print(XY.head(3), X_submission.head(3))

# 탐색적 데이터 분석
#print(XY.info(), X_submission.info())
X = XY.drop(columns=['OutletSales'])
Y = XY['OutletSales']
# print(X.shape, Y.shape) # (8523, 11) (8523,)
#print(Y.dtypes)
#print(Y.value_counts())
X_all = pd.concat([X, X_submission])
# print(X_all.isna().sum()) # 결측치: Weight 2439, OutletSize 4016

# Weight는 평균으로, OutletSize는 OutletType의 빈도로 대체
X_all['Weight'] = X_all['Weight'].fillna(X_all['Weight'].mean())
X_all['OutletSize'] = X_all.groupby('OutletType')['OutletSize'].transform(lambda x: x.fillna(x.mode()[0]))
# print(X_all.isna().sum())
# print(X_all.shape) # 14204, 11

# 범주형 컬럼에 대한 전처리(문자형 -> 수치형으로 변환)
X_all = X_all.drop(columns=['ProductID']) # ProductID 제외
column_obj = X_all.select_dtypes(include='object').columns 
# print(column_obj) # ['FatContent', 'ProductType', 'OutletID', 'OutletSize', 'LocationType', 'OutletType']
for col in column_obj:
    X_all[col] = LabelEncoder().fit_transform(X_all[col])
# print(X_all.head())
# 수치형 컬럼들의 단위를 통일하기 위해서 StandardScaler 적용
temp = StandardScaler().fit_transform(X_all)
X_all = pd.DataFrame(temp, columns = X_all.columns)
# print(X_all.head(3))

# 데이터 재분할
X = X_all.iloc[:8523]
X_submission = X_all.iloc[8523:]
# print(X.shape, Y.shape, X_submission.shape) # (8523, 10) (8523,) (5681, 10)
temp = train_test_split(X, Y, test_size=0.3, random_state=42)
x_train, x_test, y_train, y_test = temp
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (5966, 10) (2557, 10) (5966,) (2557,)

# 모델링
# LinearRegressor
model1 = LinearRegression().fit(x_train, y_train)
# print(get_scores(model1, x_train, x_test, y_train, y_test)) # r2: 0.5010, 0.5084, RMSE: 1214.8359, 1173.4892

# DecisionTreeRegressor
# model2 = DecisionTreeRegressor(random_state=0).fit(x_train, y_train)
# print(get_scores(model2, x_train, x_test, y_train, y_test)) # r2: 1.0000, 0.1920, RMSE: 0.0000, 1504.4200
# print(model2.get_depth()) # 33
# for d in range(3, 11):
    # model2 = DecisionTreeRegressor(max_depth=d, random_state=0).fit(x_train, y_train)
    # print(d, get_scores(model2, x_train, x_test, y_train, y_test)) 
    # # 3 r2: 0.5244, 0.5247, RMSE: 1186.0137, 1153.8305
model2 = DecisionTreeRegressor(max_depth=3, random_state=0).fit(x_train, y_train)
# print(get_scores(model2, x_train, x_test, y_train, y_test)) # r2: 0.5244, 0.5247, RMSE: 1186.0137, 1153.8305

# RandomForestRegressor
# model3 = RandomForestRegressor(random_state=0).fit(x_train, y_train)
# print(get_scores(model3, x_train, x_test, y_train, y_test)) # r2: 0.9370, 0.5537, RMSE: 431.7724, 1118.0226
# for d in range(3, 11):
    # model3 = RandomForestRegressor(max_depth=d, random_state=0).fit(x_train, y_train)
    # print(d, get_scores(model3, x_train, x_test, y_train, y_test))
    # 3 r2: 0.5451, 0.5479, RMSE: 1159.9068, 1125.3379
model3 = RandomForestRegressor(max_depth=3, random_state=0).fit(x_train, y_train)
# print(get_scores(model3, x_train, x_test, y_train, y_test)) # r2: 0.5451, 0.5479, RMSE: 1159.9068, 1125.3379

# AdaBoostRegressor
model4 = AdaBoostRegressor(random_state=0).fit(x_train, y_train)
# print(get_scores(model4, x_train, x_test, y_train, y_test)) # r2: 0.5251, 0.5112, RMSE: 1185.1631, 1170.1560

# GradientBoostingRegressor
model5 = GradientBoostingRegressor(random_state=0).fit(x_train, y_train)
# print(get_scores(model5, x_train, x_test, y_train, y_test)) # r2: 0.6397, 0.5950, RMSE: 1032.2970, 1065.1440

# 모델 선정, 예측값 산출, 파일 저장
final_model = model4
y_pred = final_model.predict(X_submission)
pd.DataFrame({'pred':  y_pred}).to_csv("result_bigmart_sales.csv", index=False)

# 제출파일 확인
temp = pd.read_csv("result_bigmart_sales.csv")
print(temp['pred'].describe())
print(Y.describe())