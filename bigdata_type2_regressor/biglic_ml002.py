
# Kaggle House Price Regression Dataset
# 이 데이터셋은 주택 가격 예측과 같은 회귀 문제를 연습하려는 초보자들을 위해 설계된 자료입니다.
# 총 1,000개의 행으로 구성되어 있으며, 각 행은 하나의 주택과 그 가격에 영향을 주는 다양한 속성을 나타냅니다.
# 이 데이터셋은 기초부터 중급 수준의 회귀 모델링 기법을 학습하고 실습하는 데 적합합니다.
# 주요 변수 설명
# 1. Square_Footage (주택 면적, ft²): 집의 전체 크기(면적). 큰 면적일수록 일반적으로 가격(House_Price)이 높음.
# 2. Num_Bedrooms (침실 개수): 침실 수. 침실이 많을수록 가치 상승 가능성 높음. 하지만 “단순히 많다고 무조건 가격 상승”은 아님
#   (과도한 침실 수는 가격 효율성이 떨어질 수 있음).
# 3. Num_Bathrooms (욕실 개수): 욕실 수. 욕실이 많으면 생활 편의성이 좋아져 가격이 상승하는 경향.
# 4. Year_Built (건축 연도): 지어진 연도. 오래된 집은 마모·노후화로 인해 가격이 낮을 수 있음.
#    하지만 일부 지역에서는 오래된 집이 오히려 역사적 가치(heritage) 때문에 고가일 수도 있음.
# 5. Lot_Size (대지 크기, acres): 집이 지어진 토지의 크기. 큰 대지는 희소성 때문에 가격에 긍정적 영향을 줌.
# 6. Garage_Size (차고 크기, 수용 가능 차량 수): 차고 크기 (예: 1-car, 2-car, 3-car). 차량 수용 능력이 클수록 가격 상승 요인.
# 7. Neighborhood_Quality (이웃 환경 지수, 1~10점): 동네/지역 품질을 평가한 점수. 교육, 안전, 접근성 등 반영.
#    높은 점수일수록 주택 가격에 매우 큰 영향.
#  타겟 변수 (종속 변수, Label)
#  House_Price (주택 가격, 연속형 변수): 예측하고자 하는 최종 값. 위의 독립 변수들과 선형적 또는 비선형적으로 관계를 가짐.

# 라이브러리 import
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.metrics import root_mean_squared_error as RMSE
from sklearn.metrics import root_mean_squared_log_error as RMSLE

pd.set_option("display.max_rows", 100)
pd.set_option("display.float_format", '{:.2f}'.format)

# 사용자 평가함수
def get_scores(model, x_train, x_test, y_train, y_test):
    A = model.score(x_train, y_train)
    B = model.score(x_test, y_test)
    y_pred1 = model.predict(x_train)
    y_pred2 = model.predict(x_test)
    C = RMSE(y_train, y_pred1)
    D = RMSE(y_test, y_pred2)
    return f"r2: {A:.4f}, {B:.4f}, RMSE: {C:.4f}, {D:.4f}"

# 데이터 불러오기
path = "dataset/houseprice/"
XY = pd.read_csv(path + "house_price_dataset.csv")
# print(X_all.columns)
# ['Square_Footage','Num_Bedrooms','Num_Bathrooms','Year_Built',
#  'Lot_Size','Garage_Size','Neighborhood_Quality','House_Price']
# print(X_all.shape) # (1000, 8)
X = XY.iloc[:800].drop(columns=['House_Price'])
Y = XY.iloc[:800]['House_Price']
X_submission = XY.iloc[800:].drop(columns=['House_Price'])
# print(X.shape, Y.shape, X_submission.shape) # (800, 7) (800,) (200, 7)

# 데이터 전처리(결측치 처리, Encoding, Scaling)
X_all = pd.concat([X, X_submission])
# print(X_all.isna().sum()) # 결측치 없음

# 범주형 객체 확인 및 전처리
# print(X_all.select_dtypes(include='object').nunique()) # 범주형 컬럼 없음

# 데이터 표준화 전처리
scaler = StandardScaler().fit_transform(X_all)
X_all = pd.DataFrame(scaler, columns=X_all.columns)

# 데이터 재분할
X = X_all.iloc[:800]
X_submission = X_all.iloc[800:]
# print(X.shape, Y.shape, X_submission.shape) # (800, 7) (800,) (200, 7)

# 모델링
temp = train_test_split(X, Y, test_size=0.2, random_state=1234)
x_train, x_test, y_train, y_test = temp
# print([x.shape for x in temp]) # (560, 7) (240, 7) (560,) (240,)

model1 = LinearRegression().fit(x_train, y_train)
# print(get_scores(model1, x_train, x_test, y_train, y_test)) # r2: 0.9985, 0.9986, RMSE: 9897.6078, 9707.1831

# model2 = DecisionTreeRegressor(random_state=0).fit(x_train, y_train)
# print(get_scores(model2, x_train, x_test, y_train, y_test)) # r2: 1.0000, 0.9868, RMSE: 0.0000, 29274.9033
# print(model2.get_depth()) # 16
# for d in range(3, 8):
#     model2 = DecisionTreeRegressor(max_depth=d, random_state=0).fit(x_train, y_train)
#     print(d, get_scores(model2, x_train, x_test, y_train, y_test)) # 5 r2: 0.9893, 0.9823, RMSE: 26376.6397, 34003.5070
model2 = DecisionTreeRegressor(max_depth=5, random_state=0).fit(x_train, y_train)
# print(get_scores(model2, x_train, x_test, y_train, y_test)) # r2: 0.9893, 0.9823, RMSE: 26376.6397, 34003.5070

# model3 = RandomForestRegressor(random_state=0).fit(x_train, y_train)
# print(get_scores(model3, x_train, x_test, y_train, y_test)) # r2: 0.9990, 0.9932, RMSE: 7958.2733, 21043.7581
# for d in range(5, 16):
#     model3 = RandomForestRegressor(max_depth=d, random_state=0).fit(x_train, y_train)
#     print(d, get_scores(model3, x_train, x_test, y_train, y_test)) # 6 r2: 0.9962, 0.9912, RMSE: 15723.1239, 23995.5596
model3 = RandomForestRegressor(max_depth=6, random_state=0).fit(x_train, y_train)
# print(get_scores(model3, x_train, x_test, y_train, y_test)) # r2: 0.9962, 0.9912, RMSE: 15723.1239, 23995.5596

model4 = AdaBoostRegressor(random_state=0).fit(x_train, y_train)
# print(get_scores(model4, x_train, x_test, y_train, y_test)) # r2: 0.9870, 0.9841, RMSE: 29058.8250, 32176.4210

model5 = GradientBoostingRegressor(random_state=0).fit(x_train, y_train)
# print(get_scores(model5, x_train, x_test, y_train, y_test)) # r2: 0.9987, 0.9960, RMSE: 9131.5354, 16142.8431

# 예측값 구하고 파일생성
final_model = model2
y_pred = final_model.predict(X_submission)
pd.DataFrame({'pred': y_pred}).to_csv("result_house_price.csv", index=False)

# 결과확인
temp = pd.read_csv("result_house_price.csv")
print(temp['pred'].describe())
print("=" * 30)
print(Y.describe())




