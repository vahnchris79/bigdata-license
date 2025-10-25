
# 기출문제 제6회
import pandas as pd
pd.set_option('display.max_row', None)
pd.set_option('display.max_column', None)
import warnings
warnings.filterwarnings('ignore')

path = "https://raw.githubusercontent.com/YoungjinBD/data/main/exam/"

# 작업형 제2유형
# 제공된 학습용 데이터(6_2_train.csv)는 환자의 나이, 성별, 체질량지수(BMI), 혈당 및 콜레스테롤 수치
# 등의 정보를 포함하고 있으며, 이로부터 이완기 혈압(DBP)을 예측하고자 한다.
# 학습용 데이터를 활용하여 환자의 이완기 혈압(DBP)를 예측하는 회귀모델을 개발하고, 성능이 가장
# 우수한 모델을 평가용 데이터(6_2_test.csv)에 적용하여 예측 결과를 제출하시오.
# 모델 성능 지표: RMSE(Root Mean Squared Error)
# level_0 컬럼은 인덱스 초기화 과정에서 생성된 것으로 분석 시 제외

# 데이터 확인
train = pd.read_csv(path + "6_2_train.csv")
test = pd.read_csv(path + "6_2_test.csv")
# print(train.head(), train.info(), train.shape, sep="\n") # (301, 12) 성별 11행 결측
# print(test.head(), test.info(), test.shape, sep="\n") # (129, 12) 성별 7행 결측

# 결측치 제거
train = train.dropna()
test = test.dropna()
# print(train.head(), train.info(), train.shape, sep="\n") # (290, 12)
# print(test.head(), test.info(), test.shape, sep="\n") # (122, 12)

# 라이브러리
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error as RMSE

# 모델 성능지표 함수
def get_scores(model, x_train, x_test, y_train, y_test):
    A = model.score(x_train, y_train)
    B = model.score(x_test, y_test)
    y_pred1 = model.predict(x_train)
    y_pred2 = model.predict(x_test)
    C = RMSE(y_train, y_pred1)
    D = RMSE(y_test, y_pred2)
    return f"r2: {A:.4f}, {B:.4f}, RMSE: {C:.4f}, {D:.4f}"

# 전처리를 위한 데이터 정의
X = train.drop(columns=['level_0', 'DBP']).copy()
Y = train['DBP']
X_submission = test.drop(columns=['level_0', 'DBP']).copy()
# print(X.shape, Y.shape, X_submission.shape) # (290, 10) (290,) (122, 10)

# Y의 분포확인
# print(Y.describe()) # mean 76.489655, std 11.449070, min 52.0, max 116.0

# 데이터 전처리
X_all = pd.concat([X, X_submission])
# print(X_all.head())

# 성별 Encoding
X_all['Gender'] = LabelEncoder().fit_transform(X_all['Gender'])

# 스케일링
temp = MinMaxScaler().fit_transform(X_all)
X_all = pd.DataFrame(temp, columns=X_all.columns)
# print(X_all)

# 데이터 분할
X = X_all.iloc[:290, :].copy()
X_submission = X_all.iloc[290:, :].copy()
# print(X.shape, Y.shape, X_submission.shape) # (290, 10) (290,) (122, 10)
temp = train_test_split(X, Y, test_size=0.3, random_state=42)
x_train, x_test, y_train, y_test = temp
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (203, 10) (87, 10) (203,) (87,)

# 모델 생성
model1 = LinearRegression().fit(x_train, y_train)
# print(get_scores(model1, x_train, x_test, y_train, y_test)) # r2: 0.1794, -0.1088, RMSE: 10.8231, 10.6290

# model2 = DecisionTreeRegressor(random_state=1234).fit(x_train, y_train)
# print(get_scores(model2, x_train, x_test, y_train, y_test)) # r2: 1.0000, -1.6568, RMSE: 0.0000, 16.4533
# print(model2.get_depth()) # 19
# for d in range(3, 16):
#     model2 = DecisionTreeRegressor(max_depth=d, random_state=1234).fit(x_train, y_train)
#     print(d, get_scores(model2, x_train, x_test, y_train, y_test)) # 3 r2: 0.3189, -0.3275, RMSE: 9.8606, 11.6301
model2 = DecisionTreeRegressor(max_depth=3, random_state=1234).fit(x_train, y_train)
# print(get_scores(model2, x_train, x_test, y_train, y_test)) # r2: 0.3189, -0.3275, RMSE: 9.8606, 11.6301

# model3 = RandomForestRegressor(random_state=0).fit(x_train, y_train)
# print(get_scores(model3, x_train, x_test, y_train, y_test)) # r2: 0.8663, -0.0602, RMSE: 4.3679, 10.3936
# for d in range(3, 16):
#     model3 = RandomForestRegressor(max_depth=d, random_state=0).fit(x_train, y_train)
#     print(d, get_scores(model3, x_train, x_test, y_train, y_test)) # 3 r2: 0.3760, -0.0278, RMSE: 9.4377, 10.2335
model3 = RandomForestRegressor(max_depth=3, random_state=0).fit(x_train, y_train)
# print(get_scores(model3, x_train, x_test, y_train, y_test)) # r2: 0.3760, -0.0278, RMSE: 9.4377, 10.2335

model4 = AdaBoostRegressor(random_state=0).fit(x_train, y_train)
# print(get_scores(model4, x_train, x_test, y_train, y_test)) # r2: 0.5223, -0.0341, RMSE: 8.2577, 10.2648

model5 = GradientBoostingRegressor(random_state=0).fit(x_train, y_train)
# print(get_scores(model5, x_train, x_test, y_train, y_test)) # r2: 0.8599, -0.2352, RMSE: 4.4716, 11.2188

# 예측값 생성
final_model = model1
y_pred = final_model.predict(X_submission)
pd.DataFrame({'ID': test['ID'], 'pred': y_pred}).to_csv("result.csv", index=False)

# 비율 확인
result = pd.read_csv("result.csv")
print(result['pred'].describe())
print("=" * 30)
print(Y[:122].describe())