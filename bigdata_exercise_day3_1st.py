
# 작업형 제1유형
import pandas as pd
pd.set_option('display.width', 120)
path1 = "https://raw.githubusercontent.com/Soyoung-Yoon/bigdata/main/"
path2 = "https://raw.githubusercontent.com/Soyoung-Yoon/data_01/main/"

# 3-1) 중간 수준 출력 차량의 지역별 연비 특성 분석
# 차량의 마력(horsepower)과 연비(mpg)의 관련성을 알기위해 중간 성능대 차량의 지역별 연비 경향을 분석한다.
# 다음 절차에 따라 문제를 해결하시오.
# - horsepower 값이 전체에서 하위 20% 이상, 상위 20% 이하인 차량만을 대상으로 한다. (즉, horsepower가 전체의 20% 분위수 이상, 80% 분위수 이하인 데이터만 필터링)
# - 이후, 필터링된 데이터에서 origin별로 연비(mpg)의 중앙값을 계산하고, 그 중 가장 큰 값과 가장 작은 값의 차이를 반올림하여 정수로 출력한다.
df1 = pd.read_csv(path2 + "mpg02.csv")
# horsepower 값이 전체에서 하위 20% 이상, 상위 20% 이하인 차량만을 대상으로 한다.
higher, lower = df1['horsepower'].quantile([0.8, 0.2])
df1 = df1[(df1['horsepower']>=lower) & (df1['horsepower'] <= higher)]
# 터링된 데이터에서 origin별로 연비(mpg)의 중앙값을 계산
mpg_median = df1.groupby('origin')['mpg'].median()
# print(round(mpg_median.max() - mpg_median.min())) # 6

# 3-2) 이상적인 컷의 고캐럿 다이아몬드 가격 분포 분석

# 다이아몬드의 cut 등급과 carat(중량)은 가격에 큰 영향을 미칩니다.
# 이 문제에서는 이상적인 컷(Ideal)과 고캐럿 조건을 만족하는 다이아몬드들의 가격 분포를 분석합니다.
# 다음 작업을 수행하세요.
# - diamonds.csv를 사용합니다.
# - 다이아몬드 중에서 cut 등급이 Ideal이고, carat이 전체 다이아몬드 평균보다 큰 데이터만을 필터링한다.
# - 이렇게 필터링된 데이터에서 price의 사분위 범위 (IQR, Interquartile Range) 를 결과로 구합니다.
# - IQR = Q3 - Q1
#     - Q1: 1사분위수 (25%), Q3: 3사분위수 (75%)
# - 결과를 정수로 출력합니다.
df2 = pd.read_csv(path1 + "diamonds.csv")
# 다이아몬드 중에서 cut 등급이 Ideal이고, carat이 전체 다이아몬드 평균보다 큰 데이터만을 필터링
df2 = df2[(df2['cut']=='Ideal') & (df2['carat'] > df2['carat'].mean())].copy()
# 필터링된 데이터에서 price의 사분위 범위 (IQR, Interquartile Range) 를 결과로 구합니다.
Q1, Q3 = df2['price'].quantile([0.25, 0.75])
# print(int(Q3 - Q1)) # 5336

# 3-3) 붓꽃의 품종별 특성 분석
# 붓꽃(iris) 품종은 꽃받침 길이(sepal_length)와 꽃잎 너비(petal_width)에 따라 구분되는 특성이 있습니다.
# 이 문제에서는 두 가지 기준을 동시에 만족하는 표본의 수를 알아봅니다.
# 다음 작업을 수행하세요.
# - 붓꽃 데이터셋(iris01.csv)을 사용합니다.
# - 아래의 두 조건을 모두 만족하는 표본의 개수를 구합니다.
# - 조건1: sepal_length가 해당 품종(species)별 평균보다 큰 데이터
# - 조건2: petal_width가 0.2 이상인 데이터
# - 단, species는 'setosa', 'versicolor', 'virginica'의 3가지 종류가 있음
df3 = pd.read_csv(path1 + "iris01.csv")
# print(df3.head(3))
# 조건1: sepal_length가 해당 품종(species)별 평균보다 큰 데이터
cond1 = (df3.groupby('species')['sepal_length'].transform('mean'))
# 조건2: petal_width가 0.2 이상인 데이터
cond2 = (df3['petal_width'] >= 0.2)
# print(len(df3[(df3['sepal_length'] > cond1) & (cond2)])) # 67

# 3-4) 1980년대 이후 저실린더 차량의 국가별 분포 분석
# 1980년 이후에는 연료 효율성을 고려한 저배기량 차량이 증가하였습니다.
# 이 문제에서는 해당 연도별 평균보다 적은 실린더 수(cylinders)를 가진 차량이 어느 국가(origin)에서 가장 많이 생산되었는지를 파악하고자 합니다.
# 다음 작업을 수행하세요.
# - mpg.csv 파일을 사용합니다.
# - model_year가 80 이상인 차량만 필터링합니다.
# - 위 차량 중에서, 연도(model_year)별 cylinders 수가 평균 미만인 차량만 필터링합니다.
# - 위 조건을 만족하는 차량들의 origin 중에서, 가장 빈도가 높은(origin의 최빈값) 국가 이름을 결과로 구합니다.
# - 결과는 문자열로 출력합니다. (대소문자 변경 없이 입력)
df4 = pd.read_csv(path1 + "mpg.csv")
# print(df4.head(3))
# model_year가 80 이상인 차량만 필터링합니다.
df4 = df4[df4['model_year'] >= 80].copy()
# 위 차량 중에서, 연도(model_year)별 cylinders 수가 평균 미만인 차량만 필터링
cond = df4.groupby('model_year')['cylinders'].transform('mean')
df4 = df4[df4['cylinders'] < cond].copy()
# 위 조건을 만족하는 차량들의 origin 중에서, 가장 빈도가 높은(origin의 최빈값) 국가 이름
df4['cnt'] = df4['origin']
# print(df4.groupby('origin')['cnt'].count().idxmax()) #Japan

# 3-5) 성별 대비 고요금 승객의 등급별 요금 차이 분석

# 탑승 요금(Fare)은 성별과 좌석 등급(Pclass)에 따라 차이가 날 수 있습니다.
# 이 문제에서는 성별 평균보다 더 높은 요금을 낸 승객들의 등급별 요금 차이를 분석하고자 합니다.
# 다음 작업을 수행하세요.
# - titanic.csv를 사용합니다.
# - Fare가 자신과 동일한 성별(Gender)의 평균 요금보다 높은 승객만 필터링합니다.
# - 위 조건을 만족하는 승객들을 Pclass(좌석 등급) 기준으로 그룹화하여, 각 등급별 Fare의 평균을 계산합니다.
# - 이 중 가장 높은 등급별 평균과 가장 낮은 등급별 평균의 차이를 계산하여 결과로 구합니다.
# - 최종 결과는 반올림하여 소수점 아래 3자리까지 출력합니다.
df5 = pd.read_csv(path1 + "titanic_dataq.csv")
# Fare가 자신과 동일한 성별(Gender)의 평균 요금보다 높은 승객만 필터링
cond1 = df5.groupby('Gender')['Fare'].transform('mean')
df5 = df5[df5['Fare'] > cond1].copy()
# 위 조건을 만족하는 승객들을 Pclass(좌석 등급) 기준으로 그룹화하여, 각 등급별 Fare의 평균을 계산
max_class = df5.groupby('Pclass', as_index=False)['Fare'].mean()['Fare'].max()
min_class = df5.groupby('Pclass', as_index=False)['Fare'].mean()['Fare'].min()
# print(round(max_class - min_class, 3)) # 52.015

# 작업형 제2유형
# Seoul Bike Sharing Dataset 데이터셋
# 서울시의 시간대별 자전거 대여 데이터를 포함하고 있으며, 날씨와 계절 요인이 자전거 대여량에 어떤 영향을 미치는지를 분석할 수 있는 시계열형 예측 데이터셋입니다.
# 제공된 학습용 데이터(bike_train.csv)를 이용하여 대여 자전거 대수(Rented_Bike_Count)를 예측하는 모델을 개발하고, 
# 개발한 모델에 기반하여 평가용 데이터(bike_test.csv)에 적용하여 얻은 대여 자전거 대수 예측 값을 아래 [제출형식]에 따라 csv 파일로 생성하여 제출하시오.
# - 예측 결과는 RMSE(Root Mean Squared Error) 평가지표에 따라 평가함
# - 성능이 우수한 예측 모델을 구축하기 위해서는 데이터 정제, Feature Engineering, 하이퍼 파라미터(hyper parameter) 최적화, 모델 비교 등이 필요할 수 있음.
# 다만, 과적합에 유의하여야 함
# [[제출 형식]]
# - 가. CSV 파일명: result.csv(파일명에 디렉토리/폴더 지정불가
# - 나. 예측 칼럼명 : pred
# - 다. 제출 칼럼 개수 : pred 칼럼 1개
# - 라. 평가용 데이터 개수와 예측 결과 데이터 개수 일치 : 2,716개
# [[제공 데이터]]
# - 데이터 목록
# - bike_train.csv : 학습용 데이터, 6,044개
# - bike_test.csv : 평가용 데이터, 2,716개
# - 평가용 데이터는 'Rented_Bike_Count' 칼럼 미제공

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
from sklearn.metrics import mean_squared_log_error as MSLE

# 데이터 불러오기
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
X_all['Weekday'] = X_all['Date'].dt.weekday
X_all = X_all.drop(columns=['Date'])
cols_obj = X_all.select_dtypes(include='object').columns
# print(cols_obj)
X_all['Seasons'] = LabelEncoder().fit_transform(X_all['Seasons'])
X_all['Holiday'] = LabelEncoder().fit_transform(X_all['Holiday'])
X_all['Functioning_Day'] = LabelEncoder().fit_transform(X_all['Functioning_Day'])
# X_all = pd.get_dummies(X_all, drop_first=True, dtype='int32')
# print(X_all.head(3))

# 데이터 분할
X = X_all.iloc[:len(X), :]
X_submission = X_all.iloc[len(X):, :]
# print(X.shape, X_submission.shape) # (6044, 16) (2716, 16)
temp = train_test_split(X, Y, test_size=0.2, random_state=1234)
x_train, x_test, y_train, y_test = temp
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (4230, 16) (1814, 16) (4230,) (1814,)

# 파이프라인 모델사전
models = {
    "Linear": Pipeline([
        ("scaler", MinMaxScaler()), ("model", LinearRegression())
    ]),
    "DecisionTree": Pipeline([
        ("model", DecisionTreeRegressor(max_depth=10, random_state=1234))
    ]),
    "RandomForest": Pipeline([
        ("model", RandomForestRegressor(n_estimators=200, max_depth=10, random_state=1234))
    ]),
    "AdaBoost": Pipeline([
        ("model", AdaBoostRegressor(n_estimators=200, random_state=1234))
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
    RMSLE_train = MSLE(y_train, y_pred1) ** 0.5
    RMSLE_test = MSLE(y_test, y_pred2) ** 0.5
    return model, RMSE_train, RMSE_test, RMSLE_train, RMSLE_test

# 모델별 성능평가
results = []
for name, model in models.items():
    model, RMSE_train, RMSE_test, RMSLE_train, RMSLE_test = get_scores(model, x_train, x_test, y_train, y_test)
    results.append({
        "Model": name, "RMSE_train": f"{RMSE_train:.4f}", "RMSE_test": f"{RMSE_test:.4f}", "RMSLE_train": f"{RMSLE_train:.4f}", "RMSLE_test": f"{RMSLE_test:.4f}"
    })
res = pd.DataFrame(results).sort_values("RMSE_test").reset_index(drop=True)
# print(res)

# 모델적용
model = models[res.loc[0, "Model"]]
y_pred = model.predict(X_submission)

# 제출파일 생성
# pd.DataFrame({'pred': y_pred}).to_csv("result_day3_1st.csv", index=False)

# 결과확인
# temp = pd.read_csv("result_day3_1st.csv")
# print(Y[:len(X_submission)].describe())
# print("=" * 35)
# print(temp['pred'].describe())

# 작업형 제3유형
# Day3. 로지스틱회귀(타이타닉 데이터셋)
import pandas as pd
pd.set_option('display.width', 120)
df = pd.read_csv(path2 + "titanic_03.csv")
# print(df.head(3))

# 다음과 같은 로지스틱회귀 모형을 사용한 분류모델을 만들고 결과를 확인합니다.
# - titanic_03.csv 데이터를 사용합니다.
# - 모델 생성시 상수항(=절편)을 포함하도록 하며, 규제는 사용하지 않습니다.
# - 종속변수 : Survived
# - 독립변수 : Pclass, Age, Fare, Gender (Gender만 범주형으로 사용합니다)
# - Pclass와 Age 변수의 교호작용 효과 확인을 위한 항을 포함

# 3-1) 위의 조건에 따라 모델을 생성하고, summary()를 사용하여 결과를 확인합니다.
from statsmodels.api import GLM, families
formula = "Survived ~ Pclass * Age + Fare + C(Gender)"
model = GLM.from_formula(formula, df, family=families.Binomial()).fit()
# print(model.summary())

# 3-2) 분석 결과, Pclass와 Age의 교호작용 항에 대한 p-value를 반올림하여 소수점 아래 3자리까지 출력합니다.
# print(round(model.pvalues['Pclass:Age'], 3)) # 0.686

# 3-3) 위에서 생성된 model 결과를 기반으로, 다음 해석이 올바르면 1, 틀리면 0을 입력합니다.
# Pclass가 1단계 증가할 때 생존할 오즈는 감소한다.
# print(np.exp(model.params['Pclass']))     # 0.400
# print(np.exp(model.params['Pclass'] * 2)) # 0.160
# 1

# 3-4) 모델의 로그-우도를 반올림하여 소수점 아래 3자리까지 출력합니다.
# print(round(model.llf, 3)) # -149.902

# 3-5) 모델의 잔차이탈도를 반올림하여 소수점 아래 3자리까지 출력합니다.
# print(round(model.deviance, 3)) # 299.804

#  3-6) 다음 sample의 P(Y=1)의 확률은? 반올림하여 소수점 아래 3자리까지 출력합니다.
sample = {'Age': [20], 'Fare': [50], 'Pclass': [2], 'Gender': ['female']}
data = pd.DataFrame(sample)
# print(round(model.predict(data)[0], 3)) # 0.926

# 3-7) 위 sample에 대한 로그-오즈는? 결과를 반올림하여 소수점 아래 4자리까지 출력한다.
y_1 = model.predict(data)[0]
odds = y_1 / (1 - y_1)
# print(round(np.log(odds), 4))
result = model.predict(data, which='linear')[0]
# print(round(result, 4)) # 2.5306

# 3-8) Pclass가 1단위 증가할 때 Survived 오즈가 약 몇% 감소하는가? 결과를 반올림하여 소수점 아래 2자리까지 출력
odds_ratio = np.exp(model.params['Pclass'])
# print(round((1 - odds_ratio) * 100, 2)) # 59.97

# 3-9) Gender가 'male'인 경우 Survived 오즈가 약 몇% 감소하는가? 결과를 반올림하여 소수점 아래 2자리까지 출력한다.
odds_ratio = np.exp(model.params['C(Gender)[T.male]'])
# print(round((1 - odds_ratio) * 100, 2)) # 95.82

# 3-10) Fare가 10단위 증가할 때 Survived 오즈가 약 몇% 증가하는가? 결과를 반올림하여 소수점 아래 2자리까지 출력한다.
odds_ratio = np.exp(model.params['Fare'] * 10)
# print(round((odds_ratio - 1) * 100, 2)) # 4.24

# 3-11) 성별을 제외한 모든 조건이 동일할 때,
# 남성(male)의 생존 확률이 여성(female)보다 몇 %포인트 낮은가? 단, 확률 변화량은 (남성 생존 확률 - 여성 생존 확률)로 계산하며,
# 결과는 소수 둘째 자리까지 반올림하여 %p 단위로 표현하시오. 본 문항은 '성별에 따른 생존 확률 변화량'을 구하는 것입니다.
# 표본 데이터 : Pclass=2, Age=30, Fare=50 인 남성/여성
sample = pd.DataFrame({'Pclass': [2, 2], 'Age': [30, 30], 'Fare': [50, 50], 'Gender': ['female', 'male']})
prob_female, prob_male = model.predict(sample)
result = prob_male - prob_female
# print(round(result * 100, 2)) # -64.33

# 3-12) 모델의 회귀 계수합을 구하고, 반올림하여 소수점 아래 3자리까지 출력한다.
# print(round(model.params.sum(), 3)) # 1.048
