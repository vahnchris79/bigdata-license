
import pandas as pd
pd.set_option('display.width', 150)
path1 = "https://raw.githubusercontent.com/Soyoung-Yoon/bigdata/main/"
path2 = "https://raw.githubusercontent.com/Soyoung-Yoon/data_01/main/"

# 작업형 제1유형
# 4-1) 샘플 개수 구하기
# 다음 조건을 모두 만족하는 샘플의 개수를 정수로 구하시오.
# - 단, bmi, s5, target 컬럼을 사용하며 다음 조건을 모두 만족해야 한다.
# - bmi가 전체 평균 이상인 데이터
# - s5가 전체 중앙값보다 큰 데이터
# - target 값이 150 이상인 데이터
df1 = pd.read_csv(path1 + "diabetes.csv")
# print(df1.head(3))
# bmi가 전체 평균 이상인 데이터
cond1 = df1['bmi'] >= df1['bmi'].mean()
# s5가 전체 중앙값보다 큰 데이터
cond2 = df1['s5'] > df1['s5'].median()
# target 값이 150 이상인 데이터
cond3 = df1['target'] >= 150
# print(int(len(df1[cond1 & cond2 & cond3]))) # 108

# 4-2) 펭귄의 수 구하기
# 다음 조건을 모두 만족하는 펭귄의 수를 구해 정수로 출력하시오.
# - species가 Gentoo인 데이터만 필터링한다.
# - 위의 필터링 된 데이터 중에서 bill_length_mm이 Gentoo 펭귄 전체에서 상위 20%에 해당하면서, flipper_length_mm이 220 이상인 경우만 유지한다.
# - 단, **상위 20%에 해당하는 것이 여러 개인 경우 모두 포함**하도록 한다.
df2 = pd.read_csv(path2 + "penguins03.csv")
# species가 Gentoo인 데이터만 필터링한다.
df2 = df2[df2['species']=='Gentoo'].copy()
# 의 필터링 된 데이터 중에서 bill_length_mm이 Gentoo 펭귄 전체에서 상위 20%에 해당하면서, flipper_length_mm이 220 이상인 경우만 유지
cond1 = (df2['bill_length_mm'] >= df2['bill_length_mm'].quantile(0.8))
cond2 = (df2['flipper_length_mm'] >= 220)
# print(int(len(df2[cond1 & cond2]))) # 19

# 4-3) 다이아몬드 가격
# 다음 조건을 모두 만족하는 다이아몬드의 가격(price) 합계를 정수로 구하시오.
# - cut 등급이 'Premium'이고, color가 'D' 또는 'F' 또는 'H'인 데이터만 필터링한다.
# - 위 필터링된 데이터에서, 각 (cut, color) 그룹별 **carat 중앙값**을 계산한다.
# - 그리고 필터링된 데이터에서 각 다이아몬드의 carat이 해당 그룹(cut, color)의 중앙값보다 큰 데이터를 최종적으로 선택한다.
df3 = pd.read_csv(path1 + "diamonds.csv")
# cut 등급이 'Premium'이고, color가 'D' 또는 'F' 또는 'H'인 데이터만 필터링
df3 = df3[(df3['cut']=='Premium') & (df3['color'].str.contains('D|F|H'))].copy()
# 위 필터링된 데이터에서, 각 (cut, color) 그룹별 **carat 중앙값**을 계산
carat_median = df3.groupby(['cut','color'])['carat'].transform('median')
# 필터링된 데이터에서 각 다이아몬드의 carat이 해당 그룹(cut, color)의 중앙값보다 큰 데이터를 최종적으로 선택
df3 = df3[df3['carat'] > carat_median].copy()
# 위 조건을 모두 만족하는 다이아몬드의 가격(price) 합계를 정수로 구하시오.
# print(int(df3['price'].sum())) # 23289898

# 4-4) 펭귄의 체중
# penguins 데이터셋을 사용하여 다음 조건을 모두 만족하는 펭귄의 body_mass_g 표준편차를 구하시오.
# - 모든 결측치는 제거한 후 처리하도록 한다.
# - 각 펭귄의 **body_mass_g가 자신이 속한 species의 중앙값보다 큰 경우**의 데이터를 선택한다.
# - 위의 조건을 만족하는 펭귄 중 sex가 'Male'이고 bill_depth_mm이 17을 초과하는 개체를 선택한다.
# - 결과는 반올림하여 소수점 아래 3자리까지 표시되도록 한다.
df4 = pd.read_csv(path1 + "penguins01.csv")
# print(df4.isna().sum().to_frame().T)
# 모든 결측치는 제거
df4 = df4.dropna()
# print(df4.isna().sum().to_frame().T)
# 각 펭귄의 **body_mass_g가 자신이 속한 species의 중앙값보다 큰 경우**의 데이터를 선택
mass_median = df4.groupby('species')['body_mass_g'].transform('median')
df4 = df4[df4['body_mass_g'] > mass_median].copy()
# 위의 조건을 만족하는 펭귄 중 sex가 'Male'이고 bill_depth_mm이 17을 초과하는 개체를 선택
df4 = df4[(df4['sex']=='Male') & (df4['bill_depth_mm'] > 17)].copy()
# 위 조건을 모두 만족하는 펭귄의 body_mass_g 표준편차를 반올림하여 소수점 아래 3자리까지 표시
# print(round(df4['body_mass_g'].std(ddof=1), 3)) # 364.504

# 4-5) 승객의 수 구하기
# titanic.csv를 사용하여 다음 조건을 모두 만족하는 승객의 수를 정수로 구하시오.
# - Age 컬럼의 결측치는 **Pclass와 Gender 조합별 Age 평균값**으로 채운다.
# - 그 후, Age가 30세 이상인 승객만을 필터링한다.
# - 필터링된 데이터에서 **각 승객의 요금(Fare)이 동일 성별(Gender) 승객들의 요금 중앙값 이상인 경우**만 필터링한다.
df5 = pd.read_csv(path1 + "titanic_dataq.csv")
# print(df5.head(3))
# Age 컬럼의 결측치는 **Pclass와 Gender 조합별 Age 평균값**으로 채운다.
age_mean = df5.groupby(['Pclass', 'Gender'])['Age'].transform('mean')
df5['Age'] = df5['Age'].fillna(age_mean)
# 그 후, Age가 30세 이상인 승객만을 필터링
df5 = df5[df5['Age'] >= 30].copy()
# 필터링된 데이터에서 **각 승객의 요금(Fare)이 동일 성별(Gender) 승객들의 요금 중앙값 이상인 경우**만 필터링
fare_median = df5.groupby('Gender')['Fare'].transform('median')
df5 = df5[df5['Fare'] >= fare_median].copy()
# 위 조건을 모두 만족하는 승객의 수를 정수로 구하시오.
# print(int(len(df5['PassengerId']))) # 185

# 작업형 제2유형
# 호주 멜버른 지역의 주택 거래 정보를 담고 있는 실제 부동산 데이터입니다. 
# 이 데이터는 주택 판매 기록을 포함하고 있으며, 면적, 방 개수, 건축 연도, 위치 정보 등 다양한 요인을 바탕으로 주택 가격을 분석할 수 있습니다.
# 제공된 학습용 데이터(mhousing_train.csv)를 이용하여 판매된 가격(Price)를 예측하는 모델을 개발하고, 
# 개발한 모델에 기반하여 평가용 데이터(mhousing_test.csv)에 적용하여 얻은 판매된 가격 예측 값을 
# 아래 [제출형식]에 따라 csv 파일로 생성하여 제출하시오.
# - 예측 결과는 RMSLE(Root Mean Squared Log Error) 평가지표에 따라 평가함
# - 성능이 우수한 예측 모델을 구축하기 위해서는 데이터 정제, Feature Engineering, 하이퍼 파라미터(hyper parameter) 최적화, 모델 비교 등이 필요할 수 있음. 다만, 과적합에 유의하여야 함
# [[제출 형식]]
# - 가. CSV 파일명: result.csv(파일명에 디렉토리/폴더 지정불가
# - 나. 예측 칼럼명 : pred
# - 다. 제출 칼럼 개수 : pred 칼럼 1개
# - 라. 평가용 데이터 개수와 예측 결과 데이터 개수 일치 : 1,859개

# 라이브러리
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_log_error as MSLE
pd.options.display.float_format = '{:.0f}'.format

# 데이터 확인
train = pd.read_csv(path1 + "mhousing_train.csv")
test = pd.read_csv(path1 + "mhousing_test.csv")
# print(train.head(3), train.shape, sep="\n") # (4337, 19)
# print(test.head(3), test.shape, sep="\n")   # (1859, 18)

# 데이터 전처리
X = train.drop(columns=['Price'])
Y = train['Price']
X_all = pd.concat([X, test])
X_all['Date'] = X_all['Date'].astype('datetime64[ns]')
X_all['Year'] = X_all['Date'].dt.year
X_all['Month'] = X_all['Date'].dt.month
X_all['Day'] = X_all['Date'].dt.day
X_all['Weekday'] = X_all['Date'].dt.weekday
# print(X_all.head(3))
X_all = X_all.drop(columns=['Date'])
cols_obj = X_all.select_dtypes(include='object').columns
for col in cols_obj:
    X_all[col] = LabelEncoder().fit_transform(X_all[col])
# print(X_all.head(3))
# X_all = pd.get_dummies(X_all, drop_first=True, dtype='int32')
# print(X_all.head(3))

# 데이터 분할
X = X_all.iloc[:len(X), :]
X_submission = X_all.iloc[len(X):, :]
# print(X.shape, X_submission.shape) # (4337, 21) (1859, 21)
temp = train_test_split(X, Y, test_size=0.3, random_state=42)
x_train, x_test, y_train, y_test = temp
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (3035, 21) (1302, 21) (3035,) (1302,)

# 파이프라인 모델사전
models = {
    "Linear": Pipeline([
        ("scaler", MinMaxScaler()), ("model", LinearRegression())
    ]),
    "DecisionTree": Pipeline([
        ("model", DecisionTreeRegressor(max_depth=3, random_state=42))
    ]),
    "RandomForest": Pipeline([
        ("model", RandomForestRegressor(n_estimators=70, max_depth=3, random_state=42))
    ]),
    "AdaBoost": Pipeline([
        ("model", AdaBoostRegressor(n_estimators=70, random_state=42))
    ]),
    "GradientBoosting": Pipeline([
        ("model", GradientBoostingRegressor(random_state=42))
    ])
}

# 성능평가 함수
def get_scores(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    y_pred1 = abs(model.predict(x_train))
    y_pred2 = abs(model.predict(x_test))
    RMSLE_train = MSLE(y_train, y_pred1) ** 0.5
    RMSLE_test = MSLE(y_test, y_pred2) ** 0.5
    return model, RMSLE_train, RMSLE_test

# 모델별 성능평가
results = []
for name, model in models.items():
    model, RMSLE_train, RMSLE_test = get_scores(model, x_train, x_test, y_train, y_test)
    results.append({
        "Model": name, "RMSLE_train": f"{RMSLE_train:.4f}", "RMSLE_test": f"{RMSLE_test:.4f}"
    })
res = pd.DataFrame(results).sort_values("RMSLE_test", ascending=True).reset_index(drop=True)
# print(res)

# 모델선택, 예측값 생성
model = models[res.loc[0, "Model"]]
y_pred = model.predict(X_submission)

# 제출파일 생성
# pd.DataFrame({'pred': y_pred}).to_csv("result_day4_1st.csv", index=False)

# 결과확인
# temp = pd.read_csv("result_day4_1st.csv")
# print(Y[:len(X_submission)].describe())
# print("=" * 35)
# print(temp['pred'].describe())

# 작업형 제3유형
# Day4 다중 선형 회귀
# 당뇨병(diabetes) 데이터를 사용해 다중 선형 회귀를 수행합니다.
# - 모든 feature는 표준화(z-score 변환) 되어 있어서 평균 0, 분산 1에 가깝습니다.
# - age : 환자의 나이
# - sex : 성별
# - bmi : 체질량지수
# - bp : 평균 혈압
# - s1 ~ s6 : 6가지 혈액 검사 결과
# - target : 1년 후 당뇨병 진행도 지표
df = pd.read_csv(path1 + "diabetes.csv")
# print(df.head(3))
# 다음과 같은 다중선형회귀 모형을 사용한 회귀모델을 만들고 결과를 확인합니다.
# - diabetes.csv 데이터를 사용합니다.
# - 모델 생성시 상수항(=절편)을 포함하도록 합니다.
# - 종속변수 : target
# - 독립변수 : target을 제외한 모든 변수
from statsmodels.api import OLS, add_constant
# df = add_constant(df)
formula = "target ~ " + ' + '.join(df.columns[:-1])
model = OLS.from_formula(formula, df).fit()
# print(model.summary())

#4-2) 위에서 생성한 모델의 결정계수를 반올림하여 소수점 아래 3자리까지 출력합니다.
# print(round(model.rsquared, 3)) # 0.518

#4-3) 위에서 생성한 모델의 수정된 결정계수를 반올림하여 소수점 아래 3자리까지 출력합니다.
# print(round(model.rsquared_adj, 3)) # 0.507

#4-4) 유의수준 0.05하에서 통계적으로 유의한 독립변수의 개수는 몇 개인가요?
# print(sum(model.pvalues[1:] <= 0.05)) # 4

#4-5) 유의수준 0.05하에서 통계적으로 유의한 독립변수만 사용하여 target을 종속변수로 하는 모델을 생성하여
# model2로 저장하고, summary()를 확인합니다.
s = model.pvalues[1:] <= 0.05
formula2 = 'target ~ ' + ' + '.join(s[s].index)
model2 = OLS.from_formula(formula2, df).fit()
# print(model2.summary())

#4-6) bmi 변수에 대한 회귀 계수를 구해, 반올림하여 소수점 아래 3자리까지 출력합니다.
# print(round(model2.params['bmi'], 3)) # 598.284

#4-7) 영향력이 가장 높은 변수 및 변수의 회귀계수를 구해, 반올림하여 소수점 아래 3자리까지 출력합니다.
# 중간에 공백을 1개 넣어 2개를 한 줄에 출력해 봅니다.
# 출력 예) s1 123.456
temp = model2.params[1:].abs().idxmax()
# print(temp, round(model2.params[temp], 3)) # bmi 598.284

#4-8) 영향력이 가장 낮은 변수 및 변수의 회귀계수를 구해, 반올림하여 소수점 아래 3자리까지 출력합니다.
# 중간에 공백을 1개 넣어 2개를 한 줄에 출력해 봅니다.
temp = model2.params[1:].abs().idxmin()
# print(temp, round(model2.params[temp], 3)) # sex -136.758

#4-9) F통계량을 구해, 반올림하여 소수점 아래 3자리까지 출력합니다.
# print(round(model2.fvalue, 3)) # 103.618

#4-10) 독립변수 중 가장 높은 p-value를 구해, 반올림하여 소수점 아래 3자리까지 출력합니다.
# print(round(model2.pvalues[1:].max(), 3)) # 0.017

#4-11) 통계적으로 가장 유의한 변수는 무엇인가?
result = model2.pvalues[1:].idxmin()
# print(result) # bmi

#4-12) 통계적으로 가장 유의한 변수의 회귀계수를 구해, 반올림하여 소수점 아래 3자리까지 출력합니다.
temp = model2.pvalues[1:].idxmin()
# print(round(model2.params[temp], 3)) # 598.284

