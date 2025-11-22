
# 공통: 데이터 경로
import pandas as pd
path1 = "https://raw.githubusercontent.com/Soyoung-Yoon/bigdata/main/"
path2 = "https://raw.githubusercontent.com/Soyoung-Yoon/data_01/main/"
path3 = "https://raw.githubusercontent.com/Soyoung-Yoon/data_02/main/"

# 작업형 제1유형

# 5회-1번) 종량제 봉투 가격
# 다음은 전국 시도별 종량제 봉투 가격에 대한 데이터를 사용하여 지역별 봉투 가격 차이를 분석하여 가격 정책의 차이를 파악하고자 한다.
# 다음 절차에 따라 문제를 해결하시오.
df = pd.read_csv(path1 + "waste_bags.csv")
# print(df.head(3), df.info(), sep="\n") # 571
# 가) 20L가격과 5L가격이 모두 0원이 아닌 데이터만 필터링하시오.
df = df[(df['20L가격'] > 0) & (df['5L가격'] > 0)]
# print(df.info()) # 296
# 나) 그 후, 각 행별로 '20L가격' - '5L가격'을 계산하여 '가격차이'를 구하시오.
df['가격차이'] = df['20L가격'] - df['5L가격']
# print(df.head(3))
# 다) '시도명'별 '가격차이'의 평균가격을 구한 뒤, 그 중 값이 가장 큰 평균가격을 반올림하여 정수로 출력한다.
# print(round(df.groupby('시도명')['가격차이'].mean().max())) # 626

# 5회-2번) 건강 관리 지표
# 성인 체형 분포를 파악하여 건강 관리 지표를 설정하고자 한다.
# 다음 절차에 따라 문제를 해결하시오.

df = pd.read_csv(path1 + "height_weight.csv")
# print(df.head(3))
# 가) BMI를 계산하시오. (BMI = 몸무게(kg) / (키(m) × 키(m)))
df['height(m)'] = df['height(cm)'] / 100
df['BMI'] = df['weight(kg)'] / (df['height(m)'] * df['height(m)'])
# print(df.head(3))
# 나) 초고도 비만 인원과 저체중 인원의 합계를 구하여 정수로 출력한다.
#   > BMI에 따라 초고도 비만(25 이상), 고도 비만(23 이상 25 미만), 정상(18.5 이상 23 미만), 저체중(18.5 미만)으로 분류된다.
for i in df.index:
    if df.loc[i, 'BMI'] >= 25:
        df.loc[i, 'class'] = '초고도 비만'
    elif df.loc[i, 'BMI'] >= 23 and df.loc[i, 'BMI'] < 25:
        df.loc[i, 'class'] = '고도 비만'
    elif df.loc[i, 'BMI'] >= 18.5 and df.loc[i, 'BMI'] < 23:
        df.loc[i, 'class'] = '정상'
    else:
        df.loc[i, 'class'] = '저체중'
# print(int(len(df[df['class']=='초고도 비만']) + len(df[df['class']=='저체중']))) # 9010

# 5회-3번. 년도별 서울 각 구의 초,중,고 전출 전입 인원
# 서울시 각 구별 초·중·고 전출입 인원 데이터입니다. 지역구별 순유입인원 변화를 분석하여, 인구 이동이 가장 활발한 지역을 분석한다.
# 다음 절차에 따라 문제를 해결하시오.
df = pd.read_csv(path2 + "school.csv")
# print(df.info(3))
# print(df.head(3))
# 가) 순유입인원은 (초중고 도내·도외 전입인원의 합) - (초중고 도내·도외 전출인원의 합)으로 계산한다.
df['순유입인원'] = df.iloc[:, [3,4,7,8,11,12]].sum(axis=1) - df.iloc[:, [1,2,5,6,9,10]].sum(axis=1)
# print(df.head(3))
# 나) 각년도별로 가장 큰 순유입인원을 가진 지역구의 순유입인원을 구하고 전체 기간의 해당 순유입인원들의 합을 구하여, 정수형으로 출력한다
# 나)에 대한 설명) # 문제 이해가 어렵다고 하셔서 설명을 추가해 보았습니다.
#    > 각년도별로 순유입인원의 최댓값을 뽑은 뒤, 전체 기간에 대해 순유입인원 최댓값의 합을 구하여, 정수형으로 출력한다.
# df = df.set_index('지역')
result = df.groupby('년도')['순유입인원'].max().sum()
# print(int(result)) # 13853

# 13-4) 봉투종류별 20L 봉투 할인율 분석
# '종량제봉투종류'별로 대용량 봉투(20L) 사용 시 할인 효과를 분석한다.
# 다음 절차에 따라 문제를 해결하시오.
df = pd.read_csv(path1 + "waste_bags.csv")
# print(df.head(3), df.shape, sep="\n") # (571, 10)
# 가) 5L가격 또는 20L가격이 0인 경우는 제외하고 계산한다.
df = df[df['5L가격'] > 0].copy()
df = df[df['20L가격'] > 0].copy()
# print(df.shape) # (296, 10)
# 나) '5L가격'과 '20L가격'을 사용하여, 각 행별로 할인율을 계산하시오.
#    > 할인율 = (5L가격*4 - 20L가격) / (5L가격 * 4)
df['할인율'] = ((df['5L가격'] * 4) - df['20L가격']) / (df['5L가격'] * 4)
# print(((df['5L가격'] * 4) - df['20L가격']) / (df['5L가격'] * 4))
# print(df.head(3))
# 다) ('시도명', '종량제봉투종류')별로 평균 할인율을 구하시오.
s = df.groupby(['시도명','종량제봉투종류'], as_index=False)['할인율'].mean().set_index(['시도명','종량제봉투종류'])
# print(s)
# 라) 평균 할인율이 가장 높은 '시도명'과 '종량제봉투종류'를 찾으시오.
# city = s.idxmax()[0][0]
# bongtu = s.idxmax()[0][1]
# print(city, bongtu) # 경상남도 재사용규격봉투
# 마) 라)에서 찾은 것에 대한 '20L가격'을 A, '5L가격'을 B라고 할 때 A+B의 값을 구해 정수로 출력한다
# A = df[(df['시도명']==city) & (df['종량제봉투종류']==bongtu)]['20L가격'].values[0]
# B = df[(df['시도명']==city) & (df['종량제봉투종류']==bongtu)]['5L가격'].values[0]
# print(int(A+B)) # 900

# 작업형 제2유형
# Seoul Bike Sharing Dataset 데이터셋
# 서울시의 시간대별 자전거 대여 데이터를 포함하고 있으며, 날씨와 계절 요인이 자전거 대여량에 
# 어떤 영향을 미치는지를 분석할 수 있는 시계열형 예측 데이터셋입니다.
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
# print(X_all.info())
X_all['Date'] = X_all['Date'].astype('datetime64[ns]')
X_all['Year'] = X_all['Date'].dt.year
X_all['Month'] = X_all['Date'].dt.month
X_all['Day'] = X_all['Date'].dt.day
X_all['Weekday'] = X_all['Date'].dt.weekday
X_all = X_all.drop(columns=['Date'])
X_all['Seasons'] = LabelEncoder().fit_transform(X_all['Seasons'])
X_all['Holiday'] = LabelEncoder().fit_transform(X_all['Holiday'])
X_all = pd.get_dummies(X_all, drop_first=True, dtype='int32')
# print(X_all.head())

# 데이터 재분할
X = X_all.iloc[:len(X), :]
X_submission = X_all.iloc[len(X):, :]
# print(X.shape, X_submission.shape) # (6044, 16) (2716, 16)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=900)
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (4230, 16) (1814, 16) (4230,) (1814,)

# 파이프라인 모델사전
models = {
    "Linear": Pipeline([
        ("scaler", MinMaxScaler()), ("model", LinearRegression())
    ]),
    "DecisionTree": Pipeline([
        ("model", DecisionTreeRegressor(max_depth=9, random_state=900))
    ]),
    "RandomForest": Pipeline([
        ("model", RandomForestRegressor(n_estimators=900, max_depth=9, random_state=900))
    ]),
    "AdaBoost": Pipeline([
        ("model", AdaBoostRegressor(n_estimators=900, random_state=900))
    ]),
    "Gradient": Pipeline([
        ("model", GradientBoostingRegressor(random_state=900))
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

# 모델적합, 예측값 생성
model = models[res.loc[0, 'Model']]
y_pred = model.predict(X_submission)

# 결과파일 생성
# pd.DataFrame({'pred': y_pred}).to_csv("result_day13.csv", index=False)

# 결과확인
# temp = pd.read_csv("result_day13.csv")
# print(Y[:len(test)].describe())
# print("=" * 35)
# print(temp['pred'].describe())

# 작업형 제1, 3유형
# 19-1) 펭귄 종에 따른 특성 비교 분석
# 펭귄의 날개 길이와 체중은 종(species)별 신체 특성을 파악하는 데 중요한 지표이다.
# 다음 조건에 따라 'Gentoo' 종 펭귄의 특성을 분석하시오.
df = pd.read_csv(path2 + "penguins04.csv")
# print(df.head(3), df.info()) # 335
# species가 'Gentoo'인 데이터만을 대상으로 한다.
df = df[df['species'] == 'Gentoo']
# print(df.shape) # (120, 7)
# flipper_length_mm과 body_mass_g 변수의 결측치는 'Gentoo' 종의 평균값으로 대체한다.
# print(df.isna().sum())
df['flipper_length_mm'] = df['flipper_length_mm'].fillna(df['flipper_length_mm'].mean())
df['body_mass_g'] = df['body_mass_g'].fillna(df['body_mass_g'].mean())
# print(df.isna().sum())
# 처리된 데이터 중 flipper_length_mm이 상위 20% 이상인 개체만 추출한다.
df = df[df['flipper_length_mm'] >= df['flipper_length_mm'].quantile(0.8)]
# print(df.shape) # (24, 7)
# 이 개체들의 body_mass_g 평균값을 구하고, 결과는 반올림하여 소수점 아래 3자리까지 출력한다.
# print(round(df['body_mass_g'].mean(),3)) # 5543.75

# 19-2) 부리 길이 극단 그룹의 체중 비교 분석
# 펭귄의 부리 길이(bill_length_mm)는 먹이 섭취 방식과 서식지 적응에 따라 차이가 날 수 있다.
# 다음 절차에 따라 문제를 해결하시오.
df = pd.read_csv(path2 + "penguins05.csv")
# print(df.head(3), df.info(), sep="\n") # 330
# bill_length_mm 기준으로 상위 20%와 하위 10%에 해당하는 개체들을 각각 추출한다.
df_high = df[df['bill_length_mm']>=df['bill_length_mm'].quantile(0.8)]
df_low = df[df['bill_length_mm']<=df['bill_length_mm'].quantile(0.1)]
# print(df_high.shape, df_low.shape) # (66, 7) (298, 7)
# 두 그룹에서 body_mass_g의 평균값을 각각 계산하고,
high_mean = df_high['body_mass_g'].mean()
low_mean = df_low['body_mass_g'].mean()
# 이 두 평균값의 차이(상위 20% 평균 - 하위 10% 평균)를 구하시오.
# 결과는 반올림하여 소수점 아래 3자리까지 출력한다.
# print(round(high_mean - low_mean, 3)) # 1262.879

# 19-3) 중간 수준 출력 차량의 지역별 연비 특성 분석
# 차량의 마력(horsepower)은 연비(mpg)의 관련성을 알기위해 중간 성능대 차량의 지역별 연비 경향을 분석한다.
# 다음 절차에 따라 문제를 해결하시오.
df = pd.read_csv(path2 + "mpg02.csv")
# print(df.head(), df.shape, sep="\n") # (400, 9)
# - horsepower 값이 전체에서 하위 20% 이상, 상위 20% 이하인 차량만을 대상으로 한다.
#   (즉, horsepower가 전체의 20% 분위수 이상, 80% 분위수 이하인 데이터만 필터링)
df = df[(df['horsepower']>=df['horsepower'].quantile(0.2)) & (df['horsepower']<=df['horsepower'].quantile(0.8))]
# print(df.shape) # (240, 9)
# - 이후, 필터링된 데이터에서 origin별로 연비(mpg)의 중앙값을 계산하고, 
#   그 중 가장 큰 값과 가장 작은 값의 차이를 반올림하여 정수로 출력한다.
s = df.groupby('origin')['mpg'].median()
# print(round(s.values.max() - s.values.min())) # 6

# 19-4) 펭귄의 수 구하기
# 다음 조건을 모두 만족하는 펭귄의 수를 구해 정수로 출력하시오.
df = pd.read_csv(path2 + "penguins03.csv")
# print(df.head(3), df.info(), sep="\n") # 334
# - species가 Gentoo인 데이터만 필터링한다.
df = df[df['species'] == 'Gentoo'].copy()
# print(df.shape) # 120, 7
# - 위의 필터링 된 데이터 중에서 bill_length_mm이 Gentoo 펭귄 전체에서 상위 20%에 해당하면서, 
#   flipper_length_mm이 220 이상인 경우만 유지한다.
df = df[(df['bill_length_mm']>=df['bill_length_mm'].quantile(0.8)) & (df['flipper_length_mm'] >= 220)]
# - 단, **상위 20%에 해당하는 것이 여러 개인 경우 모두 포함**하도록 한다.
# print(len(df)) # 19

# 19-5a) 지속시간의 분위수 분석
# 다음 절차에 따라 문제를 해결하시오.
df = pd.read_csv(path2 + "event_log_03.csv")
# print(df.head(3), df.info(), sep="\n") # 1000
# - 지속시간 기준 75분위수(=제3사분위수) 이상에 해당하는 데이터의 지속시간(duration) 합계를 초단위 정수로 출력한다.
# - 지속시간(duration) = end_datetime - start_datetime
df['start_datetime'] = df['start_datetime'].astype('datetime64[ns]')
df['end_datetime'] = df['end_datetime'].astype('datetime64[ns]')
df['duration'] = df['end_datetime'] - df['start_datetime']
# print(round(df[df['duration']>=df['duration'].quantile(0.75)]['duration'].sum().total_seconds())) # 133254

# 19-5b) 정확한 개수의 사용
# 다음 절차에 따라 문제를 해결하시오.
df = pd.read_csv(path2 + "event_log_03.csv")
# - 지속시간 기준 정확히 상위 25%에 해당하는 데이터 개수만 뽑아 지속시간(duration) 합계를 초단위 정수로 출력한다.
# - 지속시간(duration) = end_datetime - start_datetime
df['start_datetime'] = df['start_datetime'].astype('datetime64[ns]')
df['end_datetime'] = df['end_datetime'].astype('datetime64[ns]')
# df['duration'] = df['end_datetime'] - df['start_datetime']
s = df['end_datetime'] - df['start_datetime']
# print(round(s.nlargest(int(len(s)*0.25)).sum().total_seconds())) # 131434

# 19-6) 야간 시간대 이벤트
# 다음은 야간 시간대에 발생한 이벤트 유형 분석을 통해 사용자 활동 패턴을 파악하고자 하는 과제이다.
# 다음 절차에 따라 문제를 해결하시오.
df = pd.read_csv(path2 + "event_log_04.csv")
# print(df.head(3), df.info(), sep="\n") # 1000
# - 먼저, 시간(hour) 정보가 22시부터 02시까지(22:00:00 ~ 02:59:59)인 데이터를 야간 이벤트로 간주한다.
df['datetime'] = df['date'] + " " + df['time']
df['datetime'] = df['datetime'].astype('datetime64[ns]')
df = df[(df['datetime'].dt.hour >= 22) | (df['datetime'].dt.hour < 3)]
# - 이 야간 이벤트를 기준으로, (month, event)별 value 평균값을 구하시오.
df['month'] = df['datetime'].dt.month
cond = df.groupby(['month','event'], as_index=False)['value'].mean().sort_values('value',ascending=False)
# print(cond)
# - 각 월(month)에서 value 평균이 가장 높은 이벤트 유형을 하나씩 선택하고, 그 중 'Start'와 'Stop'이벤트의 빈도 수 합을 정수로 출력하시오.

