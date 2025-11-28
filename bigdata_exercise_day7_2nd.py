
# 공통
path1 = "https://raw.githubusercontent.com/Soyoung-Yoon/bigdata/main/"
path2 = "https://raw.githubusercontent.com/Soyoung-Yoon/data_01/main/"

# 작업형 제1유형
import pandas as pd
pd.set_option('display.width', 120)

# 7-1) 지역별 공간점수
# 다음 절차에 따라 문제를 해결하시오.
# bed와 bath의 값이 모두 0이 아닌 행만 선택한다.
# 각 행(row)별로 bed * 1.5 + bath * 2 의 값을 '공간점수'라고 정의한다.
# 지역(zip_code)별로 공간점수의 평균을 계산한 후, 지역별 평균 공간점수가 
# 가장 높은 7개 지역의 가격(price) 평균을 반올림하여 정수로 구하라.
df = pd.read_csv(path1 + "housing01.csv")
# print(df.head(3), df.shape, sep="\n") # (104667, 12)
cond = (df['bed'] > 0) & (df['bath'] > 0)
df = df[cond].copy()
# print(df.shape) # (104667, 12)
df['공간점수'] = (df['bed'] * 1.5) + (df['bath'] * 2)
s = df.groupby('zip_code')['공간점수'].mean().nlargest(7)
result = df[df['zip_code'].isin(s.index)]['price'].mean()
# print(round(result)) # 11146024

# 7-2) 이벤트 발생 빈도
# 다음 절차에 따라 문제를 해결하시오.
# (년, 월)별로 발생한 이벤트의 개수를 확인하여, 이벤트 발생 개수 TOP3의 이벤트 수 합을 구하여 정수로 출력하시오.
# TOP3는 가장 큰 값 3개를 의미한다.
# start_datetime : 발생 날짜/시간, end_datetime : 종료 날짜/시간
df2 = pd.read_csv(path2 + "event_log_03.csv")
# print(df2.head(3))
df2['year'] = df2['start_datetime'].astype('datetime64[ns]').dt.year
df2['month'] = df2['start_datetime'].astype('datetime64[ns]').dt.month
# print(df2.head(3))
result = df2.groupby(['year','month'])['event'].size().sort_values(ascending=False).nlargest(3).sum()
# print(result) # 111

# 7-3) 이벤트 발생 시간분석
# 다음 절차에 따라 문제를 해결하시오.
# 이벤트가 시작된 시각(start_datetime)을 기준으로 이벤트가 가장 많이 발생한 시(hour)를 구하시오.
# 해당 시각에 발생한 모든 이벤트의 value 값을 합산하고, 그 합계를 반올림하여 정수로 출력하시오.
df3 = pd.read_csv(path2 + "event_log_03.csv")
# print(df3.head(3))
df3['start_hour'] = df3['start_datetime'].astype('datetime64[ns]').dt.hour
# print(df3.head(3))
group = df3.groupby('start_hour')['event'].count().reset_index(name='count')
max_hour = group.sort_values('count', ascending=False).set_index('start_hour').idxmax().iloc[0]
result3 = df3[df3['start_hour'] == max_hour]['value'].sum()
# print(round(result3)) # 2714

# 7-4) 월별 이벤트 발생 빈도
# 다음 절차에 따라 문제를 해결하시오.
# (년, 월)별로 발생한 이벤트의 개수를 확인하여, 년별로 가장 많은 이벤트가 발생한 월에 해당하는 
# 데이터 개수의 합을 구하여 정수로 출력하시오.
df4 = pd.read_csv(path2 + "event_log_03.csv")
# print(df4.head(3))
df4['year'] = df4['start_datetime'].astype('datetime64[ns]').dt.year
df4['month'] = df4['start_datetime'].astype('datetime64[ns]').dt.month
count = df4.groupby(['year','month'])['event'].size().unstack()
result4 = count.max(axis=1).sum()
# print(result4) # 108

# 7-5a) 지속시간의 분위수 분석 다음 절차에 따라 문제를 해결하시오.
# 지속시간 기준 75분위수(=제3사분위수) 이상에 해당하는 데이터의 지속시간(duration) 합계를 초단위 정수로 출력한다.
# 지속시간(duration) = end_datetime - start_datetime
df5a = pd.read_csv(path2 + "event_log_03.csv")
# print(df5a.head(3))
duration = df5a['end_datetime'].astype('datetime64[ns]') - df5a['start_datetime'].astype('datetime64[ns]')
result5a = int(duration[duration>=duration.quantile(0.75)].sum().total_seconds())
# print(result5a) # 133254

# 7-5b) 지속시간의 정확한 개수 사용
# 다음 절차에 따라 문제를 해결하시오.
# 지속시간 기준 정확히 상위 25%에 해당하는 데이터 개수만 뽑아 지속시간(duration) 합계를 초단위 정수로 출력한다.
# 지속시간(duration) = end_datetime - start_datetime
df5b = pd.read_csv(path2 + "event_log_03.csv")
# print(df5b.head(3))
duration = df5b['end_datetime'].astype('datetime64[ns]') - df5b['start_datetime'].astype('datetime64[ns]')
result5b = int(duration.nlargest(int(len(duration) * 0.25)).sum().total_seconds())
# print(result5b) # 131434

# 작업형 제2유형
# Melbourne Housing Dataset 데이터셋
# 호주 멜버른 지역의 주택 거래 정보를 담고 있는 실제 부동산 데이터입니다. 이 데이터는 주택 판매 기록을 포함하고 있으며, 
# 면적, 방 개수, 건축 연도, 위치 정보 등 다양한 요인을 바탕으로 주택 가격을 분석할 수 있습니다.
# 제공된 학습용 데이터(mhousing_train.csv)를 이용하여 판매된 가격(Price)를 예측하는 모델을 개발하고, 
# 개발한 모델에 기반하여 평가용 데이터(mhousing_test.csv)에 적용하여 얻은 판매된 가격 예측 값을 
# 아래 [제출형식]에 따라 csv 파일로 생성하여 제출하시오.
# 예측 결과는 RMSLE(Root Mean Squared Log Error) 평가지표에 따라 평가함
# 성능이 우수한 예측 모델을 구축하기 위해서는 데이터 정제, Feature Engineering, 하이퍼 파라미터(hyper parameter) 최적화, 
# 모델 비교 등이 필요할 수 있음. 다만, 과적합에 유의하여야 함
# [[제출 형식]]
# 가. CSV 파일명: result.csv(파일명에 디렉토리/폴더 지정불가
# 나. 예측 칼럼명 : pred
# 다. 제출 칼럼 개수 : pred 칼럼 1개
# 라. 평가용 데이터 개수와 예측 결과 데이터 개수 일치 : 1,859개
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
pd.options.display.float_format = "{:.0f}".format

# 데이터 확인
train = pd.read_csv(path1 + "mhousing_train.csv")
test = pd.read_csv(path1 + "mhousing_test.csv")
# print(train.info(), test.info(), sep="\n") # 4337, 1859

# 데이터 전처리
X = train.drop(columns=['Price'])
Y = train['Price']
X_all = pd.concat([X, test])
X_all['Year'] = X_all['Date'].astype('datetime64[ns]').dt.year
X_all['Month'] = X_all['Date'].astype('datetime64[ns]').dt.month
X_all['Day'] = X_all['Date'].astype('datetime64[ns]').dt.day
X_all['Weekday'] = X_all['Date'].astype('datetime64[ns]').dt.weekday
X_all = X_all.drop(columns=['Date'])
X_all['Address'] = LabelEncoder().fit_transform(X_all['Address'])
X_all['SellerG'] = LabelEncoder().fit_transform(X_all['SellerG'])
X_all['CouncilArea'] = LabelEncoder().fit_transform(X_all['CouncilArea'])
X_all['Regionname'] = LabelEncoder().fit_transform(X_all['Regionname'])
X_all = pd.get_dummies(X_all, drop_first=True, dtype='int32')

# 데이터 재분할
X = X_all.iloc[:len(X), :]
X_submission = X_all.iloc[len(X):, :]
# print(X.shape, X_submission.shape) # (4337, 22) (1859, 22)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=600)
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (3035, 22) (1302, 22) (3035,) (1302,)

# 모델사전
models = {
    "Linear": Pipeline([
        ("scaler", MinMaxScaler()), ("model", LinearRegression())
    ]),
    "DecisionTree": Pipeline([
        ("model", DecisionTreeRegressor(max_depth=15, random_state=600))
    ]),
    "RandomForest": Pipeline([
        ("model", RandomForestRegressor(n_estimators=600, max_depth=15, random_state=600))
    ]),
    "AdaBoost": Pipeline([
        ("model", AdaBoostRegressor(n_estimators=600, random_state=600))
    ]),
    "Gradient": Pipeline([
        ("model", GradientBoostingRegressor(n_estimators=600, random_state=600))
    ])
}

# 성능평가 함수
def get_scores(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    y_pred1 = np.where(model.predict(x_train) < 0, abs(model.predict(x_train)), model.predict(x_train))
    y_pred2 = np.where(model.predict(x_test) < 0, abs(model.predict(x_test)), model.predict(x_test))
    RMSLE_train = np.sqrt(MSLE(y_train, y_pred1))
    RMSLE_test = np.sqrt(MSLE(y_test, y_pred2))
    return model, RMSLE_train, RMSLE_test

# 모델별 성능평가
results = []
for name, model in models.items():
    model, RMSLE_train, RMSLE_test = get_scores(model, x_train, x_test, y_train, y_test)
    results.append({
        "Model": name, "RMSLE_train": f"{RMSLE_train:.4f}", "RMSLE_test": f"{RMSLE_test:.4f}"
    })
res = pd.DataFrame(results).sort_values("RMSLE_test").reset_index(drop=True)
print(res) # RandomForest 0.0904 0.2187

# 모델적용, 예측값 산출
model = models[res.loc[0, "Model"]]
y_pred = abs(model.predict(X_submission))

# 제출파일 생성
# pd.DataFrame({'pred': y_pred}).to_csv("result_day7_type2.csv", index=False)

# 결과확인
# temp = pd.read_csv("result_day7_type2.csv")
# print(Y[:len(test)].describe())
# print("=" * 35)
# print(temp['pred'].describe())
