
# 작업형 제1유형
import pandas as pd
pd.set_option('display.width', 120)
path1 = "https://raw.githubusercontent.com/Soyoung-Yoon/bigdata/main/"
path2 = "https://raw.githubusercontent.com/Soyoung-Yoon/data_01/main/"

# 8-1) 레스토랑의 팁(tip)분석
# 다음 조건을 만족하는 분석을 수행하시오.
# 요일(day)별 total_bill의 평균이 전체 total_bill 평균보다 높은 요일의 데이터를 추출하시오.
# 위에서 추출된 데이터를 사용하여, day와 smoker로 그룹화하여 tip의 평균을 구하고, 가장 평균 tip이 높은 (day, smoker) 조합을 찾으시오.
# 위의 가장 평균 tip이 높은 조합의 데이터에 대해, total_bill의 표준편차를 구하시오.
# 표준편차 값을 반올림하여 소수점 아래 3 자리까지 출력하시오.
df = pd.read_csv(path1 + "tips.csv")
# print(df.head(3), df.shape, sep="\n") # (244, 7)
# 요일(day)별 total_bill의 평균이 전체 total_bill 평균보다 높은 요일의 데이터를 추출
cond = df.groupby('day')['total_bill'].transform('mean')
days = df[cond > df['total_bill'].mean()]['day']
df = df[df['day'].isin(days)]
# 위에서 추출된 데이터를 사용하여, day와 smoker로 그룹화하여 tip의 평균을 구하고, 가장 평균 tip이 높은 (day, smoker) 조합을 찾으시오.
day, smoker = df.groupby(['day','smoker'])['tip'].mean().idxmax()
df = df[(df['day'] == day) & (df['smoker']==smoker)]
# 위의 가장 평균 tip이 높은 조합의 데이터에 대해, total_bill의 표준편차를 구하시오.
# 표준편차 값을 반올림하여 소수점 아래 3 자리까지 출력
# print(round(df['total_bill'].std(ddof=1), 3)) # 10.443

# 8-2a) 프로모션 기획
# 온라인 플랫폼에서는 특정 요일에 이벤트의 발생 빈도를 활용한 프로모션을 기획하려 한다.
# 다음 절차에 따라 문제를 해결하시오.
# 시계열 데이터를 이용해 각 이벤트가 발생한 '요일'을 파생 변수로 추가한 뒤, 요일별 value 총합을 구하시오.
# 위의 데이터를 사용하여, 가장 큰 3개 값의 평균을 구해 반올림하여 정수로 출력하시오.
df = pd.read_csv(path2 + "event_log_04.csv")
# print(df.head(3))
# 시계열 데이터를 이용해 각 이벤트가 발생한 '요일'을 파생 변수로 추가한 뒤, 요일별 value 총합을 구하시오.
df['date'] = df['date'].astype('datetime64[ns]')
df['요일'] = df['date'].dt.day_name('ko_KR')
group = df.groupby('요일')['value'].sum()
# 위의 데이터를 사용하여, 가장 큰 3개 값의 평균을 구해 반올림하여 정수로 출력
# print(round(group.nlargest(3).mean())) # 8342

# 8-2b) 프로모션 기획
# 온라인 플랫폼에서는 특정 요일에 이벤트의 발생 빈도를 활용한 프로모션을 기획하려 한다.
# 다음 절차에 따라 문제를 해결하시오.
# 각 이벤트가 발생한 날짜에서 '요일' 정보를 추출한다.
# 요일별로 value의 총합을 구하여, 가장 높은 합계를 가진 요일 3개를 식별한다.
# 이 3개 요일에 해당하는 모든 이벤트의 value 평균을 계산하고, 그 값을 반올림하여 정수로 출력하시오.
df = pd.read_csv(path2 + "event_log_04.csv")
# print(df.head(3))
# 각 이벤트가 발생한 날짜에서 '요일' 정보를 추출
df['date'] = df['date'].astype('datetime64[ns]')
df['요일'] = df['date'].dt.day_name('ko_KR')
# print(df.head(3))
# 요일별로 value의 총합을 구하여, 가장 높은 합계를 가진 요일 3개를 식별
days = df.groupby('요일')['value'].sum().nlargest(3).index.values
# 이 3개 요일에 해당하는 모든 이벤트의 value 평균을 계산하고, 그 값을 반올림하여 정수로 출력
# print(round(df[df['요일'].isin(days)]['value'].mean())) # 57

# 8-3) 이벤트별 월간 성과
# 다양한 이벤트 카테고리별 월간 성과 분석을 위한 과제이다.
# 다음 절차에 따라 문제를 해결하시오.
# **주어진 데이터**에 대해 카테고리(category2), 연도(year), 월(month) 단위로 value의 총합을 집계하시오.
# 이후, (category2, year)별로 집계된 값 중 가장 value 합계가 높은 월을 선택하시오.
# 이렇게 선택된 월별 최댓값들을 모두 더한 뒤, 그 합계를 반올림하여 정수로 출력하시오.
df = pd.read_csv(path2 + "event_log_05.csv")
# print(df.head(3))
# 카테고리(category2), 연도(year), 월(month) 단위로 value의 총합을 집계
df['date'] = df['date'].astype('datetime64[ns]')
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df2 = df.groupby(['category2','year','month'], as_index=False)['value'].sum()
# 이후, (category2, year)별로 집계된 값 중 가장 value 합계가 높은 월을 선택하시오.
# print(df2.groupby(['category2','year'])['month'].nlargest())

# 이렇게 선택된 월별 최댓값들을 모두 더한 뒤, 그 합계를 반올림하여 정수로 출력하시오.


# 8-4) 이상 감지 및 성능 분석
# 시스템 비활성 이벤트의 이상 감지 및 성능 분석을 위한 과제이다.
# 다음 절차에 따라 문제를 해결하시오.
# 2020년 1월부터 6월까지의 데이터 중 event가 'Idle'인 이벤트만 필터링하시오.
# 월(month)별로 해당 이벤트들의 value 중앙값을 구하시오.
# 해당 월의 value 중앙값을 기준으로 그 달에 발생한 이벤트 중 value가 중앙값 이하인 이벤트만 선택하시오.
# 이렇게 선택된 이벤트들의 총 개수를 정수로 출력하시오.
df = pd.read_csv(path2 + "event_log_04.csv")
# print(df.head(3))
# 2020년 1월부터 6월까지의 데이터 중 event가 'Idle'인 이벤트만 필터링
df['date'] = df['date'].astype('datetime64[ns]')
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df = df[(df['year'] == 2020) & (df['month'] >= 1) & (df['month'] < 7)]
df = df[df['event'] == 'Idle']
# 월(month)별로 해당 이벤트들의 value 중앙값
s = df.groupby('month', as_index=False)['value'].median()
# print(s)
# 해당 월의 value 중앙값을 기준으로 그 달에 발생한 이벤트 중 value가 중앙값 이하인 이벤트만 선택
result = []
for i in s.index:
    result.append(df[(df['month'] == s.loc[i, 'month']) & (df['value'] <= s.loc[i, 'value'])])
res = pd.concat(result, ignore_index=True)
# 이렇게 선택된 이벤트들의 총 개수를 정수로 출력
# print(len(res)) # 16

# 8-5) 야간 시간대 이벤트
# 다음은 야간 시간대에 발생한 이벤트 유형 분석을 통해 사용자 활동 패턴을 파악하고자 하는 과제이다.
# 다음 절차에 따라 문제를 해결하시오.
# 먼저, 시간(hour) 정보가 22시부터 02시까지(22:00:00 ~ 02:59:59)인 데이터를 야간 이벤트로 간주한다.
# 이 야간 이벤트를 기준으로, (month, event)별 value 평균값을 구하시오.
# 각 월(month)에서 value 평균이 가장 높은 이벤트 유형을 하나씩 선택하고, 그 중 'Start'와 'Stop'이벤트의 빈도 수 합을 정수로 출력하시오.
df = pd.read_csv(path2 + "event_log_04.csv")
# print(df.head(3), df.shape, sep="\n") # (1000, 4)
# 시간(hour) 정보가 22시부터 02시까지(22:00:00 ~ 02:59:59)인 데이터를 야간 이벤트로 간주
df['datetime'] = df['date'] + ' ' + df['time']
df['datetime'] = df['datetime'].astype('datetime64[ns]')
df = df[(df['datetime'].dt.hour >= 22) | (df['datetime'].dt.hour < 3)]
# 이 야간 이벤트를 기준으로, (month, event)별 value 평균값을 구하시오.
df['month'] = df['datetime'].dt.month
df = df.groupby(['month','event'])['value'].mean()
# 각 월(month)에서 value 평균이 가장 높은 이벤트 유형을 하나씩 선택


# 작업형 제2유형
# Melbourne Housing Dataset 데이터셋
# 호주 멜버른 지역의 주택 거래 정보를 담고 있는 실제 부동산 데이터입니다. 
# 이 데이터는 주택 판매 기록을 포함하고 있으며, 면적, 방 개수, 건축 연도, 위치 정보 등 다양한 요인을 
# 바탕으로 주택 가격을 분석할 수 있습니다.
# 제공된 학습용 데이터(mhousing_train.csv)를 이용하여 판매된 가격(Price)를 예측하는 모델을 개발하고, 
# 개발한 모델에 기반하여 평가용 데이터(mhousing_test.csv)에 적용하여 얻은 판매된 가격 예측 값을 
# 아래 [제출형식]에 따라 csv 파일로 생성하여 제출하시오.
# 예측 결과는 RMSLE(Root Mean Squared Log Error) 평가지표에 따라 평가함
# 성능이 우수한 예측 모델을 구축하기 위해서는 데이터 정제, Feature Engineering, 
# 하이퍼 파라미터(hyper parameter) 최적화, 모델 비교 등이 필요할 수 있음. 다만, 과적합에 유의하여야 함
# [[제출 형식]]
# 가. CSV 파일명: result.csv(파일명에 디렉토리/폴더 지정불가
# 나. 예측 칼럼명 : pred
# 다. 제출 칼럼 개수 : pred 칼럼 1개
# 라. 평가용 데이터 개수와 예측 결과 데이터 개수 일치 : 1,859개
# 라이브러리
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_log_error as MSLE
pd.set_option('display.width', 150)
pd.options.display.float_format = "{:.3f}".format

# 데이터 확인
train = pd.read_csv(path1 + "mhousing_train.csv")
test = pd.read_csv(path1 + "mhousing_test.csv")
# print(train.head(3), train.info(), sep="\n") # 4337
# print(test.head(3), test.info(), sep="\n")   # 1859

# 데이터 전처리
X = train.drop(columns=['Price'])
Y = train['Price']
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
X_all = pd.get_dummies(X_all, drop_first=True, dtype='int32')
# print(X_all.head())

# 데이터 재분할
X = X_all.iloc[:len(X), :]
X_submission = X_all.iloc[len(X):, :]
# print(X.shape, X_submission.shape) # (4337, 21) (1859, 21)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1234)
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (3035, 21) (1302, 21) (3035,) (1302,)

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
    RMSLE_train = MSLE(y_train, y_pred1) ** 0.5
    RMSLE_test = MSLE(y_test, y_pred2) ** 0.5
    return model, RMSLE_train, RMSLE_test

# 모델별 성능확인
results = []
for name, model in models.items():
    model, RMSLE_train, RMSLE_test = get_scores(model, x_train, x_test, y_train, y_test)
    results.append({
        "Model": name, "RMSLE_train": f"{RMSLE_train:.4f}", "RMSLE_test": f"{RMSLE_test:.4f}"
    })
res = pd.DataFrame(results).sort_values("RMSLE_test", ascending=False).reset_index(drop=True)
# print(res)

# 모델적용
model = models[res.loc[0, "Model"]]
y_pred = model.predict(X_submission)

# 제출파일 생성
pd.DataFrame({'pred': y_pred}).to_csv("result_day8_type2.csv", index=False)

# 결과확인
temp = pd.read_csv("result_day8_type2.csv")
print(Y[:len(test)].describe())
print("=" * 35)
print(temp['pred'].describe())
