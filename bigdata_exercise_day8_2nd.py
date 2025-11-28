
# 공통
path1 = "https://raw.githubusercontent.com/Soyoung-Yoon/bigdata/main/"
path2 = "https://raw.githubusercontent.com/Soyoung-Yoon/data_01/main/"

# 작업형 제1유형
import pandas as pd
pd.set_option('display.width', 150)
# 8-1) 레스토랑의 팁(tip)분석
# 다음 조건을 만족하는 분석을 수행하시오.
# 요일(day)별 total_bill의 평균이 전체 total_bill 평균보다 높은 요일의 데이터를 추출하시오.
# 위에서 추출된 데이터를 사용하여, day와 smoker로 그룹화하여 tip의 평균을 구하고,
# 가장 평균 tip이 높은 (day, smoker) 조합을 찾으시오.
# 위의 가장 평균 tip이 높은 조합의 데이터에 대해, total_bill의 표준편차를 구하시오.
# 표준편차 값을 반올림하여 소수점 아래 3 자리까지 출력하시오.
df1 = pd.read_csv(path1 + "tips.csv")
# print(df1.head(3))
s1 = df1['total_bill'].mean()
group1 = df1.groupby('day')['total_bill'].mean()
days = group1[group1.values > s1].index.values
cond1 = df1[df1['day'].isin(days)].groupby(['day','smoker'])['tip'].mean().idxmax()
result1 = df1[(df1['day']==cond1[0]) & (df1['smoker']==cond1[1])]['total_bill'].std(ddof=1)
# print(round(result1, 3)) # 10.443

# 8-2a) 프로모션 기획
# 온라인 플랫폼에서는 특정 요일에 이벤트의 발생 빈도를 활용한 프로모션을 기획하려 한다.
# 다음 절차에 따라 문제를 해결하시오.
# 시계열 데이터를 이용해 각 이벤트가 발생한 '요일'을 파생 변수로 추가한 뒤,
# 요일별 value 총합을 구하시오.
# 위의 데이터를 사용하여, 가장 큰 3개 값의 평균을 구해 반올림하여 정수로 출력하시오.
df2a = pd.read_csv(path2 + "event_log_04.csv")
# print(df2a.head(3))
df2a['요일'] = df2a['date'].astype('datetime64[ns]').dt.day_name('ko_KR')
group2a = df2a.groupby('요일')['value'].sum()
result2a = round(group2a.nlargest(3).mean())
# print(result2a) # 8342

# 8-2b) 프로모션 기획
# 온라인 플랫폼에서는 특정 요일에 이벤트의 발생 빈도를 활용한 프로모션을 기획하려 한다.
# 다음 절차에 따라 문제를 해결하시오.
# 각 이벤트가 발생한 날짜에서 '요일' 정보를 추출한다.
# 요일별로 value의 총합을 구하여, 가장 높은 합계를 가진 요일 3개를 식별한다.
# 이 3개 요일에 해당하는 모든 이벤트의 value 평균을 계산하고, 그 값을 반올림하여 정수로 출력하시오.
df2b = pd.read_csv(path2 + "event_log_04.csv")
# print(df2b.head(3))
df2b['요일'] = df2b['date'].astype('datetime64[ns]').dt.day_name('ko_KR')
days = df2b.groupby('요일')['value'].sum().nlargest(3).index.values
result2b = df2b[df2b['요일'].isin(days)]['value'].mean()
# print(round(result2b)) # 57

# 8-3) 이벤트별 월간 성과
# 다양한 이벤트 카테고리별 월간 성과 분석을 위한 과제이다.
# 다음 절차에 따라 문제를 해결하시오.
# 주어진 데이터에 대해 카테고리(category2), 연도(year), 월(month) 단위로 value의 총합을 집계하시오.
# 이후, (category2, year)별로 집계된 값 중 가장 value 합계가 높은 월을 선택하시오.
# 이렇게 선택된 월별 최댓값들을 모두 더한 뒤, 그 합계를 반올림하여 정수로 출력하시오.
df3 = pd.read_csv(path2 + "event_log_05.csv")
# print(df3.head(3))
df3['year'] = df3['date'].astype('datetime64[ns]').dt.year
df3['month'] = df3['date'].astype('datetime64[ns]').dt.month
# group3 = df3.pivot_table(index=['category2','year'], columns='month', values='value', aggfunc='sum')
group3 = df3.groupby(['category2','year','month'])['value'].sum()
result3 = round(group3.unstack().max(axis=1).sum())
# print(result3) # 7605

# 8-4) 이상 감지 및 성능 분석
# 시스템 비활성 이벤트의 이상 감지 및 성능 분석을 위한 과제이다.
# 다음 절차에 따라 문제를 해결하시오.
# 2020년 1월부터 6월까지의 데이터 중 event가 'Idle'인 이벤트만 필터링하시오.
# 월(month)별로 해당 이벤트들의 value 중앙값을 구하시오.
# 해당 월의 value 중앙값을 기준으로 그 달에 발생한 이벤트 중 value가 중앙값 이하인 이벤트만 선택하시오.
# 이렇게 선택된 이벤트들의 총 개수를 정수로 출력하시오.
df4 = pd.read_csv(path2 + "event_log_04.csv")
# print(df4.head(3))
df4['date'] = df4['date'].astype('datetime64[ns]')
df4['year'] = df4['date'].dt.year
df4['month'] = df4['date'].dt.month
df4 = df4[(df4['year'] == 2020) & (df4['month'] >= 1) & (df4['month'] <= 6) & (df4['event']=='Idle')]
# group1 = filter1.groupby('month')['value'].median().reset_index(name='median')
# df4 = df4.merge(group1, "left", "month").dropna()
# result4 = int(df4[df4['value'] <= df4['median']]['event'].count())
s = df4.groupby('month')['value'].transform('median')
result4 = (sum(df4['value'] <= s))
# print(result4) # 16

# 8-5) 야간 시간대 이벤트
# 다음은 야간 시간대에 발생한 이벤트 유형 분석을 통해 사용자 활동 패턴을 파악하고자 하는 과제이다.
# 다음 절차에 따라 문제를 해결하시오.
# 먼저, 시간(hour) 정보가 22시부터 02시까지(22:00:00 ~ 02:59:59)인 데이터를 야간 이벤트로 간주한다.
# 이 야간 이벤트를 기준으로, (month, event)별 value 평균값을 구하시오.
# 각 월(month)에서 value 평균이 가장 높은 이벤트 유형을 하나씩 선택하고, 그 중 'Start'와 'Stop'이벤트의 빈도 수 합을 정수로 출력하시오.
df5 = pd.read_csv(path2 + "event_log_04.csv")
# print(df5.head(3))
df5['date_time'] = df5['date'] + ' ' + df5['time']
df5['date_time'] = df5['date_time'].astype('datetime64[ns]')
df5 = df5[(df5['date_time'].dt.hour >= 22) | (df5['date_time'].dt.hour < 3)].copy()
# print(night)
df5['month'] = df5['date_time'].dt.month
# max_event = df5.pivot_table(index='month', columns='event', values='value', aggfunc='mean').fillna(0)
s = df5.groupby(['month','event'])['value'].mean()
s = s.unstack().idxmax(axis=1).value_counts()
result5 = s['Start'] + s['Stop']
# print(result5) # 7

# 작업형 제2유형 10회 기출복원문제
path3 = "https://raw.githubusercontent.com/YoungjinBD/data/main/exam/"
# 연간 총 가스 소비량(회귀) 예측하는 모델 개발
# 라이브러리
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error as RMSE
from sklearn.metrics import mean_squared_error as MSE
pd.options.display.float_format = "{:.0f}".format

# 데이터 확인
train = pd.read_csv(path3 + "10_2_train.csv")
test = pd.read_csv(path3 + "10_2_test.csv")
# print(train.info(), test.info(), sep=", ")

# 데이터 전처리
X_all = pd.concat([train, test])
Y = train['gas_totl']
X_all = X_all.drop(columns=['gas_totl'])
# print(X_all['biz_type'].unique())
X_all = pd.get_dummies(X_all, drop_first=True, dtype='int32')

# 데이터 재분할
X = X_all.iloc[:len(train), :]
X_submission = X_all.iloc[len(train):, :]
# print(X.shape, X_submission.shape) # (160, 7) (40, 7)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (128, 7) (32, 7) (128,) (32,)

# 모델사전
models = {
    "Linear": Pipeline([
        ("scaler", MinMaxScaler()), ("model", LinearRegression())
    ]),
    "DecisionTree": Pipeline([
        ("model", DecisionTreeRegressor(max_depth=5, random_state=1))
    ]),
    "RandomForest": Pipeline([
        ("model", RandomForestRegressor(max_depth=5, random_state=1))
    ]),
    "AdaBoost": Pipeline([
        ("model", AdaBoostRegressor(n_estimators=10, random_state=1))
    ]),
    "Gradient": Pipeline([
        ("model", GradientBoostingRegressor(random_state=1))
    ])
}

# 성능평가 함수
def get_scores(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    y_pred1 = model.predict(x_train)
    y_pred2 = model.predict(x_test)
    RMSE_train1 = np.sqrt(MSE(y_train, y_pred1))
    RMSE_train2 = RMSE(y_train, y_pred1)
    RMSE_test1 = np.sqrt(MSE(y_test, y_pred2))
    RMSE_test2 = RMSE(y_test, y_pred2)
    return model, RMSE_train1, RMSE_train2, RMSE_test1, RMSE_test2

# 모델별 성능평가
results = []
for name, model in models.items():
    model, RMSE_train1, RMSE_train2, RMSE_test1, RMSE_test2 = get_scores(model, x_train, x_test, y_train, y_test)
    results.append({
        "Model": name, "RMSE_train1": f"{RMSE_train1:.4f}", "RMSE_train2": f"{RMSE_train2:.4f}", "RMSE_test1": f"{RMSE_test1:.4f}", "RMSE_test2": f"{RMSE_test2:.4f}"
    })
res = pd.DataFrame(results).sort_values(["RMSE_test1","RMSE_test2"]).reset_index(drop=True)
print(res)

# 모델적용
model = models[res.loc[0, "Model"]]
y_pred = model.predict(X_submission)

# 제출파일 생성
# pd.DataFrame({'pred': y_pred}).to_csv("result_10th.csv", index=False)

# 결과확인
# temp = pd.read_csv("result_10th.csv")
# print(Y[:len(test)].describe())
# print("=" * 35)
# print(temp['pred'].describe())

# 작업형 제3유형

# 8회 1번. 로지스틱 회귀
# 고객의 서비스 이용 특성과 이탈 여부(Churn)를 포함한 데이터를 바탕으로 이탈여부 예측을 위한 
# 로지스틱 회귀 분석을 수행합니다.
# 종속변수: churn (이탈 여부)
# 독립변수: churn을 제외한 모든 변수
# 단, 상수항(절편)을 포함하고, 규제는 적용하지 않는다.
# 위의 조건에 따라 모형을 생성, 학습하여 다음 물음에 답하시오.
# 유의수준 0.05 하에서, 유의성이 낮은 변수의 개수를 구하라.
# 위 결과에서 통계적으로 유의한 변수들만 사용하여 다시 모델을 구축한 뒤, 
# 회귀식의 모든 계수(절편 포함)의 평균을 구하라.
# number_customer_calls가 5증가하면 오즈비는 몇 배 증가하는가?
# 단, 실수의 경우 반올림하여 소수점 아래 3자리까지 표시한다.
pd.options.display.float_format = "{:.4f}".format
df1 = pd.read_csv(path1 + "churn_08031.csv")
# print(df1.head(3))
cols = []
for col in df1.columns:
    cols.append(col.replace(' ', ''))
df1.columns = cols
from statsmodels.api import GLM, families
formula = "churn ~ " + " + ".join(df1.columns[:-1])
# print(formula)
model = GLM.from_formula(formula, df1, family=families.Binomial()).fit()
# print(model.summary())
result1 = sum(model.pvalues[1:] > 0.05)
# print(result1) # 12
result2 = model.pvalues[1:] <= 0.05
# print(result2) # ['vmail_message','intl_calls','number_customer_calls']
formula2 = "churn ~ vmail_message + intl_calls + number_customer_calls"
model2 = GLM.from_formula(formula2, df1, family=families.Binomial()).fit()
result3 = round(model2.params.mean(), 3)
# print(result3) # -0.435
result4 = round(np.exp(model2.params['number_customer_calls'] * 5), 3)
# print(result4) # 6.511

# 8회 2번. 다중선형회귀
# 신체 정보에 대한 데이터를 바탕으로 개인의 인지 지수(PIQ)를 예측하기 위한 다중 선형 회귀 분석을 수행합니다.
# 종속변수: PIQ
# 독립변수: Brain, Height, Weight
# 위의 조건에 따라 모형을 생성, 학습하여 다음 물음에 답하시오.
# 위의 모델에서 통계적으로 가장 유의미한 변수의 회귀계수를 구하시오.
# 위 모델의 결정계수 값을 구하시오.
# 위에서 구한 모델을 사용하여 키:70, 몸무게:150, 뇌크기:90 인 경우의 PIQ 값을 구하시오.
# 단, 실수의 경우 반올림하여 소수점 아래 3자리까지 표시한다.
df2 = pd.read_csv(path1 + "iqsize_08032.csv")
# print(df2.head(3))
from statsmodels.api import OLS
formula = "PIQ ~ Brain + Height + Weight"
model = OLS.from_formula(formula, df2).fit()
# print(model.summary())
temp = (model.pvalues[1:] <= 0.05).idxmax()
result1 = round(model.params[temp], 3)
# print(result1) # 2.028
result2 = round(model.rsquared, 3)
# print(result2) # 0.313
data = pd.DataFrame({'Height': [70], 'Weight': [150], 'Brain': [90]})
PIQ = round(model.predict(data)[0], 3)
# print(PIQ) # 104.768

# 9회 1번. 다중선형회귀
# 미술학과 학생들의 시험 성적 데이터를 바탕으로 디자인 과목 점수(design)를 예측하기 위한 다중 선형 회귀 분석을 수행합니다.
# 해당 데이터를 다음 기준에 따라 훈련용 데이터(train)와 평가용 데이터(test)로 나누어 사용한다.
# train : 140개, test : 나머지 데이터
# 종속변수: design (디자인 과목 점수)
# 독립변수: drawing, color_theory, composition, modeling_3D
# 위의 조건에 따라 모형을 생성, 학습하여 다음 물음에 답하시오.
# 유의수준 0.05 하에서 통계적으로 유의한 변수의 개수는 몇 개인가?
# (1)에서 구한 유의한 변수만 사용하여 다시 모델을 만들고, train 데이터를 사용하여 해당 모델의 예측값과 실제값의 피어슨 상관계수를 구하여라.
# (2)에서 생성된 모델을 사용해 test 데이터에 대한 rmse를 구하여라.
# 단, 실수의 경우 반올림하여 소수점 아래 3자리까지 표시한다.
df3 = pd.read_csv(path1 + "design_data.csv")
# print(df3.head(3), df3.shape) # (210, 6)
train = df3.iloc[:140, :]
test = df3.iloc[140:, :]
# print(train.shape, test.shape) # (140, 6) (70, 6)
from statsmodels.api import OLS
formula = "design ~ drawing + color_theory + composition + modeling_3D"
model = OLS.from_formula(formula, train).fit()
# print(model.summary())
result1 = sum(model.pvalues[1:] <= 0.05)
# print(result1) # 2
# print((model.pvalues[1:] <= 0.05).head(2).index) # 'drawing', 'color_theory'
formula2 = "design ~ drawing + color_theory"
model2 = OLS.from_formula(formula2, train).fit()
y_true = train['design']
y_pred = model2.predict(train)
temp = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
result2 = round(temp.corr().iloc[0,1], 3)
# print(result2) # 0.802
from sklearn.metrics import root_mean_squared_error as rmse
y_true = test['design']
y_pred = model.predict(test)
result3 = round(rmse(y_true, y_pred), 3)
print(rmse(test['design'], model.predict(test)))
# print(result3) # 7.816

# 9회 2번. 로지스틱 회귀
# 고객의 서비스 이용 특성과 이탈 여부(Churn)를 포함한 데이터를 바탕으로 이탈여부 예측을 위한 로지스틱 회귀 분석을 수행합니다.
# 종속변수: Churn(이탈여부, 이진 변수)
# 독립변수: MonthlyCharges, ServiceSatisfactionScore, FamilyPlanSize, PhoneService
# 단, 상수항(절편)을 포함하고, 규제는 적용하지 않는다.
# 위의 조건에 따라 모형을 생성, 학습하여 다음 물음에 답하시오.
# 위 모델에서'FamilyPlanSize'에 대한 p-value를 구하시오.
# 위 모델에서 다른 변수의 영향은 무시하고 'PhoneService' 변수가 0에서 1이 됐을 때의 오즈비를 구하시오.
# 생성된 모델에 데이터를 넣어 고객이 이탈 할 확률값을 예측한 후, 그 확률값이 0.4보다 큰 고객의 수를 구하여 정수로 출력하시오.
# 단, 실수의 경우 반올림하여 소수점 아래 3자리까지 표시한다.
df4 = pd.read_csv(path1 + "churn_delco.csv")
# print(df4.head(3))
from statsmodels.api import GLM, families
formula = "Churn ~ MonthlyCharges + ServiceSatisfactionScore + FamilyPlanSize + PhoneService"
model = GLM.from_formula(formula, df4, family=families.Binomial()).fit()
# print(model.summary())
result1 = round(model.pvalues['FamilyPlanSize'], 3)
# print(result1) # 0.042
import numpy as np
result2 = round(np.exp(model.params['PhoneService']), 3)
# print(result2) # 0.688
proba = model.predict(df4)
result3 = sum(proba>0.4)
# print(result3) # 426