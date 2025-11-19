
# 작업형 제1유형
import pandas as pd
pd.set_option('display.width', 120)
pd.options.display.float_format = "{:.3f}".format
path1 = "https://raw.githubusercontent.com/Soyoung-Yoon/bigdata/main/"
path2 = "https://raw.githubusercontent.com/Soyoung-Yoon/data_01/main/"

# 9-1) 유병률 증가
# worlddata.csv 파일을 사용하여 다음의 사항을 처리합니다.
# 1999년과 2000년 간 유병률이 5배 이상 증가한 국가를 모두 선택합니다.
# 이 중 유병률 증가 비율이 가장 작은 하위 4개 국가의 유병률 증가 비율의 평균을 구해, 
# 반올림하여 소수점 아래 세자리까지 출력합니다.
# 유병률 증가 비율 = 2000년_유병률 / 1999년_유병률
# 단, 1999년의 유병률이 0인 국가는 제외합니다.
df = pd.read_csv(path1 + "worlddata.csv")
# print(df.head(3))
# 1999년과 2000년 간 유병률이 5배 이상 증가한 국가를 모두 선택
df = df.set_index('year').T
# print(df)
df = df[df[1999] != 0]
s = df[2000] / df[1999]
s = s[s >= 5]
# print(round(s.sort_values().iloc[:4].mean(), 3)) # 6.458

# 9-2) 시급과 기본급
# salary01.csv를 사용하여 시급이 가장 높은 사람의 기본급을 정수로 구합니다.
# 단, 다음 컬럼들의 '-' 값을 0으로 변경하여 사용합니다.
#  - totalSalary, specialSalary, numberOfWorker : 정수
#  - workTime : 실수
# totalSalary 또는 workTime이 0인 데이터는 제외하고 기본급과 시급을 구합니다.
# 기본급 = totalSalary - specialSalary
# 시급 = 기본급 / workTime
df = pd.read_csv(path2 + "salary01.csv")
# print(df.head(3))
# print(df['totalSalary'].unique())
for col in ['totalSalary', 'specialSalary', 'numberOfWorker', 'workTime']:
    df[col] = df[col].replace('-', 0)
    if col != 'workTime':
        df[col] = df[col].astype('int32')
    else:
        df[col] = df[col].astype('float')
df = df[(df['totalSalary'] > 0) | (df['workTime'] > 0)]
df['기본급'] = df['totalSalary'] - df['specialSalary']
df['시급'] = df['기본급'] / df['workTime']
# print(int(df[df.index == df['시급'].idxmax()]['기본급'].values)) # 4047478

# 작업형 제2유형(day4)
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
# - 가. CSV 파일명: result.csv(파일명에 디렉토리/폴더 지정불가
# - 나. 예측 칼럼명 : pred
# - 다. 제출 칼럼 개수 : pred 칼럼 1개
# - 라. 평가용 데이터 개수와 예측 결과 데이터 개수 일치 : 1,859개
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

# 데이터 확인
train = pd.read_csv(path1 + "mhousing_train.csv")
test = pd.read_csv(path1 + "mhousing_test.csv")
# print(train.head(3), train.info(), sep="\n") # 4337
# print(test.head(3), test.info(), sep="\n")   # 1859

# 데이터 전처리
X = train.drop(columns = ['Price'])
Y = train['Price']
X_all = pd.concat([X, test])
# print(X_all.info())
X_all['Date'] = X_all['Date'].astype('datetime64[ns]')
X_all['Year'] = X_all['Date'].dt.year
X_all['Month'] = X_all['Date'].dt.month
X_all['Day'] = X_all['Date'].dt.day
X_all['Weekday'] = X_all['Date'].dt.day_name('ko_KR')
X_all = X_all.drop(columns=['Date'])
cols_obj = X_all.select_dtypes(include='object').columns
for col in cols_obj:
    X_all[col] = LabelEncoder().fit_transform(X_all[col])
# print(X_all.head(3))

# 데이터 재분할
X = X_all.iloc[:len(X), :]
X_submission = X_all.iloc[len(X):, :]
# print(X.shape, X_submission.shape) # (4337, 21) (1859, 21)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1234)
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (3035, 21) (1302, 21) (3035,) (1302,)

# 파이프라인 모델사전
models = {
    "Linear": Pipeline([
        ("scaler", MinMaxScaler()), ("model", LinearRegression())
    ]),
    "DecisionTree": Pipeline([
        ("model", DecisionTreeRegressor(max_depth=5, random_state=1234))
    ]),
    "RandomForest": Pipeline([
        ("model", RandomForestRegressor(max_depth=5, random_state=1234))
    ]),
    "AdaBoost": Pipeline([
        ("model", AdaBoostRegressor(n_estimators=100, random_state=1234))
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
res = pd.DataFrame(results).sort_values("RMSLE_test").reset_index(drop=True)
# print(res)

# 모델적용, 예측값 산출
model = models[res.loc[0, "Model"]]
y_pred = model.predict(X_submission)

# 결과파일 생성
# pd.DataFrame({'pred': y_pred}).to_csv("result.csv", index=False)

# 결과확인
# temp = pd.read_csv("result.csv")
# print(Y[:len(test)].describe())
# print("=" * 35)
# print(temp['pred'].describe())

# 작업형 제3유형
# 정규모집단으로부터 추출한 표본이 있다. 남자의 키 평균에 대한 일표본t-검정(one sample t-test)을 통해 답하고자 한다. 가설은 아래와 같다.
# 문제지 참고
# 다음에 대해 유의수준 0.05로 일표본 t-검정한다.
# 성별(gender)은 남자=1, 여자=0의 값을 갖는다.
# 3개의 값을 1개 행에 공백을 구분자로 하여 출력한다.
# 1. 위의 가설을 검정하기 위한 검정통계량을 입력하시오.(반올림하여 소수 넷째자리까지 계산)
# 2. 위의 통계량에 대한 p-값을 구하여 입력하시오. (반올림하여 소수 넷째자리까지 계산)
# 3. 유의수준 0.05 하에서 가설 검정의 결과를 (채택/기각) 중 하나를 선택하여 입력하시오.
import pandas as pd
import scipy.stats as stats
s = pd.Series(dir(stats))
cond = s.str.startswith('ttest')
# print(s[cond])
from scipy.stats import ttest_1samp
path3 = "https://raw.githubusercontent.com/Soyoung-Yoon/data_02/main/"
df = pd.read_csv(path3 + "human_traits_sample.csv")
# print(df.head())
male = df[df['gender']==1]['height_cm'].to_numpy()
statistic, pvalue = ttest_1samp(male, 171, alternative="less")
# print(f"검정통계량: {statistic:.4f}")
# print(f"p-value: {pvalue:.4f}")
# print("기각" if round(pvalue, 4) <= 0.05 else "채택")
# 검정통계량: -2.0328
# p-value: 0.0266
# 기각

# 2. 이표본 t-검정
# 정규모집단으로부터 추출한 표본이 있다.
# 여자 그룹의 체중(weight_kg)이 남자 그룹의 체중보다 큰지 이표본 t-검정(two sample t-test)를 통해 답하고자 한다. 
# 가설은 아래와 같다.
# - $H_0: \mu_1 \le \mu_2 \quad$
# - $H_1: \mu_1 > \mu_2 \quad$ ($\mu_1$: 여자그룹의 평균 체중 $\mu_2$: 남자그룹의 평균 체중)
# 다음의 사항을 참고하여 답안을 작성하시오.
# 성별(gender)의 값은 여자=0, 남자=1로 되어 있다.
# 4개의 값을 1개 행에 공백을 구분자로 하여 출력한다.
# 등분산성 검정을 bartlett을 사용하여 수행한다.
# 1. 두 그룹의 등분산 검정 결과를 (등분산/이분산) 중 하나를 선택하여 입력하시오.
# 2. 위의 t-검정 관련 가설을 검정하기 위한 검정통계량을 구하시오. (반올림하여 소수점아래 셋째자리까지 계산)
# 3. 위의 검정통계량에 대한 p-value를 구하라. (반올림하여 소수점아래 셋째자리까지 계산)
# 4. 유의수준 0.05 하에서 가설 검정의 결과를 (채택/기각) 중 하나를 선택하여 입력하시오.
import pandas as pd
df = pd.read_csv(path3 + "human_traits_sample.csv")
# print(df.head(3))
from scipy.stats import bartlett, ttest_ind
# print(help(bartlett))
female = df.loc[df['gender']==0, 'weight_kg']
male = df.loc[df['gender']==1, 'weight_kg']
statistic, pvalue = bartlett(female, male)
result = "등분산" if pvalue > 0.05 else "이분산"
statistic, pvalue = ttest_ind(female, male, alternative="greater", equal_var=True)
result2 = "기각" if pvalue <= 0.05 else "채택"
# print(result, round(statistic, 3), round(pvalue, 3), result2)
# 등분산 0.728 0.235 채택

# 3. Paired t-검정
# 정규모집단으로부터 추출한 표본이 있다. 교육 전/후의 시험 점수 평균에 대해 t검정을 실시한다.
# 학생 30명의 교육 전후의 점수가 저장되어 있다. 해당 교육이 효과가 있는지 (즉, 학습 후의 점수가 증가했는지) 
# 쌍체표본 t-검정(paired t-test)를 통해 답하고자 한다. 가설은 아래와 같다
# - $\mu_d$ : (교육 후 점수 - 교육 전 점수)의 평균
# - $H_0$ : $\mu_d \leq 0 \quad$ (교육 점수 감소 또는 효과 없음)
# - $H_1$ : $\mu_d > 0 \quad$(교육 후 점수 증가)
# 다음에 대해 유의수준 0.05로 Paired t-검정한다.
# 4개의 값을 1개 행에 공백을 구분자로 하여 출력한다.
# 1. 위의 가설을 검정하기 위한 검정통계량을 구하시오. (반올림하여 소수점아래 셋째자리까지 계산)
# 2. 위의 검정통계량에 대한 p-value를 구하시오. (반올림하여 소수점아래 넷째자리까지 계산)
# 3. 유의수준 0.05 하에서 가설검정의 결과를 (채택/기각) 중 하나를 선택하여 입력하시오.
# 4. 가설 검정의 결과 교육 후 시험 점수 변화를 (증가/감소) 중 하나를 선택하여 입력하시오.
import pandas as pd
df = pd.read_csv(path3 + "paired_score.csv")
# print(df.head(3))
before = df['score_before']
after = df['score_after']
from scipy.stats import ttest_rel
statistic, pvalue = ttest_rel(after, before, alternative="greater")
result = "기각 증가" if round(pvalue, 4) <= 0.05 else "채택 감소"
# print(f"검정통계량: {statistic:.3f}, p-value: {pvalue:.4f}, {result}")
# 검정통계량: 3.701, p-value: 0.0004, 기각 증가

# 4. paired t-test
# 주어진 데이터(data/blood_pressure.csv)에는 고혈압 환자 120명의 치료 전후의 혈압이 저장되어 있다. 
# 해당 치료가 효과가 있는지 (즉, 치료 후의 혈압이 감소했는지) 쌍체표본 t-검정(paired t-test)를 통해 답하고자 한다. 
# 가설은 아래와 같다
# $\mu_d$ : (치료 후 혈압 - 치료 전 혈압)의 평균
# $H_0$ : $\mu_d \geq 0 \quad$ (치료 효과 없음 또는 증가)
# $H_1$ : $\mu_d < 0 \quad$(치료 후 혈압 감소)
# 1. $u_d$의 표본 평균을 구하시오 (반올림하여 소수 둘째자리까지 계산)
# 2. 위의 가설을 검정하기 위한 검정통계량을 입력하시오.(반올림하여 소수 넷째자리까지 계산)
# 3. 위의 통계량에 대한 p-값을 구하여 입력하시오. (반올림하여 소수 넷째자리까지 계산)
# 4. 유의수준 0.05 하에서 가설검정의 결과를 (채택/기각) 중 하나를 선택하여 입력하시오.
from scipy.stats import ttest_rel
df = pd.read_csv(path3 + "blood_pressure.csv")

groupA = df['bp_after']
groupB = df['bp_before']
mean = (groupA - groupB).mean()
statistic, pvalue = ttest_rel(groupA, groupB, alternative='less')
result = "기각" if pvalue <= 0.05 else "채택"
# print(f"표본평균: {mean:.2f}, 검정통계량: {statistic:.4f}, p-value: {pvalue:.4f}, {result}")
# 표본평균: -5.09, 검정통계량: -3.3372, p-value: 0.0006, 기각