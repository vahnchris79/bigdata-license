
# 작업형 제1유형
import pandas as pd
pd.set_option('display.width', 120)
path1 = "https://raw.githubusercontent.com/Soyoung-Yoon/bigdata/main/"
path2 = "https://raw.githubusercontent.com/Soyoung-Yoon/data_01/main/"

# 2회-1번) 보스턴 주택 가격

# 보스턴 주택 가격 데이터셋(Boston)을 사용해 극단적인 값이 분석에 미치는 영향을 줄이기 위한 
# 전처리 과정을 수행한 뒤,특정 조건에 해당하는 평균 범죄율을 확인하려 한다.
# 다음 절차에 따라 문제를 해결하시오.
df = pd.read_csv(path1 + "boston.csv")
# print(df.head(3))
# 1) 범죄율(CRIM)컬럼에서 TOP10(즉, 상위 10위)을 찾는다.
s = df['CRIM'].sort_values(ascending=False).head(10)
# print(type(s))
# 2) 이 중 10번째 값으로, 상위 10개의 값을 모두 대체한다.
df.loc[s.index, 'CRIM'] = s.iloc[-1]
# print(df)
# 3) 값을 변경한 데이터에서 AGE 변수가 80이상인 것을 대상으로, 범죄율의 평균을 산출한다.
result = df[df['AGE'] >= 80]['CRIM'].mean()
# 4) 산출된 평균을 반올림하여 소수점 아래 4자리까지 출력한다.
# print(round(result, 4)) # 5.7594

# 2회-2번) 캘리포니아 주택 정보 - 결측값
# 캘리포니아 주택 정보 데이터셋(Housing)의 앞부분만 사용하여 결측값 처리 전후의 변화를 
# 비교해보는 연습을 진행한다.
# 다음 절차에 따라 문제를 해결하시오.
df = pd.read_csv(path1 + "housing.csv")
# print(df.head(3))
# 1) 전체 데이터의 첫 번째 행부터 순서대로 80%에 해당하는 데이터만 추출하여 사용한다.
df = df[:round(len(df) * 0.8)]
# 2) 추출된 데이터에서 total_bedrooms 변수에 결측값이 있다면, 해당 결측값을 total_bedrooms 변수의 
#    중앙값으로 대체한다.
df2 = df.copy()
# print(df2.isna().sum()) total_bedrooms 159개 결측
df2['total_bedrooms'] = df2['total_bedrooms'].fillna(df2['total_bedrooms'].median())
# print(df2.isna().sum())
# 3) 결측값을 대체하기 전과 후의 total_bedrooms 변수의 표준편차를 각각 계산한다.
before_std = df['total_bedrooms'].std(ddof=1)
after_std = df2['total_bedrooms'].std(ddof=1)
# 4) 마지막으로, 대체 전 표준편차 - 대체 후 표준편차의 값을 구하고, 반올림하여 
#    소수점 아래 3자리까지 출력한다.
# print(round(before_std - after_std, 3)) # 1.975

# 2회 3번) 캘리포니아 주택 정보 - 이상값
# 캘리포니아 주택 정보 데이터셋(Housing)을 사용하여 latitude 컬럼의 이상값을 찾아 그 합을 계산한다.
# 다음 절차에 따라 문제를 해결하시오.
df = pd.read_csv(path1 + "housing.csv")
# print(df.head(3))
# 1) 'latitude' 컬럼의 이상값를 찾아, 이상치들의 합을 산출하시오.
# 2) 이상치 기준은 다음과 같다.
#    > 평균 - (표준편차 * 1.5), 평균 + (표준편차 * 1.5)
lower = df['latitude'].mean() - (df['latitude'].std(ddof=1) * 1.5)
upper = df['latitude'].mean() + (df['latitude'].std(ddof=1) * 1.5)
df = df[(df['latitude'] < lower) | (df['latitude'] > upper)]
result = df['latitude'].sum()
# 3) 계산 결과를 반올림하여 정수형으로 출력한다.
# print(round(result)) # 45816

# 10-4) 최다 발생/검거 범죄유형의 빈도 분석
# 연도별로 가장 많이 발생하거나 검거된 범죄유형을 조사하여, 자주 1위를 기록한 유형은 
# 우선적으로 대응하려합니다.
# 다음 절차에 따라 문제를 해결하시오.
df = pd.read_csv(path2 + "crime_data02.csv")
# print(df.head(3))
# 1) 연도별로 '발생건수'에 대해 가장 많이 발생하는 범죄유형을 찾고, 가장 많이 1위를 차지한 범죄유형의 번호를 A라고한다.
# print(df[df['구분']=='발생건수'].set_index('연도').T)
A = df.loc[df['구분']=='발생건수'].drop(columns=['구분']).set_index('연도').idxmax(axis=1)
A = A.value_counts().index[0][4:]
# 2) 연도별로 '검거건수'에 대해 가장 많이 발생하는 범죄유형을 찾고, 가장 많이 1위를 차지한 범죄유형의 번호를 B라고한다.
B = df.loc[df['구분']=='검거건수'].drop(columns=['구분']).set_index('연도').idxmax(axis=1)
B = B.value_counts().index[0][4:]
# 3) A, B의 범죄유형 번호를 정수로 사용하여 두 번호의 합을 구해 출력한다.
# print(int(A) + int(B)) # 17

# 0-5) 연도별 발생 대비 검거 차이 분석
# 범죄 발생과 검거 간의 차이는 치안의 효율성을 나타낸다. 어느 해에 그 차이가 가장 컸는지 확인해 본다.
# 다음 절차에 따라 문제를 해결하시오.
df = pd.read_csv(path2 + 'crime_data02.csv')
# print(df.head(3))
# 1) 연도별로 '발생건수' 합계와 '검거건수' 합계를 구한 뒤, 그 차이를 계산한다.
#    > 차이 = '발생건수' 합계 - '검거건수' 합계
df['합계'] = df.iloc[:, 2:].sum(axis=1)
df1 = df[df['구분']=='발생건수'][['연도','합계']].set_index('연도')
df2 = df[df['구분']=='검거건수'][['연도','합계']].set_index('연도')
result = df1 - df2
# 2) 차이가 가장 큰 연도의 연도값을 정수로 출력한다.
# print(int(result.idxmax().values)) # 2018

# 작업형 제2유형
# Bank Marketing 데이터셋
# 고객의 인구통계적 정보 및 이전 마케팅 이력 데이터를 바탕으로 "이 사람이 정기예금 상품에 가입할 것인가?" 를 
# 예측하는 것이 목적입니다.
# 제공된 학습용 데이터(bank_train.csv)를 이용하여 정기예금 상품에 가입 여부를 예측하는 모델을 개발하고, 
# 개발한 모델에 기반하여 평가용 데이터(bank_test.csv)에 적용하여 얻은 정기예금 상품에 가입 여부 예측 확률을 
# 아래 [제출형식]에 따라 csv 파일로 생성하여 제출하시오.
# 예측 결과는 ROC-AUC 평가지표에 따라 평가함
# 성능이 우수한 예측 모델을 구축하기 위해서는 데이터 정제, Feature Engineering, 
# 하이퍼 파라미터(hyper parameter) 최적화, 모델 비교 등이 필요할 수 있음. 다만, 과적합에 유의하여야 함
# [[제출 형식]]
# 가. CSV 파일명: result.csv(파일명에 디렉토리/폴더 지정불가
# 나. 예측 성별 칼럼명 : pred
# 다. 제출 칼럼 개수 : pred 칼럼 1개
# 라. 평가용 데이터 개수와 예측 결과 데이터 개수 일치 : 1,221개
# 라이브러리
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score as AUC

# 데이터 확인
train = pd.read_csv(path1 + "bank_train.csv")
test = pd.read_csv(path1 + "bank_test.csv")
# print(train.head(3), train.info(), sep="\n") # 3300
# print(test.head(3), test.info(), sep="\n")   # 1221

# 데이터 전처리
X = train.drop(columns=['term_deposit'])
Y = train['term_deposit']
X_all = pd.concat([X, test])
# print(X_all.info())
cols_obj = X_all.select_dtypes(include='object').columns
for col in cols_obj:
    X_all[col] = LabelEncoder().fit_transform(X_all[col])
# print(X_all.head(3))

# 데이터 재분할
X = X_all.iloc[:len(X), :]
X_submission = X_all.iloc[len(X):, :]
# print(X.shape, X_submission.shape) # (3300, 13) (1221, 13)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1234)
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (2310, 13) (990, 13) (2310,) (990,)

# 파이프라인 모델사전
models = {
    "Logistic": Pipeline([
        ("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=500, tol=0.0, random_state=1234))
    ]),
    "DecisionTree": Pipeline([
        ("model", DecisionTreeClassifier(max_depth=10, random_state=1234))
    ]),
    "RandomForest": Pipeline([
        ("model", RandomForestClassifier(n_estimators=500, max_depth=10, random_state=1234))
    ]),
    "AdaBoost": Pipeline([
        ("model", AdaBoostClassifier(n_estimators=500, random_state=1234))
    ]),
    "Gradient": Pipeline([
        ("model", GradientBoostingClassifier(random_state=1234))
    ])
}

# 성능평가 함수
def get_scores(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    y_proba1 = model.predict_proba(x_train)[:, 1]
    y_proba2 = model.predict_proba(x_test)[:, 1]
    AUC_train = AUC(y_train, y_proba1)
    AUC_test = AUC(y_test, y_proba2)
    return model, AUC_train, AUC_test

# 모데별 성능평가
results = []
for name, model in models.items():
    model, AUC_train, AUC_test = get_scores(model, x_train, x_test, y_train, y_test)
    results.append({
        "Model": name, "AUC_train": f"{AUC_train:.4f}", "AUC_test": f"{AUC_test:.4f}"
    })
res = pd.DataFrame(results).sort_values("AUC_test", ascending=False).reset_index(drop=True)
# print(res)

# 모델선택, 예측확률 생성
model = models[res.loc[0, "Model"]]
y_pred = model.predict_proba(X_submission)[:, 1]

# 제출파일 생성
# pd.DataFrame({'pred': y_pred}).to_csv("result.csv", index=False)

# 결과확인
# temp = pd.read_csv("result.csv")
# print(Y[:len(X_submission)].describe())
# print("=" * 35)
# print(temp['pred'].describe())

# 작업형 제3유형
path3 = "https://raw.githubusercontent.com/Soyoung-Yoon/data_02/main/"
# 1. 주사위 눈의 확률
# 주사위를 던져 관찰한 자료를 사용해 주사위의 1부터 6의 눈이 나올 확률이 같은지를 확인하는 검정을 수행해 답하고자 한다. 
# 가설은 다음과 같다.
# - $H_0$ : 주사위의 1부터 6의 눈이 나올 확률이 같다.
# - $H_1$ : 주사위의 1부터 6의 눈이 나올 확률이 다르다.
# 1. 주사위의 1부터 6의 눈이 나올 확률이 다르다는 것을 확인하는 검정을 수행하시오.
df = pd.read_csv(path3 + "dice_play.csv")
from scipy.stats import chisquare
import numpy as np
observed = df['주사위눈'].value_counts().sort_index()
expected = sum(observed) * np.array([1/6] * 6)
# print(expected)
# 2. 위의 검정 결과 얻은 통계값을 입력하시오. (반올림하여 소수점아래 셋째자리까지 계산)
statistic, pvalue = chisquare(observed, expected)
# print(f"검정통계량: {statistic:.3f}, pvalue: {pvalue:.4f}")
# 검정통계량: 13.226, 
# 3. 통계값에 따른 p-value를 입력하시오. (반올림하여 소수점아래 넷째자리까지 계산)
# pvalue: 0.021
# 4. 유의수준 0.05 하에서 가설 검정의 결과를 (채택/기각) 중 하나를 선택하여 입력하시오.
result = "기각 다르다" if pvalue <= 0.05 else "채택 같다."
# 5. 위의 가설 검정 결과 주사위의 1부터 6의 눈이 나올 확률이 (같다/다르다) 중 하나를 선택하여 입력하시오.
# print(round(statistic, 3), round(pvalue, 4), result)

# 2. 취업 현황 비율
# 통계학과의 매년 졸업생의 취업 현황을 조사해보면 대체적으로 45%는 IT 기업에, 20%는 제조기업에, 
# 20%는 금융기관에 취업하고, 나머지 15%는 대학원에 진학한다고 한다. 
# 올 해의 통계학과 졸업생 중 235명을 임의로 선택하여 조사한 결과를 사용하여 예년과 다르게 
# 취업이 이루어졌다고 할 수 있는지 검정을 수행해 답하고자 한다. 가설은 다음과 같다.
# - $H_0$ : 올 해의 졸업생 취업 현황과 기존의 취업 현황 비율은 같다.
# - $H_1$ : 올 해의 졸업생 취업 현황과 기존의 취업 현황 비율은 다르다.
# 1. 올 해의 졸업생 취업 현황과 기존의 취업 현황 비율이 다르다는 것을 확인하는 검정을 수행한다.
# 2. 위의 검정결과 얻은 통계값을 입력하시오. (반올림하여 소수점아래 넷째자리까지 계산)
# 2. 통계값에 따른 p-value를 입력하시오. (반올림하여 소수점아래 넷째자리까지 표기)
# 3. 유의수준 0.025 하에서 가설 검정의 결과를 (채택/기각) 중 하나를 선택하여 입력하시오.
from scipy.stats import chisquare
path4 = "https://raw.githubusercontent.com/Soyoung-Yoon/data_02/"
df = pd.read_csv(path4 + "main/job.csv")
# print(df.head(3))
observed = df['취업현황'].value_counts()
# print(observed)
expected = sum(observed) * np.array([0.45, 0.2, 0.15, 0.2])
# print(expected)
statistic, pvalue=chisquare(observed, expected)
result = "기각" if pvalue <=0.025 else "채택"
# print(round(statistic, 4), round(pvalue, 4), result) # 18.1655 0.0004 기각

# 3. 질환과 교통위반
# 당뇨병, 심장질환, 간질 등의 질환을 가지고 있는 운전자 집단과 질환을 갖지 않은 운전자들 집단 대한 
# 교통 위반에 대한 기록이 파일로 주어져 있다.
# 두 범주형 변수, 질환과 교통 위반에 대한 연관성 여부를 검정을 통해 답하고자 한다. 가설은 다음과 같다.
# $H_0$ : 질환의 종류와 교통위반이 독립이다. (질환 종류와 교통 위반은 연관이 없다)
# $H_1$ : 질환의 종류와 교통위반이 독립이 아니다 (질환 종류와 교통 위반연관이 있다.)
# 1. 질환의 종류와 교통위반 간에 연관이 있다는 것을 확인하는 검정을 수행한다.
# 2. 위 검정결과 얻은 통계값을 입력하시오.(반올림하여 소수점 아래 넷째자리까지 계산)
# 3. 자유도를 산출하여 입력하시오.(정수)
# 4. 유의수준 0.05하에서 가설검정의 결과를 (채택/기각) 중 하나를 선택하여 입력하시오.
path3 = 'https://raw.githubusercontent.com/Soyoung-Yoon/data_02/main/'
from scipy.stats import chi2_contingency
df = pd.read_csv(path3 + "disease.csv", index_col=0)
# print(df.head(3))
statistic, pvalue, dof, expected = chi2_contingency(df)
result = "기각" if pvalue <= 0.05 else "채택"
# print(round(statistic, 4), dof, result) # 41.6187 3 기각

# 4. 살균제사용과 생존여부

# 대중적으로 인기 있는 살균제 리스터린은 영국의 내과의사로서 상부제 사용의 개척자인 조세프 리스터의 이름을 따서 명명한 것이다. 
# 그는 다년간 75명의 절단수술을 하였는데 40명에 대하여 살균제로 리스터린을 사용하고 35명에 대해 어떤 살균제도 사용하지 않았다. 
# 그 기록이 담긴 파일을 사용해 독립성 검정을 수행을 통해 답하고자 한다. 가설은 다음과 같다.
# $H_0$ : 살균제사용여부와 생존여부는 독립이다. (연관이 없다)
# $H_1$ : 살균제사용여부와 생존여부는 독립이 아니다 (연관이 있다.)
# 1. 살균제사용여부와 생존여부가 연관이 있다는 것을 확인하는 검정을 수행한다.
# 2. 위 검정의 결과로 얻은 통계값을 입력하시오.(반올림하여 소수점 아래 넷째자리까지 계산)
# 3. 통계값에 따른 p-value를 입력하시오. (반올림하여 소수점아래 넷째자리까지 표기)
# 4. 유의수준 0.05 하에서 가설 검정의 결과를 (채택/기각) 중 하나를 선택하여 입력하시오.
df = pd.read_csv(path3 + "listerin.csv")
# print(df.head(3))
from scipy.stats import chi2_contingency
c_table = pd.crosstab(df['살균제사용여부'], df['생존여부'])
# print(c_table)
statistic, pvalue, dof, _ = chi2_contingency(c_table)
result = "기각" if pvalue <= 0.05 else "채택"
# print(round(statistic, 4), round(pvalue, 4), result) # 7.0781 0.0078 기각

# 5. 멀미약 제품별 멀미 정도 비율
# 비행기 멀미로 인하여 고통 받는 여행객들이 있다고 한다. 멀미약 A 제품과 B 제품이 있는데 이 두 제품을 비교하기 위해 
# 90명의 여행객을 임의로 선택하여 45명은 A 제품, 나머지 45명은 B제품을 주어 나온 결과를 사용해 그 비율이 다른지 확인하고 싶다. 
# 가설은 다음과 같다.
# $H_0$ : 멀미약 제품별 멀미 정도 분포는 동일하다.
# $H_1$ : 멀미약 제품별 멀미 정도 분포는 동일하지 않다.
# 1. A제품과 B제품 간에 멀미정도 비율이 다른지 확인하는 검정을 수행한다
# 2. 위 검정의 결과로 얻은 통계값을 구해 입력하시오. (반올림하여 소수점 아래 둘째자리까지 표시)
# 3. 통계값에 따른 p-value를 구해 입력하시오. (반올림하여 소수점 아래 넷째자리까지 표시)
# 4. 자유도를 산출하여 입력하시오. (정수)
# 5. 유의수준 0.05 하에서 가설 검정의 결과를 (채택/기각) 중 하나를 선택하여 입력하시오.
from scipy.stats import chi2_contingency
df = pd.read_csv(path3 + 'drug.csv')
# print(df.head(3))
c_table = pd.crosstab(df['제품'], df['멀미정도'])
# print(c_table)
statistic, pvalue, dof, _ = chi2_contingency(c_table)
result = "기각" if pvalue <= 0.05 else "채택"
# print(round(statistic, 2), round(pvalue, 4), dof, result) # 5.58 0.1339 3 채택

# 6회-1번 (1) 흡연과 성별 비율
# A도시의 흡연여부 및 성별에 대한 조사 자료를 사용하여 남성과 여성간 흡연여부에 따른 비율이 동일한지 확인하는 
# 검정 수행을 통해 답하고자 한다. 가설은 다음과 같다.
# $H_0$ : 성별별 흡연자 비율은 동일하다
# $H_1$ : 성별별 흡연자 비율은 동일하지 않다
# 1.  A 도시의 남성 600명과 여성 550명이 있다. 남성들 중 흡연자 비율은 0.2이며 여성들 중 흡연자 비율은 0.26이다.
# 2.  남성과 여성 간에 흡연 여부에 따른 비율이 다른지 확인하는 검정을 수행한다.
# 3.  위 검정의 유의 수준 0.05에서 귀무가설에 대해 (기각/채택) 여부와 통계값, p-value값을 각각 출력하라. 
#     (반올림하여 소수점 아래 셋째자리까지 표시)
import pandas as pd
from scipy.stats import chi2_contingency
crs_tab = pd.DataFrame({"흡연": [600*0.2, 550*0.26], "비흡연": [600*0.8, 550*0.74]},
                       index=['남성','여성'], dtype="int")
crs_tab = crs_tab.T
# print(crs_tab)
statistic, pvalue, dof, expected_freq = chi2_contingency(crs_tab)
result = "기각" if pvalue <= 0.05 else "채택"
# print(round(statistic, 3), round(pvalue, 3), dof, result)
# 5.521 0.019 1 기각

# 6회-1번 (2) 흡연과 성별 비율
# A도시의 흡연여부 및 성별에 대한 조사 자료를 사용하여 남성과 여성간 흡연여부에 따른 비율이 
# 동일한지 확인하는 카이제곱검정을 수행을 통해 답하고자 한다. 가설은 다음과 같다.
# $H_0$ : 성별별 흡연자 비율은 동일하다.
# $H_1$ : 성별별 흡연자 비율은 동일하지 않다.
# 1. 남성과 여성 간에 흡연 여부에 따른 비율이 다른지 확인하는 검정을 수행한다.
# 2. 유의 수준 0.05에서 귀무가설에 대해 (기각/채택) 여부와 통계값, p-value값을 각각 출력하라. 
#    (반올림하여 소수점 아래 셋째자리까지 표시)
import pandas as pd
from scipy.stats import chi2_contingency
df = pd.read_csv(path1 + "gender_smoke_data.csv")
# print(df.head(3))
c_table = pd.crosstab(df['Smoke'], df['Gender'])
# print(c_table)
statistic, pvalue, dof, expected_freq = chi2_contingency(c_table)
result = "기각" if pvalue <= 0.05 else "채택"
# print(result, round(statistic, 3), round(pvalue, 3))
# 기각 5.521 0.019