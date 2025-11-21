
# 작업형 제1유형
import pandas as pd
pd.set_option('display.width', 120)
path1 = "https://raw.githubusercontent.com/Soyoung-Yoon/bigdata/main/"
path2 = "https://raw.githubusercontent.com/Soyoung-Yoon/data_01/main/"
path3 = "https://raw.githubusercontent.com/Soyoung-Yoon/data_02/main/"

# 3회-1번) 캘리포니어 집값  
# 주택 데이터에서 housing_median_age의 분포 파악을 위해 사분위수 범위(IQR)를 확인하려한다.
# 다음 절차에 따라 문제를 해결하시오.
df = pd.read_csv(path1 + "housing03.csv")
# print(df.isna().sum(), df.info(), sep="\n") # 20640
# 1) 결측치를 포함하는 모든 행을 제거한 후, 처음부터 순서대로 70%를 추출한다.
df = df.dropna()
df = df.loc[:int(len(df) * 0.7), :]
# print(df.shape) # (14169, 10)
# 2) 'housing_median_age' 컬럼의 사분위 범위(IQR, Interquartile Range)의 값을 구한다.
#    > IQR = 3사분위수 - 1사분위수
Q1, Q3 = df['housing_median_age'].quantile([0.25, 0.75])
IQR = Q3 - Q1
# 3) 사분위 범위를 반올림하여 정수로 출력한다.
# print(round(IQR)) # 19

# 3회 2번) 연도별 나라별 유병률 데이터
# 연도별 국가별 유병률 데이터를 사용하여, 특정 연도에 건강 지표가 상대적으로 높은 국가를 파악하고자 한다.
# 다음 절차에 따라 문제를 해결하시오.
df = pd.read_csv(path1 + "worlddata.csv")
# print(df.head())
# 1) 2002년 데이터에 대해, 국가 전체 유병률 평균값보다 큰 유병률을 가진 국가의 개수를 구하시오.
df = df[df['year']==2002].set_index('year').T
result = len(df[df[2002] > df[2002].mean()])
# 2) 결과는 정수형으로 제출한다.
# print(int(result)) # 90

# 3회-3번) 타이타닉 데이터 결측치 비율 분석
# 타이타닉 데이터셋의 일부 열에는 결측치가 존재한다. 해당 열의 결측치를 비교한다.
# 다음 절차에 따라 문제를 해결하시오.
df = pd.read_csv(path1 + "titanic_train03.csv")
# print(df.head(3))
# 1) 타이타닉 데이터셋에서 각 열의 결측치 비율을 계산한 후, 결측치 비율이 가장 높은 변수명을 구하시오.
s = df.isna().sum().to_frame()
total_null = df.isna().sum().sum()
result = (s[0] / total_null).idxmax()
# 2) 결과는 변수명(문자형)으로 제출한다.
# print(str(result)) # Age

# 11-4) 범죄 유형의 발생 빈도 분석
# 특정 범죄 유형이 평균 이상으로 발생한 연도 수를 파악하는 것은 추세 분석에 유용하다.
# 다음 절차에 따라 문제를 해결하시오.
df = pd.read_csv(path2 + "crime_data02.csv")
# print(df.head(3))
# 1) '구분'이 '발생건수'인 데이터에서 '범죄유형5'의 연도별 발생건수를 구한다.
s = df[df['구분']=='발생건수'].drop(columns=['구분']).set_index('연도')['범죄유형5']
# print(s)
# 2) 1)의 결과에서 '범죄유형5'의 전체 평균 발생건수보다 큰 연도의 비율을 구한다.
# - 비율 계산 방법
#   > 비율 = (평균보다 큰 연도 개수) ÷ (전체 연도 개수)
ratio = len(s[s.values > s.values.mean()]) / len(s)
# 3) 최종 결과는 반올림하여 소수점 3자리까지 실수형으로 출력한다.
# print(round(ratio, 3)) # 0.429

# 11-5) 발생 비율이 높은 범죄유형
# 년별, 범죄유형별 발생건수 데이터를 사용해 자주 많이 발생한 유형을 분석한다.
# 다음 절차에 따라 문제를 해결하시오.
df = pd.read_csv(path2 + "crime_data02.csv")
# print(df.head(3))
# 1) 각 범죄유형별 연도별 발생건수를 구한다.
df = df[df['구분']=="발생건수"].drop(columns=['구분']).set_index("연도").T
# 2) 1)번의 결과를 사용해 각 범죄유형에 대해, 전체 연도 동안의 평균 발생건수를 계산한다.
s = df.loc[:, 2014:2020].mean(axis=1)
# print(s)
# 3) 각 범죄유형에 대해 평균보다 큰 발생건수를 기록한 연도의 비율을 구한다.

# 4) 3)에서 구한 비율이 0.5 이상인 범죄유형의 개수를 구해 정수로 출력한다.

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
cols_obj = X_all.select_dtypes(include='object').columns
for col in cols_obj:
    X_all[col] = LabelEncoder().fit_transform(X_all[col])
X_all = pd.get_dummies(X_all, drop_first=True, dtype='int32')
# print(X_all.head(3))

# 데이터 재분할
X = X_all.iloc[:len(X), :]
X_submission = X_all.iloc[len(X):, :]
# print(X.shape, X_submission.shape) # (3300, 13) (1221, 13)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=300)
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (2310, 13) (990, 13) (2310,) (990,)

# 파이프라인 모델사전
models = {
    "Logistic": Pipeline([
        ("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=100, tol=0.05, random_state=300))
    ]),
    "DecisionTree": Pipeline([
        ("model", DecisionTreeClassifier(max_depth=5, random_state=300))
    ]),
    "RandomForest": Pipeline([
        ("model", RandomForestClassifier(n_estimators=500, max_depth=5, random_state=300))
    ]),
    "AdaBoost": Pipeline([
        ("model", AdaBoostClassifier(n_estimators=500, random_state=300))
    ]),
    "GradientBoosting": Pipeline([
        ("model", GradientBoostingClassifier(random_state=300))
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

# 모델별 성능평가
results = []
for name, model in models.items():
    model, AUC_train, AUC_test = get_scores(model, x_train, x_test, y_train, y_test)
    results.append({
        "Model": name, "AUC_train": f"{AUC_train:.4f}", "AUC_test": f"{AUC_test:.4f}"
    })
res = pd.DataFrame(results).sort_values("AUC_test", ascending=False).reset_index(drop=True)
# print(res)

# 모델적용, 예측산출
model = models[res.loc[0, "Model"]]
y_pred = model.predict_proba(X_submission)[:, 1]

# 제출파일 생성
# pd.DataFrame({'pred': y_pred}).to_csv("result_day11.csv", index=False)

# 결과확인
# temp = pd.read_csv("result_day11.csv")
# print(temp['pred'].describe())

# 작업형 제3유형
# 펭귄 품종별 체중 차이
# penguins 데이터에서, 품종(species)별 체중(body_mass_g)의 평균에 차이가 있는지 
# 확인하는 검정을 수행해 답하고자 한다. 가설은 다음과 같다.
# $H_0$ : 세 가지 품종의 body_mass_g 평균은 동일하다.
# $H_1$ : 적어도 한 품종의 body_mass_g 평균이 다르다.
import pandas as pd
from statsmodels.api import OLS
from statsmodels.stats.anova import anova_lm


