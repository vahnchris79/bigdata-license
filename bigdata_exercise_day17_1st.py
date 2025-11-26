# 공통: 데이터 경로
import pandas as pd
path1 = "https://raw.githubusercontent.com/Soyoung-Yoon/bigdata/main/"
path2 = "https://raw.githubusercontent.com/Soyoung-Yoon/data_01/main/"

# 9회-1번) 대출액 차이 분석
# 은행과 자본 대출 데이터를 활용하여 각 지역에 대해 연도 및 성별별 총대출액을 집계하고, 
# 지역 간 대출 격차를 분석하려 한다.
# 다음 절차에 따라 문제를 해결하시오.
# 각 지역(Region_Code)에 대해 연도(Year) 및 성별(Gender)별로 총대출액의 합계를 구하시오.
# 총대출액은 은행대출(Bank_Loan) + 캐피탈대출(Capital_Loan)으로 계산한다.
# 총대출액의 연도 및 성별별 합계액 차이의 절댓값이 가장 큰 지역을 구해,
# 지역(Region_Code)을 정수형으로 출력합니다.
df = pd.read_csv(path1 + "loan_data.csv")
# print(df.head(3))
df['Total_Loan'] = df['Bank_Loan'] + df['Capital_Loan']
s1 = df.groupby(['Region_Code','Year'])['Total_Loan'].sum()
s2 = df.groupby(['Region_Code','Gender'])['Total_Loan'].sum()
# print(int(abs(s1-s2).idxmax()[0])) # 4146510700

# 9회-2번) 연도별 최고 검거율 범죄 유형 분석
# 범죄 발생 및 검거 데이터를 기반으로 범죄유형별 검거율을 계산하고,
# 각 연도별로 검거율이 가장 높은 범죄유형의 검거 실적을 분석한다.
# 다음 절차에 따라 문제를 해결하시오.
# 검거율은 다음과 같이 계산한다.
# 검거율 = 검거건수 / 발생건수
# 범죄유형별로 연도별 검거율을 계산하시오.
# 각 연도별로 검거율이 가장 높은 범죄유형을 찾으시오.
# 해당 범죄유형의 검거건수를 구하고, 그 값들을 모두 합한 값을 정수형으로 출력하시오.
df = pd.read_csv(path1 + "crime_data.csv")
# print(df.head(3))
dfA = df[df['구분']=='검거건수'].drop(columns='구분').set_index('연도')
dfB = df[df['구분']=='발생건수'].drop(columns='구분').set_index('연도')
dfC = dfA / dfB
s = dfC.idxmax(axis=1)
# print(s)
temp = [dfA.loc[year, crime] for year, crime in zip(s.index, s.values)]
# print(sum(temp)) # 7799

# 9회-3번) 근속연수 및 교육참가 분석
# 사원 데이터의 결측값을 조건에 따라 적절히 처리하고, 부서와 조건별 평균을 활용하여 특정 계산을 수행한다.
# 다음 절차에 따라 문제를 해결하시오.
# 평균만족도에 결측치가 있는 경우, 전체 평균만족도의 평균값으로 채우시오.
# 근속연수에 결측치가 있는 경우, 같은 부서와 같은 성과등급을 가진 사원들의 근속연수 평균을 정수로 변환하여 채우시오.
# 변수 A는 부서가 'Sales'이고 성과등급이 'C'인 사원들의 평균 근속연수로 정의하시오.
# 변수 B는 부서가 'Operations'이고 평균만족도가 2.5 이상인 사원들의 평균 교육참가횟수로 정의하시오.
# 최종적으로 A + B의 값을 정수로 출력하시오.
df = pd.read_csv(path1 + "hr_data.csv")
# print(df.head(3))
# print(df.isna().sum())
df['평균만족도'] = df['평균만족도'].fillna(df['평균만족도'].mean())
cond = df.groupby(['부서','성과등급'])['근속연수'].transform('mean')
df['근속연수'] = df['근속연수'].fillna(cond)
# print(df.isna().sum())
A = df[(df['부서']=='Sales') & (df['성과등급']=='C')]['근속연수'].mean()
B = df[(df['부서']=='Operations') & (df['평균만족도'] >= 2.5)]['교육참가횟수'].mean()
# print(int(A + B)) # 20

# 17-4) 연도별 최고 고용비중 산업 분석
# 산업별 고용 데이터를 활용하여 각 연도별 고용비중이 가장 높은 산업을 분석한다.
# 다음 절차에 따라 문제를 해결하시오.
# 1) 산업(Industry)별 연도(Year)별 고용 인원을 기반으로, 전년도 대비 고용 증가율을 다음과 같이 계산하시오.
#    고용 증가율 = (이번년도 고용 - 전년도 고용) / 전년도 고용
# 2) 각 연도별로 고용 증가율이 가장 높은 산업을 찾으시오.
# 3) 2)에서 찾은 산업의 고용 증가량을 구하시오.
#    고용 증가량 = (이번년도 고용 - 전년도 고용)
# 4) 연도별로 구한, 고용 증가량을 합산하여 정수형으로 출력하시오.
#    단, 전년도 데이터가 존재하는 연도(2011년 이상)에 대해서만 계산한다.
df = pd.read_csv(path1 + "employment_data.csv")
# print(df.head(3))
# 이번년도, 전년도 고용 -> 년도별/산업별
thisYear = df.pivot(index='Year', columns='Industry', values='Employment')
# print(thisYear)
lastYear = thisYear.shift(1)
# print(lastYear)
# 고용 증가율 = (이번년도 고용 - 전년도 고용) / 전년도 고용
increased_value = thisYear.diff(1)
increased_rate = increased_value / lastYear
# print(increased_rate)
s = increased_rate.dropna().idxmax(axis=1)
# print(s)
temp = [increased_value.loc[year, industry] for year, industry in zip(s.index, s.values)]
# print(int(sum(temp)))


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
X_all['job'] = LabelEncoder().fit_transform(X_all['job'])
X_all['marital'] = LabelEncoder().fit_transform(X_all['marital'])
X_all['education'] = LabelEncoder().fit_transform(X_all['education'])
X_all['poutcome'] = LabelEncoder().fit_transform(X_all['poutcome'])
X_all = pd.get_dummies(X_all, drop_first=True, dtype='int32')
# print(X_all.head())

# 데이터 재분할
X = X_all.iloc[:len(X), :]
X_submission = X_all.iloc[len(X):, :]
# print(X.shape, X_submission.shape) # (3300, 14) (1221, 14)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=300)
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (2640, 14) (660, 14) (2640,) (660,)

# 모델사전
models = {
    "Logistic": Pipeline([
        ("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=300, tol=0.03, random_state=300))
    ]),
    "DecisionTree": Pipeline([
        ("model", DecisionTreeClassifier(max_depth=15, random_state=300))
    ]),
    "RandomForest": Pipeline([
        ("model", RandomForestClassifier(n_estimators=300, max_depth=15, random_state=300))
    ]),
    "AdaBoost": Pipeline([
        ("model", AdaBoostClassifier(n_estimators=300, random_state=300))
    ]),
    "Gradient": Pipeline([
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
# pd.DataFrame({'pred': y_pred}).to_csv("result_day17.csv", index=False)

# 결과확인
# temp = pd.read_csv("result_day17.csv")
# print(Y[:len(test)].describe())
# print("=" * 35)
# print(temp['pred'].describe())

# 작업형 제3유형
# 다음과 같은 로지스틱회귀 모형을 사용한 분류모델을 만들고 결과를 확인합니다.
# wine01.csv 데이터를 사용합니다.
# 모델 생성시 상수항(=절편)을 포함하도록 하며, 규제는 사용하지 않습니다.
# 종속변수 : wine_variety
# 독립변수 : alcohol, color_intensity, proline, flavanoids, malic_acid
df = pd.read_csv(path2 + "wine01.csv")
# print(df.head(3))
# 1-2) GLM.from_formula() 를 사용해 분석하려고 합니다. formula를 작성하고, 로지스틱 회귀모형을 생성합니다.
from statsmodels.api import GLM, families
formula = "wine_variety ~ alcohol + color_intensity + proline + flavanoids + malic_acid"
model = GLM.from_formula(formula, df, family=families.Binomial()).fit()
# print(model.summary())

#1-3) 모델의 로그-우도를 반올림하여 소수점 아래 3자리까지 출력합니다.
# print(round(model.llf, 3)) # -3.206

#1-4) 잔차이탈도(Deviance)를 반올림하여 소수점 아래 3자리까지 출력합니다.
# print(round(model.deviance, 3)) # 6.411

#1-5) 'proline'을 독립변수로 하였을 때의 오즈비(Odds Ratio)는?
# 반올림하여 소수점 아래 3자리까지 출력합니다.
import numpy as np
odds_ratio = np.exp(model.params['proline'])
# print(round(odds_ratio, 3)) # 0.975

#1-6) 'proline'가 3증가하면 오즈는 몇 % 감소 또는 증가하는가?
#(단, 감소율 또는 증가율은 반올림하여 소수점아래 2자리까지 표시합니다.)
# 감소율 = (1 - odds_ratio) * 100
# 증가율 = (odds_ratio - 1) * 100
odds_ratio = np.exp(model.params['proline'] * 3)
# print(round((1 - odds_ratio) * 100, 2)) # 7.45

# 1-7) 아래의 sample을 사용하여 P(Y=1)에 대한 확률을 구하고,
# 반올림하여 소수점 아래 3자리까지 출력하세요.
# sample => alcohol : 13.5, color_intensity: 5.0, proline : 450, flavanoids : 2.8, malic_acid : 1.8
sample = pd.DataFrame({'alcohol': [13.5], 'color_intensity': [5], 'proline': [450],
                       'flavanoids': [2.8], 'malic_acid': [1.8]})
y_1 = model.predict(sample)[0]
# print(round(y_1, 3)) # 0.659

# 1-8) 위 샘플에 대한 odds 를 구하고, 반올림하여 소수점 아래 4자리까지 출력하세요.
odds = y_1 / (1 - y_1)
# print(round(odds, 4)) # 1.9321

#1-9) 유의수준 5%하에서, 유의성이 낮은 변수의 개수는 몇 개인가?
# print((model.pvalues[1:] > 0.05).sum()) # 5

#1-10) 아래 샘플에 대한 P(Y=1)에 대한 95% 신뢰구간의 상한은?
# 반올림하여 소수점 아래 4자리까지 출력합니다.
# sample => alcohol : 13.5, color_intensity: 5.0, proline : 850, flavanoids : 2.8, malic_acid : 1.8
sample = pd.DataFrame({'alcohol': [13.5], 'color_intensity': [5], 'proline': [850],
                       'flavanoids': [2.8], 'malic_acid': [1.8]})
result = model.get_prediction(sample)
# print(round(result.conf_int(alpha=0.05)[0][1], 4)) # 0.9904

#1-11) 정확도를 구해 반올림하여 소수점 아래 3자리까지 출력합니다.
from sklearn.metrics import accuracy_score
y_true = df['wine_variety']
y_pred = model.predict(df).round()
# print(y_pred)
result = accuracy_score(y_true, y_pred)
# print(round(result, 3)) # 0.985

#1-12) f1_score를 구해 반올림하여 소수점 아래 3자리까지 출력합니다.
from sklearn.metrics import f1_score
y_true = df['wine_variety']
y_pred = model.predict(df).round()
result = f1_score(y_true, y_pred)
# print(round(result, 3)) # 0.986
