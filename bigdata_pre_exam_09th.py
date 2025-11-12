
# 기출문제 제9회 2024-11-30
# 작업형 제1유형
import pandas as pd
path = "https://raw.githubusercontent.com/YoungjinBD/data/main/exam/"

# 1. 데이터에서 (연도, 성별, 지역코드)별 총 대출액의 합계를 구하시오. 이후, 각 (연도, 지역코드)별로 남성과 여성의 총 대출액 차이의 절댓값을 계산하고,
# 성별 간 총 대출액 차이가 가장 큰 지역코드를 구하시오.(총 대출액 = 금액1 + 금액2)
df1 = pd.read_csv(path + "9_1_1.csv")
# print(df1.head(3))
df1['총대출액'] = df1['금액1'] + df1['금액2']
group = df1.groupby(['year','gender','지역코드'], as_index=False)['총대출액'].sum()
gM = group[group['gender']==0].groupby(['year','지역코드'])['총대출액'].sum().reset_index(name='총대출액(남성)').sort_values(['year','지역코드'], ascending=True)
gF = group[group['gender']==1].groupby(['year','지역코드'])['총대출액'].sum().reset_index(name='총대출액(여성)').sort_values(['year','지역코드'], ascending=True)
result = pd.merge(gM, gF, "left", on=['year','지역코드']).fillna(0)
result['총대출액차이'] = abs(result['총대출액(남성)'] - result['총대출액(여성)'])
# print(result[result['총대출액차이'] == result['총대출액차이'].max()]['지역코드'].values[0]) # 143

# 2. 각 연도별 최대 검거율을 가진 범죄유형을 찾아서 해당 연도 및 유형의 검거건수들의 총합을 구하시오. (검거율 = 검거건수 / 발생건수)
df2 = pd.read_csv(path + "9_1_2.csv")
# print(df2.head(3))
# 검거건수, 발생건수 분리
df21 = df2[df2['구분']=='검거건수'].set_index('연도').drop(columns='구분')
df22 = df2[df2['구분']=='발생건수'].set_index('연도').drop(columns='구분')
# 검거율 산출
ratio = df21 / df22
# print(ratio)
# 최대 검거율 추출
max_ratio = ratio.max(axis=1)
# 최대검거율인 범죄유형 추출
def is_max_col(col):
    return col == max_ratio
mask = ratio.apply(is_max_col, axis=0)
# print(mask.iloc[0, :])
# print(df21[mask].fillna(0).sum().sum()) # 36977

# 3. 제시된 문제를 순서대로 풀고 해달을 제시하시오.
# ① 평균만족도: 결측치는 평균만족도 컬럼의 전체 평균으로 채우시오.
df3 = pd.read_csv(path + "9_1_3.csv")
# print(df3.head())
# print(df3.isna().sum().to_frame().T)
df3['평균만족도'] = df3['평균만족도'].fillna(df3['평균만족도'].mean())
# print(df3.isna().sum().to_frame().T)

# ② 근속연수: 결측치는 각 부서와 등급별 평균 근속연수로 채우시오.(평균값의 소수점은 버림 처리)
import numpy as np
s = np.floor(df3.groupby(['부서','등급'])['근속연수'].transform('mean'))
df3['근속연수'] = df3['근속연수'].fillna(s)
# print(df3.isna().sum().to_frame().T)

# ③ A: 부서가 'HR'이고 등급이 'A'인 사람들의 평균 근속연수를 계산하시오.
A = df3[(df3['부서']=='HR') & (df3['등급']=='A')]['근속연수'].mean()

# ④ B: 부서가 'Sales'이고 등급이 'B'인 사람들의 평균 교육참가횟수를 계산하시오.
B = df3[(df3['부서']=='Sales') & (df3['등급']=='B')]['교육참가횟수'].mean()

# ⑤ A와 B를 더한 값을 구하시오.
# print(A + B) # 25.225

# 작업형 제2유형
# 제공된 학습용 데이터(9_2_train.csv)는 지역의 특성과 해당 지역의 농업 유형 정보를 포함하고 있다.
# 학습용 데이터를 활용하여 지역의 농업 유형(라벨)을 예측하는 다중분류 모델을 개발하고, 가장 우수한 모델을
# 평가용 데이터(9_2_test.csv)에 적용하여 예측 결과를 제출하시오. 모델 성능 지표는 Macro, F1 Score
# Data Description
# ID: 고유식별자, 지역: 관측지역, 등급: 농업등급, 농업면적: 해당지역의 농업면적, 연도: 데이터가 수집된 연도, 라벨: 농업유형
# 제출형식
# 파일명: result.csv, 제출컬럼: ID, pred(예측된 농업 유형, 정수형, 0, 1, 2 중 하나), 평가용 데이터와 예측 결과 데이터의 개수는 동일해야 함.

# 라이브러리
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score

# 데이터 확인
train = pd.read_csv(path + "9_2_train.csv")
test = pd.read_csv(path + "9_2_test.csv")
# print(train.head(3), train.shape, sep="\n") # (1680, 6)
# print(test.head(3), test.shape, sep="\n")   #  (720, 6)
# print(train.isna().sum().to_frame().T)      # 결측치 없음
# print(test.isna().sum().to_frame().T)

# 데이터 전처리
X = train.drop(columns=['ID','라벨'])
Y = train['라벨']
X_submission = test.drop(columns=['ID','라벨'])
X_all = pd.concat([X, X_submission])
X_all['지역'] = LabelEncoder().fit_transform(X_all['지역'])
X_all['등급'] = LabelEncoder().fit_transform(X_all['등급'])
# print(X_all.head())
# X_all = pd.get_dummies(X_all, drop_first=True, dtype='int32')
# print(X_all.head())

# 데이터 재분할
X = X_all.iloc[:len(X), :]
X_submission = X_all.iloc[len(X):, :]
# print(X.shape, X_submission.shape) # (1680, 4) (720, 4)
temp = train_test_split(X, Y, test_size=0.2, random_state=42)
x_train, x_test, y_train, y_test = temp
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (1176, 4) (504, 4) (1176,) (504,)

# 파이프라인 모델사전
models = {
    "Logistic": Pipeline([
        ("scaler", MinMaxScaler()), ("model", LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "DecisionTree": Pipeline([
        ("model", DecisionTreeClassifier(max_depth=3, random_state=42))
    ]),
    "RandomForest": Pipeline([
        ("model", RandomForestClassifier(max_depth=3, random_state=42))
    ]),
    "AdaBoost": Pipeline([
        ("model", AdaBoostClassifier(random_state=42))
    ]),
    "Gradient": Pipeline([
        ("model", GradientBoostingClassifier(random_state=42))
    ])
}

# 성능평가 함수
def get_scores(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    y_pred1 = model.predict(x_train)
    y_pred2 = model.predict(x_test)
    F1_train = f1_score(y_train, y_pred1, average="macro")
    F1_test = f1_score(y_test, y_pred2, average="macro") 
    return model, F1_train, F1_test

# 모델별 성능평가
results = []
for name, model in models.items():
    model, F1_train, F1_test = get_scores(model, x_train, x_test, y_train, y_test)
    results.append({
        "Model": name, "F1_train": f"{F1_train:.4f}", "F1_test": f"{F1_test:.4f}"
    })
res = pd.DataFrame(results).sort_values("F1_test", ascending=False).reset_index(drop=True)
# print(res)

# 모델적합
model = models[res.loc[0, "Model"]]
y_pred = model.predict(X_submission)

# 파일생성
# pd.DataFrame({'ID': test['ID'],'pred': y_pred}).to_csv("result_9th.csv", index=False)

# 결과확인
# temp = pd.read_csv("result_9th.csv")
# print(Y[len(X_submission):].value_counts(normalize=True))
# print("=" * 35)
# print(temp['pred'].value_counts(normalize=True))

# 작업형 제3유형
# 문제1
# 한 제조 회사에서 생산성을 높이고자 직원들의 주요 생산성 요인을 분석하기로 결정하였다. 이를 위해
# 200명의 직원 데이터를 수집했으며, 직원들의 근무기간, 특성 정보, 그리고 개인적인 속성을 조사하였다.
# Data Description
# id: 데이터의 고유 식별자, tenure: 사용기간, f2: 고객의 두 번째 특성, f3: 고객의 세 번째 특성, 
# f4: 고객의 네 번째 특성, f5: 고개의 다섯 번째 특성, design: 생산성 점수
df3 = pd.read_csv(path + "9_3_1.csv")
# print(df3.head(3))
# ① design을 예측하는 다중회귀 분석을 시행한 후, 유의하지 않은 설명변수 개수를 구하시오.(단, 불필요한 컬럼은 제외하며, 모델의 절편항을 포함)
# 데이터 분리조건
# 훈련데이터: id1 ~ id140, 테스트데이터: id141 ~ id200
from statsmodels.api import OLS, add_constant
train = df3[:140]
test = df3[140:]
train = add_constant(train)
formula = "design ~ tenure + f2 + f3 + f4 + f5"
# model = OLS.from_formula(formula, train).fit()
model = OLS.from_formula(formula, train).fit()
# print(model.summary())
# print(sum(model.pvalues[1:] >= 0.05)) # 2

# ② 훈련데이터(학습용 데이터)의 예측값과 실제값의 피어슨 상관계수를 구하시오.(소수점 셋째 자리에서 반올림)
y_pred_train = pd.Series(model.predict(train))
y_true_train = pd.Series(train['design'])
# print(round(y_pred_train.corr(y_true_train), 4)) # 0.9148

# ③ 적합한 모델을 활용하여 테스트데이터에서의 RMSE를 구하시오.
from sklearn.metrics import mean_squared_error as MSE
y_pred_test = model.predict(test)
y_true_test = test['design']
# print(MSE(y_true_test, y_pred_test) ** 0.5) # 4.396152958589427

# 문제2
# 한 통신 회사에서는 고객 이탈을 줄이고자 주요 요인들을 분석하기로 결정하였다. 이를 위해 500명의 고객 데이터를 수집했으며, 
# 고객의 서비스 이용 및 가입 정보, 그리고 일부 개인적인 속성을 조사하였다.
# Data Description
# col1: 고객의 첫 번째 특성, col2: 고객의 두 번째 특성, Phone_Service: 폰 서비스 가입 여부, Tech_Insurance: 기술 보험 가입 여부, churn: 이탈 여부
# ① 고객 이탈을 예측하는 로지스틱 회귀를 시행한 후 col1칼럼의 p-value를 구하시오.(소수점 넷째 자리에서 반올림)
df4 = pd.read_csv(path + "9_3_2.csv")
# print(df4.head())
from statsmodels.api import GLM, families
formula = 'churn ~ col1 + col2 + Phone_Service + Tech_Insurance'
model = GLM.from_formula(formula, df4, family=families.Binomial()).fit()
# print(model.summary())
# print(round(model.pvalues['col1'], 4)) # 0.000

# ② 폰 서비스를 받지 않는 고객 대비 받은 고객의 이탈 확률 오즈비를 구하시오.(소수점 넷째 자리에서 반올림)
odds_ratio = np.exp(model.params['Phone_Service'])
# print(round(odds_ratio, 4)) # 1.8671

# ③ 이탈할 확률이 0.3 이상인 고객 수를 구하시오.
pred_proba = model.predict(df4)
# print(sum(pred_proba >= 0.3)) # 450

