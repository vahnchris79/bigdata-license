
# 작업형 제1유형
import pandas as pd
pd.set_option('display.width', 120)
path = "https://raw.githubusercontent.com/Soyoung-Yoon/bigdata/main/"

# 5-1) tips 데이터셋을 이용한다.
# - tip / total_bill 비율이 0.15를 초과하고, size가 3 이상인 경우만 필터링한다.
# - 이 조건을 만족하는 손님의 총 인원 수(size의 합)를 구하시오.
df = pd.read_csv(path + "tips.csv")
# print(df.head(3))
cond1 = (df['tip'] / df['total_bill']) > 0.15
cond2 = (df['size'] >= 3)
# print(df[cond1 & cond2]['size'].sum()) # 143

# 5-2) 데이터에서 month가 'Jan'부터 'Jun'까지인 경우를 상반기,
# 'Jul'부터 'Dec'까지인 경우를 하반기로 나누어 분석한다.
# - 상반기와 하반기 각각에 대해, 연도(year)별 총 승객 수(passengers)를 계산한다.
# - 각 데이터에서, 연도별 승객 수 증가량을 계산하라.
# 단, 첫 번째 연도는 증가량을 0으로 둔다.
# - 상반기와 하반기 각각의 증가량 중, 가장 많이 증가한 연도를 구한다.
# - flights 전체 데이터에서, 최대 승객 수를 구한다.
# - (상반기 최대 증가 연도) + (하반기 최대 증가 연도) + (최대 승객 수)의 결과를 정수로 출력하시오.
df = pd.read_csv(path + "flights.csv")
# print(df.head(3))
# print(df['month'].unique())
df1 = df[df['month'].str.contains('Jan|Feb|Mar|Apr|May|Jun')].copy()
df2 = df[df['month'].str.contains('Jul|Aug|Sep|Oct|Nov|Dec')].copy()
# print(df1['month'].unique())
# print(df2['month'].unique())
group1 = df1.groupby('year')['passengers'].sum()
diff1 = group1.diff().fillna(0)
group2 = df2.groupby('year')['passengers'].sum()
diff2 = group2.diff().fillna(0)
max_year1 = diff1.idxmax()
max_year2 = diff2.idxmax()
max_passengers = df['passengers'].max()
# print(int(max_year1 + max_year2 + max_passengers)) # 4541

# 5-3) tips 데이터셋에서 다음 조건을 만족하는 손님의 total_bill 평균을 반올림하여 소수점 아래 3자리까지 구하시오.
# - total_bill이 자신이 속한 요일(day)의 평균 이상인 손님만 선택한다.
# - 위 조건을 만족하는 손님 중 sex가 Female이고, tip이 중앙값 이상인 데이터만 선택한다.
df = pd.read_csv(path + "tips.csv")
# total_bill이 자신이 속한 요일(day)의 평균 이상인 손님만 선택
s = df.groupby('day')['total_bill'].transform('mean')
df = df[df['total_bill'] >= s]
# 조건을 만족하는 손님 중 sex가 Female이고, tip이 중앙값 이상인 데이터만 선택
cond = (df['sex']=='Female') & (df['tip'] >= df['tip'].median())
# print(df[cond]['total_bill'].mean().round(3)) # 29.335

# 5-4) titanic_dataq.csv를 사용하여 다음 조건을 만족하는 승객 수를 정수로 구하시오.
# - Cabin 컬럼의 결측치를 Cabin 컬럼의 첫 글자 중 최빈값으로 채우기 한 뒤, Cabin 컬럼의 값을 첫 글자로 변경한다. 
#   (예) C12 => C
# - Cabin 컬럼의 최빈값에 해당하는 데이터만 추출하여, 나이(Age) 컬럼의 결측치를 Embarked와 Pclass 조합별 평균 나이로 채운다.
# - Age 컬럼의 결측치를 채운 후, Age가 Embarked와 Pclass 조합별 중앙값 이상이고, Fare가 중앙값보다 높은 승객만 남긴다.
df = pd.read_csv(path + "titanic_dataq.csv")
# print(df.head(3))
# Cabin 컬럼의 결측치를 Cabin 컬럼의 첫 글자 중 최빈값으로 채우기
first_word = df['Cabin'].str[0].mode()[0]
df['Cabin'] = df['Cabin'].fillna(first_word)
df['Cabin'] = df['Cabin'].str[0]
# Cabin 컬럼의 최빈값에 해당하는 데이터만 추출하여, 나이(Age) 컬럼의 결측치를 Embarked와 Pclass 조합별 평균 나이로 채운다
df = df[df['Cabin']==df['Cabin'].mode()[0]].copy()
s1 = df.groupby(['Embarked','Pclass'])['Age'].transform('mean')
df['Age'] = df['Age'].fillna(s1)
# Age가 Embarked와 Pclass 조합별 중앙값 이상이고, Fare가 중앙값보다 높은 승객만 남긴다.
s2 = df.groupby(['Embarked','Pclass'])['Age'].transform('median')
cond = (df['Age'] >= s2) & (df['Fare'] > df['Fare'].median())
# print(len(df[cond])) # 194

# 5-5) California housing 데이터셋을 활용하여 다음 조건을 모두 만족하는 계산을 수행하시오.
# - Latitude 값을 기준으로 데이터를 5개 구간으로 분위수 분할하여 ocean_proximity 컬럼을 생성한다.
# - 다음의 필터링을 수행하기 전, 후의 데이터에 대해 ocean_proximity별 HouseAge의 중앙값(median)을 계산한다.
# - 필터링 전 데이터에 대한 계산 결과를 S1, 필터링 후 데이터에 대한 계산 결과를 S2로 저장한다.
# - HouseAge가 자신이 속한 ocean_proximity 그룹의 중앙값보다 큰 경우만 필터링한다.
# - 최종적으로 S2에서 S1의 값을 뺀(S2 - S1) 값 중 최소값을 정수로 구하시오.
df = pd.read_csv(path + "california.csv")
# print(df.head(3))
# Latitude 값을 기준으로 데이터를 5개 구간으로 분위수 분할하여 ocean_proximity 컬럼을 생성한다.
df['ocean_proximity'] = pd.qcut(df['Latitude'], q=5, labels=False)
# print(df['ocean_proximity'].unique())
S1 = df.groupby('ocean_proximity')['HouseAge'].median()
# df2 = df[df['HouseAge'] > df['HouseAge'].median()]
df2 = df[df['HouseAge'] > df.groupby('ocean_proximity')['HouseAge'].transform('median')]
S2 = df2.groupby('ocean_proximity')['HouseAge'].median()
# print(int((S2 - S1).min())) # 7

# 작업형 제2유형
# **Bank Marketing 데이터셋**
# 고객의 인구통계적 정보 및 이전 마케팅 이력 데이터를 바탕으로 "이 사람이 정기예금 상품에 가입할 것인가?" 를 예측하는 것이 목적입니다.
# 제공된 학습용 데이터(bank_train.csv)를 이용하여 정기예금 상품에 가입 여부를 예측하는 모델을 개발하고, 
# 개발한 모델에 기반하여 평가용 데이터(bank_test.csv)에 적용하여 얻은 정기예금 상품에 가입 여부 예측 확률을 
# 아래 [제출형식]에 따라 csv 파일로 생성하여 제출하시오.
# - 예측 결과는 ROC-AUC 평가지표에 따라 평가함
# - 성능이 우수한 예측 모델을 구축하기 위해서는 데이터 정제, Feature Engineering, 
#   하이퍼 파라미터(hyper parameter) 최적화, 모델 비교 등이 필요할 수 있음. 
# 다만, 과적합에 유의하여야 함
# [[제출 형식]]
# - 가. CSV 파일명: result.csv(파일명에 디렉토리/폴더 지정불가
# - 나. 예측 성별 칼럼명 : pred
# - 다. 제출 칼럼 개수 : pred 칼럼 1개
# - 라. 평가용 데이터 개수와 예측 결과 데이터 개수 일치 : 1,221개
# 라이브러리
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

# 데이터 확인
path = "https://raw.githubusercontent.com/Soyoung-Yoon/bigdata/main/"
train = pd.read_csv(path + "bank_train.csv")
test = pd.read_csv(path + "bank_test.csv")
# print(train.head(3), train.shape, sep="\n") # (3300, 14)
# print(test.head(3), test.shape, sep="\n")   # (1221, 13)

# 데이터 전처리
X = train.drop(columns=['term_deposit'])
Y = train['term_deposit']
X_all = pd.concat([X, test])
cols_obj = X_all.select_dtypes(include='object').columns
for col in cols_obj:
    X_all[col] = LabelEncoder().fit_transform(X_all[col])
# print(X_all.head(3))
# X_all = pd.get_dummies(X_all, drop_first=True, dtype='int32')
# print(X_all.head(3))

# 데이터 재분할
X = X_all.iloc[:len(X), :]
X_submission = X_all.iloc[len(X):, :]
# print(X.shape, X_submission.shape) # (3300, 13) (1221, 13)
temp = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=42)
x_train, x_test, y_train, y_test = temp
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (2310, 13) (990, 13) (2310,) (990,)

# 파이프라인 모델사전
models = {
    "Logistic": Pipeline([
        ("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=100, tol=0.05, random_state=42))
    ]),
    "DecisionTree": Pipeline([
        ("model", DecisionTreeClassifier(max_depth=3, random_state=42))
    ]),
    "RandomForest": Pipeline([
        ("model", RandomForestClassifier(max_depth=3, random_state=42))
    ]),
    "AdaBoost": Pipeline([
        ("model", AdaBoostClassifier(n_estimators=100, random_state=42))
    ]),
    "GradientBoosting": Pipeline([
        ("model", GradientBoostingClassifier(random_state=42))
    ])
}

# 성능평가 함수
def get_scores(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    y_pred_proba1 = model.predict_proba(x_train)[:, 1]
    y_pred_proba2 = model.predict_proba(x_test)[:, 1]
    AUC_train = roc_auc_score(y_train, y_pred_proba1)
    AUC_test = roc_auc_score(y_test, y_pred_proba2)
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

# 모델선택, 예측값 산출
model = models[res.loc[0, 'Model']]
y_pred1 = model.predict_proba(X_submission)[:, 1]
y_pred2 = model.predict(X_submission)

# 제출형식 생성
# pd.DataFrame({'pred': y_pred1}).to_csv("result_day5(1st)_ratio.csv", index=False)
# pd.DataFrame({'pred': y_pred2}).to_csv("result_day5(1st).csv", index=False)

# 결과확인
# temp = pd.read_csv("result_day5(1st).csv")
# print(temp['pred'].value_counts(normalize=True))
# print("=" * 35)
# print(Y[len(X_submission):].value_counts(normalize=True))

# 작업형 제3유형
# Day5. 다중 선형 회귀
# Advertising 데이터셋
# TV: TV 광고비(천 달러 단위), radio: 라디오 광고비, newspaper: 신문 광고비, sales: 제품 판매량 (단위: 천 개)
# 200개 데이터
import pandas as pd
url = "https://www.statlearning.com/s/Advertising.csv"
# 다음과 같은 다중선형회귀 모형을 사용한 회귀모델을 만들고 결과를 확인합니다.
# Advertising.csv 데이터를 사용합니다.
# 모델 생성시 상수항(=절편)을 포함하도록 합니다.
# 종속변수 : sales
# 독립변수 : sales를 제외한 모든 변수
# 처음부터 순서대로 150개 데이터를 train, 나머지 50개를 test로 사용합니다.
# 모델 생성시 train 데이터를 사용합니다.
df = pd.read_csv(url, index_col=0)
# print(df.head(3))
train = df.iloc[:150]
test = df.iloc[150:]

#5-1) 위의 조건에 맞게 OLS / OLS.from_formula 모델을 생성하고 summary를 출력한다.
from statsmodels.api import OLS
formula = "sales ~ TV + radio + newspaper"
model = OLS.from_formula(formula, train).fit()
# print(model.summary())

#5-2) 위에서 생성한 모델에서 적합된 모형 결정계수를 구해 반올림하여 소수점 아래 3자리까지 출력한다.
# print(model.rsquared.round(3)) # 0.896

#5-3) 위에서 생성한 모델에서 통계적으로 유의미한 변수는 몇 개인가? 유의수준 : 5%
# print(sum(model.pvalues[1:] < 0.05)) # 2

#5-4) 위의 모델에서 통계적으로 유의한 변수들과 TV와 Radio의 교호작용항을 사용하여
# 새롭게 모델링하여 model2를 생성한다.
formula2 = "sales ~ TV * radio + TV + radio"
model2 = OLS.from_formula(formula2, train).fit()
# print(model2.summary())

#5-5) 다음의 값을 사용하여 예측값을 구해, 반올림하여 소수점아래 4자리까지 출력한다.
# TV = 170.5, radio=13.2, newspaper=43.2
data = pd.DataFrame({'TV': [170.5], 'radio': [13.2], 'newspaper': [43.2]})
# print(round(model2.predict(data)[0], 4)) # 12.8629

#5-6) 다음의 모델은 통계적 유의미해석에 사용하는 가설에서 모델은 귀무가설을 기각하는가 채택하는가?
# 귀무가설 : 모든 독립변수가 종속변수에 영향을 주지 않는다.
# 대립가설 : 적어도 하나의 독립변수가 종속변수에 영향을 준다.
# 신뢰수준 : 95%
# for col in model2.params[1:].index:
#     print(f"{col}, 기각" if model2.pvalues[col] < 0.05 else f"{col}, 채택")
# 기각

# 5-7) 모델에서 가장 영향력 있는 변수의 t-value를 구해, 반올림하여 소수점 아래 3자리까지 출력한다.
temp = model2.params[1:].abs().idxmax()
# print(round(model2.tvalues[temp], 3)) # 2.125

# 5-8) 모델에서 가장 유의미한 변수의 회귀계수를 구해, 반올림하여 소수점 아래 4자리까지 출력한다.
temp = model2.pvalues[1:].abs().idxmin()
# print(round(model2.params[temp], 4)) # 0.0011

# # 5-9) train 데이터를 사용하여 해당 모델의 예측값과 실제값의 피어슨(pearson) 상관계수를 구하여라.
# 결과는 반올림하여 소수점 아래 3자리까지 출력한다.
x_train = train[['TV','radio','newspaper']]
x_true = train['sales']
x_pred = model2.predict(x_train)
# print(round(x_true.corr(x_pred), 3)) # 0.983

# 5-10) train 데이터를 사용하여 해당 모델의 예측값과 실제값의 스피어만(spearman) 상관계수를 구하여라.
# 결과는 반올림하여 소수점 아래 3자리까지 출력한다.
# print(round(x_true.corr(x_pred, method="spearman"), 3)) # 0.994

# 5-11) test 데이터를 사용하여 rmse를 구해, 반올림하여 소수점 아래 4자리까지 출력한다.
from sklearn.metrics import root_mean_squared_error as RMSE
y_train = test[['TV', 'radio', 'newspaper']]
y_true = test['sales']
y_pred = model2.predict(y_train)
# print(round(RMSE(y_true, y_pred), 4)) # 0.8665

# 5-12) train 데이터를 사용하여 잔차를 구하고, 잔차의 IQR을 구해, 반올림하여 소수점 아래 4자리까지 출력한다.
residual = x_true - x_pred
Q1, Q3 = residual.quantile([0.25, 0.75])
IQR = round(Q3 - Q1, 4)
# print(IQR) # 0.9658

#5-13) 통계적으로 가장 유의하지 않은 변수의 표준오차(Standard Error)를 소수점 아래 4자리까지 출력한다.
temp = model2.pvalues[1:].idxmax()
# print(round(model2.bse[temp], 4)) # 0.0104
