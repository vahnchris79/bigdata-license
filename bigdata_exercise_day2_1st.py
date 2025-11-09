
# 작업형 제1유형

import pandas as pd
pd.set_option('display.width', 120)

path1 = "https://raw.githubusercontent.com/Soyoung-Yoon/bigdata/main/"
path2 = "https://raw.githubusercontent.com/Soyoung-Yoon/data_01/main/"

# 2-1) 상위 지출 고객의 팁 분석
# 식사 지출이 높은 고객일수록 팁을 더 많이 주는지 알아보기 위해, 총 지출(total_bill)이 높은 고객들의 팁(tip)을 분석하고자 합니다.
# 다음 작업을 수행하세요.
# - 'total_bill' 컬럼에서 결측치를 제거한 뒤, 'total_bill' 기준 상위 20%에 해당하는 데이터를 추출합니다.
# - 위에서 추출된 데이터에 대해 'tip' 평균을 계산하여 결과로 제출합니다.
# - 결과는 반올림하여 소수점 아래 3자리까지 출력합니다.
df = pd.read_csv(path2 + "tips02.csv")
# print(df.isna().sum().to_frame().T, df.shape, sep="\n") # (244, 7

# 'total_bill' 컬럼에서 결측치를 제거
df = df[~df['total_bill'].isnull()].copy()
# print(df.isna().sum().to_frame().T)

# 'total_bill' 기준 상위 20%에 해당하는 데이터를 추출
df2 = df[df['total_bill'] >= df['total_bill'].quantile(0.8)]
# print(df2.shape)

# 위에서 추출된 데이터에 대해 'tip' 평균을 계산하여 결과로 제출
# print(round(df2['tip'].mean(), 3)) # 4.214

# 2-2) 고알코올 와인의 산도 특성 분석
# 알코올 도수가 높은 와인은 특정 종류(wine_variety)에 따라 화학적 특성이 다를 수 있습니다. 
# 이 문제에서는 고알코올 와인의 산도(malic_acid) 특성을 분석하고자 합니다.
# 다음 작업을 수행하세요.
# - wine.csv를 사용합니다.
# - alcohol 컬럼의 값이 중앙값보다 큰 데이터 중, wine_variety가 1인 데이터의 malic_acid 평균을 
# 계산하여 결과로 제출합니다.
# - 결과는 반올림하여 소수점 아래 3자리까지 출력합니다.
df = pd.read_csv(path1 + "wine.csv")
# alcohol 컬럼의 값이 중앙값보다 큰 데이터 중
df = df[df['alcohol'] > df['alcohol'].median()].copy()
# wine_variety가 1인 데이터의 malic_acid 평균을 계산하여 결과로 제출
# print(round(df[df['wine_variety'] ==1]['malic_acid'].mean(), 3)) # 1.274

# 2-3) 펭귄 부리 깊이 특성 분석
# 펭귄의 날개 길이와 부리 깊이는 종(species)에 따라 다르게 분포한다. 
# 이 문제에서는 특정 조건을 만족하는 펭귄들의 부리 깊이 특성을 분석하고자 한다.
# 다음 작업을 수행하세요.
# - penguins01.csv 파일을 사용하도록 합니다.
# - 모든 결측치를 제거합니다. (결측치가 1개라도 포함된 행 제거를 합니다.)
# - species가 'Chinstrap'이면서 flipper_length_mm이 40% 분위수 이상인 펭권들만 필터링하여 사용합니다.
# - 위에서 필터링된 데이터에서 bill_depth_mm이 해당 그룹의 중앙값보다 큰 개체의 수를 출력합니다.
df = pd.read_csv(path1 + "penguins01.csv")
# print(df.isna().sum().to_frame().T, df.shape, sep="\n") # (344, 7)

# 모든 결측치를 제거
df = df.dropna()
# print(df.isna().sum().to_frame().T)

# species가 'Chinstrap'이면서 flipper_length_mm이 40% 분위수 이상인 펭권들만 필터링
df = df[(df['species']=='Chinstrap') * (df['flipper_length_mm'] >= df['flipper_length_mm'].quantile(0.4))].copy()
# print(df.shape) #  (43, 7)

# 위에서 필터링된 데이터에서 bill_depth_mm이 해당 그룹의 중앙값보다 큰 개체의 수를 출력
# print(len(df[df['bill_length_mm'] > df['bill_length_mm'].median()])) # 20

# 2-4) 다이아몬드 컷 등급별 특성 비교
# 다이아몬드의 cut 등급에 따라 가격과 깊이(depth) 특성이 달라집니다. 
#  문제에서는 가장 저렴한 cut과 가장 비싼 cut의 내부 특성을 비교 분석합니다.
# 다음 작업을 수행하세요.
# - diamonds.csv 파일을 사용하도록 합니다.
# - 다음의 두 값(A, B)을 구하고, 이 두 값을 더한 결과를 정수로 출력합니다.
# - A : 평균 가격(price)이 가장 낮은 cut의 데이터 중, depth가 해당 cut의 데이터에서 중앙값 이상인 데이터의 개수
# - B : 평균 가격(price)이 가장 높은 cut의 데이터 중, depth가 해당 cut의 데이터 평균 depth보다 작은 데이터의 개수
df = pd.read_csv(path1 + "diamonds.csv")

# 평균 가격(price)이 가장 낮은 cut의 데이터 중, depth가 해당 cut의 데이터에서 중앙값 이상인 데이터의 개수
dfA = df[df['cut'] == df.groupby('cut')['price'].mean().idxmin()]
A = len(dfA[dfA['depth'] >= dfA['depth'].median()])
# 평균 가격(price)이 가장 높은 cut의 데이터 중, depth가 해당 cut의 데이터 평균 depth보다 작은 데이터의 개수
dfB = df[df['cut'] == df.groupby('cut')['price'].mean().idxmax()]
B = len(dfB[dfB['depth'] < dfB['depth'].mean()])
# print(int(A+B)) # 17441

# 2-5) 연도별 승객 증가와 최고 월 분석
# 승객 수는 연도와 월에 따라 다르게 변동하며, 특정 연도에 큰 성장이 일어날 수 있습니다. 
# 이 문제에서는 연도별 증가량과 해당 연도의 최고 월 승객 수를 분석합니다.
# 다음 작업을 수행하세요.
# - passengers.csv를 사용하세요.
# - 연도(year)별 총 승객 수(passengers)를 계산한 후, 연도 중 가장 승객 수가 많이 증가한 연도(year)를 구합니다.
# - 이때, 첫 번째 년도는 증가치를 0으로 사용합니다.
# - 위의 결과로 구한 연도(year)에서 가장 승객이 많았던 달(month)의 승객수를 정수로 출력합니다.
df = pd.read_csv(path1 + "flights.csv")
# 연도(year)별 총 승객 수(passengers)를 계산한 후, 연도 중 가장 승객 수가 많이 증가한 연도(year)를 구합니다.
year = df.groupby('year')['passengers'].sum().diff().fillna(0).idxmax()

# 위의 결과로 구한 연도(year)에서 가장 승객이 많았던 달(month)의 승객수를 정수로 출력
# print(int(df.loc[df['year']==year, ['month','passengers']]['passengers'].max())) # 622

# 작업형 제2유형
# Global Cancer Patients 데이터셋
# 2015년부터 2024년까지 보고된 전 세계 암 환자 데이터를 포함하고 있으며, 암의 진단, 치료, 생존에 영향을 미치는 주요 요인들을 
# 시뮬레이션한 데이터입니다.
# 제공된 학습용 데이터(cacner_train.csv)를 이용하여 암의 심각도(Severity)를 예측하는 모델을 개발하고, 
# 개발한 모델에 기반하여 평가용 데이터(cancer_test.csv)에 적용하여 얻은 암의 심각도 예측 값을 아래 [제출형식]에 따라 csv 파일로 생성하여 제출하시오.
# - 예측 결과는 F1 Score(macro) 평가지표에 따라 평가함
# [[제출 형식]]
# - 가. CSV 파일명: result.csv(파일명에 디렉토리/폴더 지정불가
# - 나. 예측 칼럼명 : pred
# - 다. 제출 칼럼 개수 : pred 칼럼 1개
# - 라. 평가용 데이터 개수와 예측 결과 데이터 개수 일치 : 2,193개

# 라이브러리
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score

# 데이터 불러오기
train = pd.read_csv(path1 + "cancer_train.csv")
test = pd.read_csv(path1 + "cancer_test.csv")
# print(train.head(3), train.shape, sep="\n") # (5115, 15)
# print(test.head(3), test.shape, sep="\n")   # (2193, 14)

# 데이터 전처리
X = train.drop(columns=['Severity'])
Y = train['Severity']
X_all = pd.concat([X, test]).drop(columns=['Patient_ID'])
cols_obj = X_all.select_dtypes(include='object').columns
for col in cols_obj:
    X_all[col] = LabelEncoder().fit_transform(X_all[col])
# print(X_all.head(3))
# X_all = pd.get_dummies(X_all, drop_first=True, dtype='int32')
# print(X_all.head(3))

# 데이터 분할
X = X_all.iloc[:len(X), :]
X_submission = X_all.iloc[len(X):, :]
# print(X.shape, X_submission.shape) # (5115, 13) (2193, 13)
temp = train_test_split(X, Y, test_size=0.3, random_state=123)
x_train, x_test, y_train, y_test = temp
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (3580, 13) (1535, 13) (3580,) (1535,)

# 파이프라인 모델사전
models = {
    "Logistic": Pipeline([
        ("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=100, tol=0.05, random_state=123))
    ]),
    "DecisionTree": Pipeline([
        ("model", DecisionTreeClassifier(max_depth=3, random_state=123))
    ]),
    "RandomForest": Pipeline([
        ("model", RandomForestClassifier(max_depth=3, random_state=123))
    ]),
    "AdaBoost": Pipeline([
        ("model", AdaBoostClassifier(n_estimators=100, random_state=123))
    ]),
    "GradientBoosting": Pipeline([
        ("model", GradientBoostingClassifier(random_state=123))
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

# 모델적용 후 예측
model = models[res.loc[0, "Model"]]
y_pred = model.predict(X_submission)

# 제출파일 생성
# pd.DataFrame({'pred': y_pred}).to_csv("result_day2_1st.csv", index=False)

# 결과확인
# temp = pd.read_csv("result_day2_1st.csv")
# print(temp['pred'].value_counts(normalize=True))
# print("=" * 35)
# print(Y[len(X_submission):].value_counts(normalize=True))

# 작업형 제3유형
# Day2 로지스틱회귀(유방암 데이터셋)
import pandas as pd
pd.set_option('display.width', 120)
df = pd.read_csv(path2 + "breast_cancer04.csv")
# print(df.head(3))

# 2-1) 종속변수는 'target', 범주의 종류와 개수확인
# print(df['target'].value_counts()) # 1: 357, 0: 212

# 2-2) 순서대로 398개는 train 데이터, 171개는 test 데이터로 사용합니다.
# train, test 분리 후, shape을 출력해서 확인하세요.
train = df.iloc[:398, :]
test = df.iloc[398:, :]
# print(train.shape, test.shape) # (398, 13) (171, 13)

# 다음과 같은 로지스틱회귀 모형을 사용한 분류모델을 만들고 결과를 확인합니다.
# - breast_cancer04.csv 데이터를 사용합니다.
# - 모델 생성시 상수항(=절편)을 포함하도록 하며, 규제는 사용하지 않습니다.
# - 종속변수 : target
# - 독립변수 : target을 제외한 모든 나머지 변수

# 2-3) GLM.from_formula() 를 사용해 분석하려고 합니다.
# formula를 작성하고, train 데이터를 사용해 로지스틱 회귀모형을 생성합니다.
# formula를 만들때 str.join()을 사용해 만들면 편하게 만들 수 있습니다.
# 생성 후, model.summary()를 출력해 봅니다.
from statsmodels.api import GLM, add_constant, families
formula = "target ~ " + " + ".join(df.columns[:-1])
train = add_constant(train)
model = GLM.from_formula(formula, train, family=families.Binomial()).fit()
# print(model.summary())

# 2-4) model에서 유의미한 설명변수만 선택하고, 그 개수를 출력합니다.
# 유의수준 0.05을 사용한다.
# print(sum(model.pvalues[1:] <= 0.05)) # 5

# 2-5) 학습데이터(train)를 사용하여, 2-4)에서 찾은 유의미한 설명변수만으로 로지스틱 회귀모형을 만듭니다.
# 생성 후, model.summary()를 출력해 봅니다.
formula2 = 'target ~ mean_radius + mean_area + mean_concave_points + compactness_error + worst_radius'
model2 = GLM.from_formula(formula2, train, family=families.Binomial()).fit()
# print(model2.summary())

# 2-6)  설명변수의 가장 높은 p-value를 구하여 주세요.
# 반올림하여 소수점 아래 3자리까지 출력합니다.
# print(round(model2.pvalues[1:].max(), 3)) # 0.908

# 2-7) 유의수준 0.05하에서 유의성이 낮은 변수의 개수는 몇 개인가요?
result = sum(model2.pvalues[1:] > 0.05)
# print(result) # 1

# 2-8)  평가 데이터를 사용하여 정확도를 구해 반올림하여 소수점 아래 3자리까지 출력합니다.
from sklearn.metrics import accuracy_score
y_true = df['target'][398:]
y_pred = model2.predict(test).round().astype('int32')
# print(round(accuracy_score(y_true, y_pred), 3)) # 0.959

# 2-9) 아래의 sample을 사용하여 P(Y=0)에 대한 확률을 구하고,
# 반올림하여 소수점 아래 3자리까지 출력합니다.
# sample => mean_radius : 14.2, mean_area: 650.0, mean_concave_points : 0.05, compactness_error : 0.02, worst_radius : 16.5
sample = {'mean_radius': [14.2], 'mean_area': [650.0], 'mean_concave_points': [0.05], 
          'compactness_error': [0.02], 'worst_radius': [16.5]}
data = pd.DataFrame(sample)
p_y1 = model2.predict(data)[0]
p_y0 = round(1 - p_y1, 3)
# print(p_y0) # 0.895

# 2-10)  위의 sample에 대한 오즈(odds)는?
# 결과는 반올림하여 소수점 아래 3자리까지 출력합니다.
odds = round(p_y1 / p_y0, 3)
# print(odds) # 0.118

# 2-11) 'mean_area'을 설명변수로 하였을 때의 오즈비(Odds Ratio)는?
# 결과는 반올림하여 소수점 아래 3자리까지 출력합니다.
import numpy as np
odds_ratio = np.exp(model2.params['mean_area'])
# print(round(odds_ratio, 3)) # 0.938

# 2-12) 'mean_radius'가 2증가하면 오즈는 몇 배 증가하는가?
# 반올림하여 정수로 출력합니다.
result = np.exp(model2.params['mean_radius'] * 2)
# print(round(result)) # 3849086

# 2-13) test 데이터에 대한 roc_auc 점수를 구합니다.
# 결과는 반올림하여 소수점 아래 3자리까지 출력합니다.
from sklearn.metrics import roc_auc_score
# print(round(roc_auc_score(y_true, y_pred), 3)) # 0.955
