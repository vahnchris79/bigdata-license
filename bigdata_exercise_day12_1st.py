
# 공통: 데이터 경로
path1 = "https://raw.githubusercontent.com/Soyoung-Yoon/bigdata/main/"
path2 = "https://raw.githubusercontent.com/Soyoung-Yoon/data_01/main/"
path3 = "https://raw.githubusercontent.com/Soyoung-Yoon/data_02/main/"

# 작업형 제1유형
# 4회-1번) 신상 데이터
# 'age' 컬럼은 사용자의 나이를 나타내는 정보이며, 나이 분포의 범위를 이해하기 위한 분석을 수행하고자 한다.
# 다음 절차에 따라 문제를 해결하시오.
import pandas as pd
df = pd.read_csv(path1 + "basic1_data04.csv")
# print(df.head())
# (가) age 컬럼의 제1사분위수와 제 3사분위수를 구하시오.
Q1, Q3 = df['age'].quantile([0.25, 0.75])
# (나) 두 값의 차이 절댓값을 구하시오.
result = abs(Q1 - Q3)
# (다) (나)의 결과를 소수점 이하를 버리고 정수로 출력한다.
# print(int(result)) # 50

# 4회-2번) Facebook 데이터
# Facebook 게시물에 대한 데이터이다. 데이터에 포함된, 사용자 반응 패턴을 분석을 수행하고자 한다.
# 다음 절차에 따라 문제를 해결하시오.
import pandas as pd
df = pd.read_csv(path1 + "fb_data04.csv")
# print(df.head(3))
# (가) 각 데이터에 대해 (loves + wows) 값을 reactions로 나눈 비율을 계산하시오.
df['ratio'] = (df['loves'] + df['wows']) / df['reactions']
# (나) 계산한 비율이 0.4보다 크고 0.5보다 작은 데이터를 필터링하시오.
df2 = df[(df['ratio'] > 0.4) & (df['ratio'] < 0.5)]
# (다) 위 조건을 만족하는 데이터 중에서 type이 'video'인 데이터의 개수를 구하여 정수형으로 출력한다
# print(int(len(df2[df2['type']=='video']))) # 90

# 4회-3번) 넷플릭스 데이터
# Netflix 콘텐츠 정보를 담은 데이터셋을 사용하여,
# 2018년 1월에 추가된 콘텐츠 중 제작 국가가 'United Kingdom'인 데이터를 검색하고자 한다.
# 다음 절차에 따라 문제를 해결하시오.
import pandas as pd
pd.set_option('display.width', 200)
df = pd.read_csv(path1 + "nf_data04.csv")
# print(df.head(3))
# - (가) date_added 컬럼에서 2018년 1월에 추가된 데이터를 선택하시오.
df['date_added'] = df['date_added'].astype('datetime64[ns]').astype(str)
df2 = df[df['date_added'].str.startswith('2018-01')]
# - (나) country 컬럼의 값이 'United Kingdom'만 포함된 데이터를 필터링하시오.
df3 = df2[df2['country']=='United Kingdom']
# - (다) 위 조건을 모두 만족하는 데이터의 개수를 정수형으로 출력한다.
# print(len(df3)) # 6

# 12-4) 총 매출액 구하기
# 특정 품목의 한 달간 매출액을 계산함으로써 품목별 성과를 모니터링하고 향후 판매 전략에 반영하고자 합니다.
# sales_data.csv : 매출 발생 일자(sale_date), 품목 ID(item_id), 판매 수량(quantity)을 포함하는 데이터
# price_list.csv : 품목 ID(item_id)와 해당 품목의 단가(unit_price)를 포함하는 데이터
# 두 데이터셋을 활용하여 다음 절차에 따라 문제를 해결하시오.
import pandas as pd
sales = pd.read_csv(path2 + "sales_data.csv")
price = pd.read_csv(path2 + "price_list.csv")
# 1. sales_data.csv와 price_list.csv를 item_id를 기준으로 병합(merge)하시오.
df = pd.merge(sales, price, how="left", on="item_id")
# print(df.head(3))
# 2. 병합된 데이터에 대해 매출액(sales_amount)을 구하시오.
#    > 매출액(sales_amount) = 판매 수량(quantity) × 단가(unit_price)
df['sales_amount'] = df['quantity'] * df['unit_price']
# print(df.head(3))
# 3. 2024년 6월에 판매된 품목 중 'ITEM_001'에 대한 총 매출액을 구하여, 정수형으로 출력한다.
result = int(df[(df['sale_date'].str.startswith('2024-06')) & (df['item_id']=='ITEM_001')]['sales_amount'].sum())
# print(int(result)) # 405720

# 12-5) 월별 최고 매출 품목의 빈도 분석
# 월별로 가장 높은 매출을 기록한 품목을 파악하여 인기 품목의 추세를 분석하고자 한다.
# sales_data01.csv : 매출 발생 일자(sale_date), 품목 ID(item_id), 판매 수량(quantity)을 포함한다.
# price_list.csv : 품목 ID(item_id)와 해당 품목의 단가(unit_price)를 포함한다.
# 두 데이터셋을 활용하여 다음 절차에 따라 문제를 해결하시오.
sales = pd.read_csv(path2 + "sales_data01.csv")
price = pd.read_csv(path2 + "price_list.csv")
# 1. sales_data01.csv와 price_list.csv를 item_id를 기준으로 병합(merge)하시오.
df = pd.merge(sales, price, how="left", on="item_id")
# print(df.head(3))
# 2. 병합된 데이터에 '년도(year)', '월(month)', '매출액(sales_amount) 컬럼을 추가하시오.
df['year'] = df['sale_date'].str[:4]
df['month'] = df['sale_date'].str[5:7]
df['sales_amount'] = df['quantity'] + df['unit_price']
# print(df.head(3))
#    > 매출액(sales_amount) = 판매 수량(quantity) × 단가(unit_price)
# 3. 각 (년도, 월, item_id)별로 매출액을 집계하시오.
s1 = df.groupby(['year','month','item_id'], as_index=False)['sales_amount'].sum()
# print(s)
# 4. 각 (년도, 월)별로 매출액이 가장 높은 item_id를 찾으시오.
s1 = s1.set_index('item_id')
result = s1.sort_values('sales_amount',ascending=False).idxmax()
# 5. 가장 높은 매출을 기록한 item_id의 빈도를 계산하여 가장 빈도가 높은 item_id의 숫자부분만 정수로 출력하시오.
# print(int(result.str.split('_')[1][1])) # 10

# 작업형 제2유형
# Global Cancer Patients 데이터셋
# 2015년부터 2024년까지 보고된 전 세계 암 환자 데이터를 포함하고 있으며, 암의 진단, 치료, 생존에 영향을 미치는 주요 요인들을 시뮬레이션한 데이터입니다.
# 제공된 학습용 데이터(cacner_train.csv)를 이용하여 암의 심각도(Severity)를 예측하는 모델을 개발하고, 개발한 모델에 기반하여 평가용 데이터(cancer_test.csv)에 적용하여 얻은 암의 심각도 예측 값을 아래 [제출형식]에 따라 csv 파일로 생성하여 제출하시오.
# - 예측 결과는 F1 Score(macro) 평가지표에 따라 평가함
# - 성능이 우수한 예측 모델을 구축하기 위해서는 데이터 정제, Feature Engineering, 하이퍼 파라미터(hyper parameter) 최적화, 모델 비교 등이 필요할 수 있음. 다만, 과적합에 유의하여야 함
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

# 데이터 확인
train = pd.read_csv(path1 + "cancer_train.csv")
test = pd.read_csv(path1 + "cancer_test.csv")
# print(train.head(3), train.info(), sep="\n") # 5115
# print(test.head(3), test.info(), sep="\n")   # 2193

# 데이터 전처리
X = train.drop(columns=['Severity'])
Y = train['Severity']
X_all = pd.concat([X, test]).drop(columns=['Patient_ID'])
# print(X_all.info())
X_all['Country_Region'] = LabelEncoder().fit_transform(X_all['Country_Region'])
X_all['Cancer_Type'] = LabelEncoder().fit_transform(X_all['Cancer_Type'])
X_all['Cancer_Stage'] = LabelEncoder().fit_transform(X_all['Cancer_Stage'])
X_all = pd.get_dummies(X_all, drop_first=True, dtype='int32')
# print(X_all.head(3))

# 데이터 재분할
X = X_all.iloc[:len(X), :]
X_submission = X_all.iloc[len(X):, :]
# print(X.shape, X_submission.shape) # (5115, 14) (2193, 14)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=500)
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (3580, 14) (1535, 14) (3580,) (1535,)

# 파이프라인 모델사전
models = {
    "Logistic": Pipeline([
        ("scaler", MinMaxScaler()), ("model", LogisticRegression(max_iter=500, tol=0.05, random_state=500))
    ]),
    "DecisionTree": Pipeline([
        ("model", DecisionTreeClassifier(max_depth=5, random_state=500))
    ]),
    "RandomForest": Pipeline([
        ("model", RandomForestClassifier(n_estimators=500, max_depth=5, random_state=500))
    ]),
    "AdaBoost": Pipeline([
        ("model", AdaBoostClassifier(n_estimators=500, random_state=500))
    ]),
    "Gradient": Pipeline([
        ("model", GradientBoostingClassifier(random_state=500))
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

# 모델적합, 예측값 산출
model = models[res.loc[0, "Model"]]
y_pred = model.predict(X_submission)
# print(y_pred)

# 제출파일 생성
# pd.DataFrame({'pred': y_pred}).to_csv("result_day12.csv", index=False)

# 결과확인
# temp = pd.read_csv("result_day12.csv")
# print(Y[:len(test)].value_counts(normalize=True))
# print("=" * 35)
# print(temp['pred'].value_counts(normalize=True))

# 작업형 제3유형
# 1.감기약 vs 감기약의 위약 부작용
# 감기약을 복용할 때 부작용에 대한 유형과 비율 데이터를 사용해 감기약의 위약 효과가 있는지 253건의 데이터를 추출하여 검증하고자 한다.
# 감기약의 부작용에 대한 유형과 비율이 아래의 표와 같을 때, 감기약의 위약 효과 부작용 비율과 같은지를 카이제곱 검정으로 검증하여 보자.
df = pd.read_csv(path1 + "flu_side_effects02.csv")
# print(df.T)
# 1) 감기약의 위약 표본 데이터에서 '부작용 없음'인 데이터의 비율을 0~1 사이의 값으로 출력하시오. (반올림하여 소수점 아래 셋째자리까지 계산)
observed = df['코드'].value_counts().sort_index()
# print(observed / 253)
import numpy as np
expected = len(df) * np.array([0.05, 0.1, 0.05, 0.8])
# print(expected)
# 2) 카이제곱 검정 결과에서 검정통계량을 구해 출력하시오. (반올림하여 소수점 아래 셋째자리까지 계산)
from scipy.stats import chisquare
statistic, pvalue = chisquare(observed, expected)
# print(round(statistic, 3)) # 0.997
# 3) 카이제곱 검정 결과에서 유의 확률(p-value)를 출력하시오. (반올림하여 소수점 아래 셋째자리까지 계산)
# print(round(pvalue, 3)) # 0.802

# 2.다중 선형 회귀
# 주어진 데이터는 공기질 측정을 기반으로 하며, 오존 농도, 태양 복사량, 풍속, 기온 등의 변수를 포함한다.
# Ozone : 오존 농도(ppb)
# Solar : 일사량
# Wind : 풍속(mph)
# Temperature: 최소 기온(화씨)
# > 독립변수 : Ozone, Solar, Wind, 종속변수 : Temperature
# 위의 조건에 따라 다중 선형 회귀 모델을 생성하여 다음 물음에 답하시오.
df = pd.read_csv(path2 + "airquality.csv")
# print(df.head(3))
from statsmodels.api import OLS
formula = "Temperature ~ Ozone + Solar + Wind"
model = OLS.from_formula(formula, df).fit()
# print(model.summary())
# 1. 위에서 생성된 모델에 대해 오존농도(Ozone) 변수의 회귀계수를 반올림하여 소수점 아래 3자리까지 출력하시오.
# print(round(model.params['Ozone'], 3)) # 0.172

# 2. 오존농도(Ozone)와 일사량(Solar) 변수 값이 고정된 상태에서 풍속(Wind)이 증가함에 따라 온도(Temperature)가 낮아진다는 것을 검증했다. 
#    t-검증값의 유의 확률(p-value)를 출력하시오. (반올림하여 소수점 아래 3자리까지 계산)
# print(round(model.pvalues['Wind'], 3)) # 0.169

# 3. 위 모델을 기반으로 오존농도 10, 일사량 90, 풍속 20일 때 온도의 값을 예측하여 출력하시오.(반올림하여 소수점 아래 3자리까지 계산)
data = pd.DataFrame({'Ozone': [10], 'Solar': [90], 'Wind': [20]})
result = model.predict(data)[0]
# print(round(result, 3)) # 68.334

# 3. 로지스틱 회귀
# 다음은 전복의 생물학적 특성과 성별(Gender)에 대한 데이터이다.
# 총 300개 샘플로 구성되어 있으며, 다음과 같이 학습용과 평가용으로 분할하여 사용한다.
# 학습용: 1 ~ 210번 샘플 (학습 모델 생성에 사용), 평가용: 210 ~ 300번 샘플
# 상수항(절편)을 포함하고, 규제는 적용하지 않는다.
# 단, Gender는 이진 변수(0: 암컷, 1: 수컷)로 처리되어 있으며, 분석에 적합한 형태로 변환되어 있다.
df = pd.read_csv(path2 + "abalone.csv")
train = df.iloc[:210, ]
test = df.iloc[210:, ]
# print(train.shape, test.shape) # (210, 6) (90, 6)
# 위 데이터를 바탕으로 다음 물음에 답하시오.
# 1. Gender를 종속변수, Weight를 독립변수로 사용하여 로지스틱 회귀 모형을 만들고, Weight 변수가 한 단위 증가할 때 
#    수컷일 오즈비(odds ratio)를 소수점 아래 3자리까지 반올림하여 구하시오. (학습용 데이터 이용)
from statsmodels.api import GLM, families
import numpy as np
formula = "Gender ~ Weight"
model = GLM.from_formula(formula, train, family=families.Binomial()).fit()
# print(model.summary())
# print(round(np.exp(model.params['Weight']), 3)) # 0.772 → 0.791

# 2. Gender를 종속변수, 나머지 5개의 변수를 독립변수로 하여, 적합한 로지스틱 회귀 모형의 잔차이탈도(residual deviance)를 
#    반올림하여 소수점 아래 4자리까지 구하시오.  (학습용 데이터 이용)
# print(train.head(3))
formula2 = "Gender ~ " + " + ".join(train.columns.drop('Gender'))
model2 = GLM.from_formula(formula2, train, family=families.Binomial()).fit()
# print(model2.summary())
# print(round(model2.deviance, 4)) # 280.3017 → 279.8859
#  
# 3. 1번에서 생성한 로지스틱 회귀 모형에 평가용 데이터를 적용해 Gender를 예측하고, 오분류율(Error Rate)을 반올림하여 소수점 아래 3자리까지 구하시오.
# print(test.head(3))
from sklearn.metrics import accuracy_score
y_true = test['Gender']
test = test.drop(columns=['Gender'])
y_pred = model.predict(test).round(0).astype('int')
# print(y_pred)
# print(round(1 - accuracy_score(y_true, y_pred), 3)) 0.456

# 4.다중 선형 회귀
# 여러 개의 독립변수를 기반으로 target 값을 예측하기 위해 다중 선형 회귀모형을 구축하시오.
# 종속변수: target
# 독립변수: target을 제외한 모든 변수
# 위 데이터를 바탕으로 다음 물음에 답하시오.
import pandas as pd
df = pd.read_csv(path1 + "mlr_noisy.csv")
# print(df.head(3))
# 1. feature3과 가장 강한 상관관계에 있는 변수와의 상관계수를 구하여 반올림하여 소수점 아래 3자리까지 출력하시오.
from statsmodels.api import OLS
corr = df.corr()['feature_3'].drop('feature_3')
# print(corr)
temp = corr.abs().idxmax() # feature_59
# print(temp)
result1 = corr[temp]
# 2. 선형 회귀 모형의 적합된 결정계수를 구해, 반올림하여 소수점 아래 4자리까지 출력하시오.
formula = "target ~ " + " + ".join(df.columns.drop('target'))
# print(formula)
model = OLS.from_formula(formula, df).fit()
result2 = model.rsquared
# 3. 변수들의 p-value 중 가장 높은 p-value를 구해, 반올림하여 소수점 아래 3자리까지 출력하시오.
result3 = model.pvalues[1:].max()
# print(f"상관계수: {round(result1, 3)}, 결정계수: {round(result2, 4)}, p-value: {round(result3, 3)}")
# 상관계수: -0.086, 결정계수: 0.9847, p-value: 0.996