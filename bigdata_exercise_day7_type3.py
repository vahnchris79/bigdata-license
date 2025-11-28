
# 공통
path1 = "https://raw.githubusercontent.com/Soyoung-Yoon/bigdata/main/"
path2 = "https://raw.githubusercontent.com/Soyoung-Yoon/data_01/main/"

# 작업형 제3유형
# 6회 2번. 다중 선형 회귀
# 다음은 연령(age), 몸무게(weight), 콜레스테롤 수치(cholesterol)에 대한 일부 표본 데이터를 사용하여 선형 회귀 분석을 수행하여 다음 물음에 답하시오.
# 종속변수 : weight
# 독립변수 : age, cholesterol
# 위의 조건에 따라 모형을 생성, 학습하여 다음 물음에 답하시오.
# 위에서 생성된 모델에 대해 age의 회귀계수를 구하고, 반올림하여 소수점 아래 3자리까지 출력하시오.
# age가 고정된 상태에서 cholesterol과 weight 사이에 선형관계가 존재한다는 가설을 세운다. 
# 이 가설을 유의수준 0.05 하에에 검정하고, 통계적으로 유의미한 관계가 있는지 여부를 "있음" 또는 "없음"으로 표시하시오.
# 위 모델을 기반으로 age = 55, cholesterol = 72.6일 때의 weight 값을 예측하고, 
# 반올림하여 소수점 아래 4자리까지 출력하시오.
import pandas as pd
df1 = pd.read_csv(path1 + "cholesterol.csv")
# print(df1.head(3))
from statsmodels.api import OLS
formula = "weight ~ age + cholesterol"
model = OLS.from_formula(formula, df1).fit()
# print(model.summary())
result1 = round(model.params['age'], 3)
# print(result1) # -0.036
result2 = "없음" if model.pvalues['cholesterol'] > 0.05 else "있음"
# print(result2) # 있음
data = pd.DataFrame({'age': [55], 'cholesterol': [72.6]})
result3 = round(model.predict(data)[0], 4)
# print(result3) # 78.6219

# 7회 1번. 로지스틱 회귀
# 다음은 생물학적 특성(age, diameter, height, weight)과 성별(gender)에 대한 데이터이다.
# 총 300개 샘플로 구성되어 있으며, 다음과 같이 학습용과 평가용으로 분할하여 사용한다.
# 학습용: 1 ~ 210번 샘플 (학습 모델 생성에 사용)
# 평가용: 211 ~ 300번 샘플
# 종속변수: gender, 독립변수: age, diameter, height, weight
# 상수항(절편)을 포함하고, 규제는 적용하지 않는다.
# 단, gender는 이진 변수로 처리되어 있으며, 분석에 적합한 형태로 변환되어 있다.
# 위 데이터를 바탕으로 로지스틱 회귀 분석을 수행하여 다음 물음에 답하시오.
# 로지스틱 회귀 모형에서 'weight' 변수를 설명변수로 사용할 때의 오즈비(odds ratio)를 
# 소수점 아래 3자리까지 반올림하여 구하시오.
# 로지스틱 회귀 모형의 잔차이탈도(residual deviance)를 반올림하여 소수점 아래 4자리까지 구하시오.
# 로지스틱 회귀 모형에 평가용 데이터를 적용해, gender를 예측하고, 예측값과 실제값 간의 
# 오차율(error rate)을 반올림하여 소수점 아래 3자리까지 구하시오.
df2 = pd.read_csv(path1 + "gender_classification.csv")
train = df2.iloc[:210, :]
test = df2.iloc[210: ,:]
# print(train.shape, test.shape, sep=", ") # (210, 5), (90, 5)
from statsmodels.api import GLM, families
formula = "gender ~ age + diameter + height + weight"
model = GLM.from_formula(formula, train, family=families.Binomial()).fit()
# print(model.summary())
import numpy as np
result4 = round(np.exp(model.params['weight']), 3)
# print(result4) # 0.997
result5 = round(model.deviance, 4)
# print(result5) # 57.2937
from sklearn.metrics import accuracy_score
y_true = test['gender']
y_pred = model.predict(test).round().astype(int)
acc = accuracy_score(y_true, y_pred)
# result6 = round(1-acc, 3)
proba = model.predict(test)
y_pred = round(proba)
result6 = round((test['gender'] != y_pred).mean(), 3)
print(result6) # 0.033

# 7회 2번. 다중 선형 회귀
# 여러 개의 독립변수를 기반으로 target 값을 예측하기 위해 다중 선형 회귀모형을 구축하시오.
# 종속변수: target
# 독립변수: target을 제외한 모든 변수
# 위의 구축된 회귀모형을 사용하여 다음 물음에 답하시오.
# 가장 높은 회귀계수를 구해, 반올림하여 소수점 아래 3자리까지 출력하시오.
# 적합된 선형 회귀 모형의 결정계수를 구해, 반올림하여 소수점 아래 4자리까지 출력하시오.
# 독립변수 중 가장 높은 p-value를 구해, 반올림하여 소수점 아래 3자리까지 출력하시오.
df3 = pd.read_csv(path1 + "mlr_noisy.csv")
# print(df3.head(3))
from statsmodels.api import OLS
formula = "target ~ " + " + ".join(df3.columns[:-1])
model = OLS.from_formula(formula, df3).fit()
# print(model.summary())
result7 = round(model.params[1:].max(), 3)
# print(result7) # 84.177
result8 = round(model.rsquared, 4)
# print(result8) # 0.9847
result9 = round(model.pvalues[1:].max(), 3)
# print(result9) # 0.996