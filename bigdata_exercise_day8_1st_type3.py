# 작업형 제3유형
import pandas as pd
pd.set_option('display.width', 120)
path1 = "https://raw.githubusercontent.com/Soyoung-Yoon/bigdata/main/"
path2 = "https://raw.githubusercontent.com/Soyoung-Yoon/data_01/main/"
# 8회 1번. 로지스틱 회귀
# 고객의 서비스 이용 특성과 이탈 여부(Churn)를 포함한 데이터를 바탕으로 이탈여부 예측을 위한 로지스틱 회귀 분석을 수행합니다.
# 종속변수: churn (이탈 여부)
# 독립변수: churn을 제외한 모든 변수
# 단, 상수항(절편)을 포함하고, 규제는 적용하지 않는다.
# 위의 조건에 따라 모형을 생성, 학습하여 다음 물음에 답하시오.
df = pd.read_csv(path1 + "churn_08031.csv")
# print(df.head(3), df.info(), sep="\n")
df.columns = df.columns.str.replace(" ", "_")
from statsmodels.api import GLM, add_constant, families
formula = "churn ~ " + " + ".join(df.columns[:-1])
# print(formula)
df = add_constant(df)
model = GLM.from_formula(formula, df, family=families.Binomial()).fit()
# print(model.summary())
# 1. 유의수준 0.05 하에서, 유의성이 낮은 변수의 개수를 구하라.
# print(sum(model.pvalues[1:] > 0.05)) # 12
# 2. 위 결과에서 통계적으로 유의한 변수들만 사용하여 다시 모델을 구축한 뒤, 회귀식의 모든 계수(절편 포함)의 평균을 구하라.
# print(model.pvalues[1:] <= 0.05) # vmail_message,intl_calls,number_customer_calls
formula2 = "churn ~ vmail_message + intl_calls + number_customer_calls"
model2 = GLM.from_formula(formula2, df, family=families.Binomial()).fit()
# print(model2.summary())
# print(round(model2.params.mean(), 3)) # -0.435
# 3. number_customer_calls가 5증가하면 오즈비는 몇 배 증가하는가?
# - 단, 실수의 경우 반올림하여 소수점 아래 3자리까지 표시한다.
import numpy as np
result = round(np.exp(model2.params['number_customer_calls'] * 5), 3)
# print(result) # 6.511

# 8회 2번. 다중선형회귀
# 신체 정보에 대한 데이터를 바탕으로 개인의 인지 지수(PIQ)를 예측하기 위한 다중 선형 회귀 분석을 수행합니다.
# 종속변수: PIQ
# 독립변수: Brain, Height, Weight
# 위의 조건에 따라 모형을 생성, 학습하여 다음 물음에 답하시오.
import pandas as pd
df = pd.read_csv(path1 + "iqsize_08032.csv")
# print(df.head(3), df.shape, sep="\n")
from statsmodels.api import OLS
formula = "PIQ ~ Brain + Height + Weight"
model = OLS.from_formula(formula, df).fit()
# print(model.summary())
# 1. 위의 모델에서 통계적으로 가장 유의미한 변수의 회귀계수를 구하시오.
temp = (model.pvalues[1:] < 0.05).idxmax()
# print(model.params[temp]) # 2.0277326073473505
# 2. 위 모델의 결정계수 값을 구하시오.
# print(model.rsquared) # 0.3128659546345375
# 3. 위에서 구한 모델을 사용하여 키:70, 몸무게:150, 뇌크기:90 인 경우의 PIQ 값을 구하시오.
# 단, 실수의 경우 반올림하여 소수점 아래 3자리까지 표시한다.
data = pd.DataFrame({'Height': [70], 'Weight': [150], 'Brain': [90]})
# print(round(model.predict(data)[0], 3)) # 104.768

# 9회 1번. 다중선형회귀
# 미술학과 학생들의 시험 성적 데이터를 바탕으로 디자인 과목 점수(design)를 예측하기 위한 다중 선형 회귀 분석을 수행합니다.
# 해당 데이터를 다음 기준에 따라 훈련용 데이터(train)와 평가용 데이터(test)로 나누어 사용한다.
# train : 140개, test : 나머지 데이터
# 종속변수: design (디자인 과목 점수)
# 독립변수: drawing, color_theory, composition, modeling_3D
# 위의 조건에 따라 모형을 생성, 학습하여 다음 물음에 답하시오.
df = pd.read_csv(path1 + 'design_data.csv')
train = df.iloc[:140, :]
test= df.iloc[140:, :]
from statsmodels.api import OLS
formula = "design ~ drawing + color_theory + composition + modeling_3D"
model = OLS.from_formula(formula, train).fit()
# print(model.summary())
# 1. 유의수준 0.05 하에서 통계적으로 유의한 변수의 개수는 몇 개인가?
# print(sum(model.pvalues[1:] < 0.05)) # 2
# 2. (1)에서 구한 유의한 변수만 사용하여 다시 모델을 만들고, train 데이터를 사용하여 해당 모델의 예측값과 실제값의 피어슨 상관계수를 구하여라.
# print(model.pvalues[1:] < 0.05) # drawing, color_theory
formula2 = "design ~ drawing + color_theory"
model2 = OLS.from_formula(formula2, train).fit()
y_true = train['design']
y_pred = model2.predict(train.drop(columns=['design']))
temp = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred}).corr()
# print(temp)
result = temp.loc['y_true', 'y_pred']
# print(round(result, 3)) # 0.802
# 3. (2)에서 생성된 모델을 사용해 test 데이터에 대한 rmse를 구하여라.
# - 단, 실수의 경우 반올림하여 소수점 아래 3자리까지 표시한다.
from sklearn.metrics import root_mean_squared_error as rmse
y_true = test['design']
y_pred = model.predict(test)
print(round(rmse(y_true, y_pred), 3)) # 7.816

# 9회 2번. 로지스틱 회귀
# 고객의 서비스 이용 특성과 이탈 여부(Churn)를 포함한 데이터를 바탕으로 이탈여부 예측을 위한 로지스틱 회귀 분석을 수행합니다.
# 종속변수: Churn(이탈여부, 이진 변수)
# 독립변수: MonthlyCharges, ServiceSatisfactionScore, FamilyPlanSize, PhoneService
# 단, 상수항(절편)을 포함하고, 규제는 적용하지 않는다.
# 위의 조건에 따라 모형을 생성, 학습하여 다음 물음에 답하시오.
df = pd.read_csv(path1 + "churn_delco.csv")
# print(df.info())
from statsmodels.api import GLM, families
formula = "Churn ~ MonthlyCharges + ServiceSatisfactionScore + FamilyPlanSize + PhoneService"
model = GLM.from_formula(formula, df, family=families.Binomial()).fit()
# print(model.summary())
# 1. 위 모델에서'FamilyPlanSize'에 대한 p-value를 구하시오.
# print(model.pvalues['FamilyPlanSize']) # 0.04241074831969642
# 2. 위 모델에서 다른 변수의 영향은 무시하고 'PhoneService' 변수가 0에서 1이 됐을 때의 오즈비를 구하시오.
import numpy as np
# print(np.exp(model.params['PhoneService'])) # 0.6881706474636209
# 3. 생성된 모델에 데이터를 넣어 고객이 이탈 할 확률값을 예측한 후, 그 확률값이 0.4보다 큰 고객의 수를 구하여 정수로 출력하시오.
# 단, 실수의 경우 반올림하여 소수점 아래 3자리까지 표시한다.
result = sum(model.predict(df) > 0.4)
print(result) # 426