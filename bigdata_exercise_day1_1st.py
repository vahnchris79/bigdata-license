
# 작업형 제1유형
# 펭귄 데이터셋을 사용하여 다음을 구하여 출력하세요
# species가 'Gentoo'인 데이터 중 flipper_length_mm이 상위 20% 이상인 개체들의 body_mass_g 평균값은?
# flipper_length_mm 및 body_mass_g의 결측치는 species가 'Gentoo'인 표본의 평균값으로 채우며,
# 결과는 반올림하여 소수점 아래 3자리까지 출력한다.

path = "https://raw.githubusercontent.com/Soyoung-Yoon/data_01/main/"

import pandas as pd
pd.set_option('display.width', 120)

df = pd.read_csv(path + "penguins04.csv")
# print(df.head(3))

# species가 'Gentoo'인 데이터 추출
df_gentoo = df[df['species'] == 'Gentoo'].copy()
# flipper_length_mm 및 body_mass_g의 결측치는 species가 'Gentoo'인 표본의 평균값으로 채우며
mean_length = df_gentoo['flipper_length_mm'].mean()
mean_mass = df_gentoo['body_mass_g'].mean()
df_gentoo['flipper_length_mm'] = df_gentoo['flipper_length_mm'].fillna(mean_length)
df_gentoo['body_mass_g'] = df_gentoo['body_mass_g'].fillna(mean_mass)
# flipper_length_mm이 상위 20% 이상인 개체들의 body_mass_g 평균값
# result = df[df['flipper_length_mm'] >= df['flipper_length_mm'].quantile(0.8)]['body_mass_g'].mean()
s = df_gentoo['flipper_length_mm']
cond = (s >=s.quantile(0.8))
result = df_gentoo.loc[cond, 'body_mass_g'].mean()
# print(round(result, 3)) # 5543.75

# body_mass_g변수에서 이상치를 제거한 후(평균 ± 1.5 × 표준편차 기준), 남은 데이터 중 sex가 'Female'인
# 펭귄들의 bill_length_mm 평균값은?
# 결측치를 포함한 모든 행을 제거하고 사용, 결과는 반올림하여 소수점 아래 3자리까 출력
df = pd.read_csv(path + "penguins04.csv")

# 결측치를 포함한 모든 행을 제거
df = df.dropna(axis=0)
# print(df.head())
# 이상치를 제거한 후(평균 ± 1.5 × 표준편차 기준),
mean_mass = df['body_mass_g'].mean()
std_mass = df['body_mass_g'].std(ddof=1)
lower = mean_mass - (1.5 * std_mass)
upper = mean_mass + (1.5 * std_mass)

# 남은 데이터 중 sex가 'Female'인 펭귄들의 bill_length_mm 평균값
# result = df[(df['body_mass_g'] > lower) | (df['body_mass_g'] < upper) & (df['sex'] == 'Female')]['bill_length_mm'].mean()
df = df[(df['body_mass_g'] >= lower) & (df['body_mass_g'] <= upper)]
result = df.loc[df['sex'] == 'Female', 'bill_length_mm'].mean()
# print(round(result, 3)) # 42.352

# flipper_length_mm의 결측치를 해당 컬럼의 중앙값으로 대체한 후, '대체 전의 표준편차' - 
# '대체 후 표준편차'값을 구하면? 결과는 반올림하여 소수점 3자리까지 출력한다.
df = pd.read_csv(path + "penguins01.csv")
# flipper_length_mm의 결측치를 해당 컬럼의 중앙값으로 대체
df2 = df.copy()
df2['flipper_length_mm'] = df2['flipper_length_mm'].fillna(df2['flipper_length_mm'].median())

# '대체 전의 표준편차' - '대체 후 표준편차'값
before_std = df['flipper_length_mm'].std(ddof=1)
after_std = df2['flipper_length_mm'].std(ddof=1)
result = before_std - after_std
# print(round(result, 3)) # 0.038

# species의 body_mass_g의 평균의 차이가 가장 큰 두 가지 품종의 sex이 Female인 표본개수는?
# 모든 결측치는 제거한 뒤 작업하도록 한다.
df = pd.read_csv(path + "penguins01.csv")

# 모든 결측치는 제거
df = df.dropna(axis=0)

# species의 body_mass_g의 평균의 차이가 가장 큰 두 가지 품종
min_species = df.groupby('species')['body_mass_g'].mean().idxmin()
max_species = df.groupby('species')['body_mass_g'].mean().idxmax()

# 품종의 sex이 Female인 표본개수는
A = len(df.loc[(df['species']==min_species)&(df['sex']=='Female')])
B = len(df.loc[(df['species']==max_species)&(df['sex']=='Female')])
print(A+B) # 131

# bill_length_mm기준으로 상위20%와 하위10%에 해당하는 개체들의 body_mass_g의 평균값을 각각 구한 뒤,
# 이 두 평균값의 차이를 구하면?
# 단, 모든 결측치를 제거한 후 작업하고 결과는 반올림하여 소수점 아래 3자리까지 출력
df = pd.read_csv(path + "penguins05.csv")

# 모든 결측치를 제거
df = df.dropna(axis=0)

# bill_length_mm기준으로 상위20%에 해당하는 개체들의 body_mass_g의 평균값
upper_mean = df[df['bill_length_mm'] >= df['bill_length_mm'].quantile(0.8)]['body_mass_g'].mean()
lower_mean = df[df['bill_length_mm'] <= df['bill_length_mm'].quantile(0.1)]['body_mass_g'].mean()
print(round(upper_mean - lower_mean, 3)) # 1262.879

# 작업형 제2유형
# Bank Marketing 데이터셋
# 고객의 인구통계적 정보 및 이전 마케팅 이력 데이터를 바탕으로 '이 사람이 정기예금 상품에 가입할 것인가?'를
# 예측하는 것이 목적
# 제공된 학습용 데이터(bank_train.csv)를 이용하여 정기예금 상품에 가입 여부를 예측하는 모델을 개발하고,
# 개발한 모델에 기반하여 평가용 데이터(bank_test.csv)에 적용하여 얻은 정기예금 상품에 가입 여부 예측 확률을
# 아래 제출형식에 따라 csv파일로 생성하여 제출하시오.
# 예측결과는 ROC_AUC 평가지표에 따라 평가
# [[ 제출 형식 ]]
# 1. CSV파일명: result.csv
# 2. 예측 성별 칼럼명: pred
# 3. 제출 칼럼 개수: pred칼럼 1개
# 4. 평가용 데이터 개수와 예측 결과 데이터 개수 일치: 1,221개

# 라이브러리
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

# 데이터 불러오기
path = "https://raw.githubusercontent.com/Soyoung-Yoon/bigdata/main/"
train = pd.read_csv(path + "bank_train.csv")
test = pd.read_csv(path + "bank_test.csv")
# print(train.isna().sum().to_frame().T)
# print(test.isna().sum().to_frame().T)

# 데이터 전처리
X = train.drop(columns=['term_deposit'])
Y = train['term_deposit']
X_all = pd.concat([X, test])
cols_obj = X_all.select_dtypes(include='object').columns
for col in cols_obj:
    X_all[col] = LabelEncoder().fit_transform(X_all[col])
# print(X_all.head(3))
# X_all = pd.get_dummies(X_all, drop_first=True, dtype='int')
# print(X_all.head(3))

# 데이터 분할
X = X_all.iloc[:len(X), :]
X_submission = X_all.iloc[len(X):, :]
# print(X.shape, X_submission.shape) # (3300, 13) (1221, 13)
temp = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=100)
x_train, x_test, y_train, y_test = temp
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (2310, 13) (990, 13) (2310,) (990,)

# 파이프라인 모델사전
models = {
    "Logistic": Pipeline([
        ('scaler', StandardScaler()), ('model', LogisticRegression(max_iter=100, tol=0.05, random_state=100))
    ]),
    "DecisionTree": Pipeline([
        ('model', DecisionTreeClassifier(max_depth=3, random_state=100))
    ]),
    "RandomForest": Pipeline([
        ('model', RandomForestClassifier(max_depth=3, random_state=100))
    ]),
    "AdaBoost": Pipeline([
        ('model', AdaBoostClassifier(n_estimators=100, random_state=100))
    ]),
    "GradientBoosting": Pipeline([
        ('model', GradientBoostingClassifier(random_state=100))
    ])
}

# 성능평가 함수
def get_scores(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    y_proba1 = model.predict_proba(x_train)[:, 1]
    y_proba2 = model.predict_proba(x_test)[:, 1]
    ACC_train = roc_auc_score(y_train, y_proba1)
    ACC_test = roc_auc_score(y_test, y_proba2)
    return model, ACC_train, ACC_test

# 모델별 성능평가
results = []
for name, model in models.items():
    model, ACC_train, ACC_test = get_scores(model, x_train, x_test, y_train, y_test)
    results.append({
        "Model": name, "ACC_train": f"{ACC_train:.4f}", "ACC_test": f"{ACC_test:.4f}"
    })
# res = pd.DataFrame(results).sort_values("ACC_test", ascending=False).reset_index(drop=True)
# print(res)

# 모델선택, 예측산출
# model = models[res.loc[0, 'Model']]
# y_proba = model.predict_proba(X_submission)[:, 1]

# 제출파일 생성
# pd.DataFrame({'pred': y_proba}).to_csv("result_day1_type2.csv", index=False)

# 결과검토
# temp = pd.read_csv("result_day1_type2.csv")
# print(temp['pred'].describe())
# print("=" *  30)
# print(Y[:len(X_submission)].describe())

# 작업형 제3유형
# Day1. 로지스틱 회귀(와인 데이터셋)

# 데이터 확인
path = "https://raw.githubusercontent.com/Soyoung-Yoon/data_01/main/"
df = pd.read_csv(path + "wine01.csv")
# print(df.head(3))

# 1-1) 종속변수는 'wine_variety'입니다. 범주의 종류와 개수 확인
# print(df['wine_variety'].value_counts()) # 1: 71, 0: 59

# 다음과 같은 로지스틱회귀 모형을 사용한 분류모델을 만들고 결과를 확인
# 모델생성 시 상수항(=절편)을 포함하도록 하며 규제는 사용하지 않습니다.
# 종속변수: wine_variety
# 독립변수: alcohol, color_intensity, proline, flavanoids, malic_acid

# 1-2) GLM.from_formula()를 사용, 로지스틱 회귀모형 생성
# print(df.info())
from statsmodels.api import GLM, families
formula = 'wine_variety ~ alcohol + color_intensity + proline + flavanoids + malic_acid'
model = GLM.from_formula(formula, df).fit()
# print(model.summary())

# 1-3) 모델의 로그-우도를 반올림하여 소수점 아래 3자리까지 출력
# print(round(model.llf, 3)) # 15.602

# 1-4) 잔차이탈도를 반올림하여 소수점 아래 3자리까지 출력
# print(round(model.deviance, 3)) # 5.987

# 1-5) 'proline'을 독립변수로 하였을 때 오즈비는?
# 반올림하여 소수점 아래 3자리까지 출력
import numpy as np
# print(round(np.exp(model.params['proline']), 3)) # 0.999

# 1-6) 'proline'가 3 증가하면 오즈는 몇 % 감소 또는 증가하는가?
odds_ratio = np.exp(model.params['proline'] * 3) # 오즈비가 1보다 작음
decrease = round((1 - odds_ratio) * 100, 2)
# print(decrease) # 0.21

# 1-7) 아래의 sample을 사용하여 P(Y=1)에 대한 확률을 구하고,
# 반올림하여 소수점 아래 3자리까지 출력하시오
sample = {'alcohol': [13.5], 'color_intensity': [5.0], 
          'proline': [450],  'flavanoids': [2.8], 
          'malic_acid': [1.8]}
data = pd.DataFrame(sample)
# print(round(model.predict(data)[0], 3)) # 0.650

# 1-8) 위 샘플에 대한 odds를 구하고, 반올림하여 소수점 아래 4자리까지 출력하시오
p_y1 = model.predict(data)[0]
p_y0 = 1 - p_y1
odds = p_y1 / p_y0
# print(round(odds, 4)) # 1.8537

# 1-9) 유의수준 5% 하에서 유의성이 낮은 변수의 개수
res = model.pvalues[1:]<=0.05
# print(res.sum()) # 4

# 1-10) 아래 샘플에 대한 P(Y=1)에 대한 95% 신뢰구간의 상한은?
# 반올림하여 소수점 아래 4자리까지 출력
sample = {'alcohol': [13.5], 'color_intensity': [5.0], 
          'proline': [850],  'flavanoids': [2.8], 
          'malic_acid': [1.8]}
data = pd.DataFrame(sample)
result = model.get_prediction(data)
# print(result.summary_frame(alpha=0.5))
# print(round(result.conf_int(alpha=0.05)[0][1], 4)) # 0.4178

# 1-11) 정확도를 구하여 소수점 아래 3자리까지 출력
from sklearn.metrics import accuracy_score, f1_score
y_true = df['wine_variety']
y_pred = model.predict(df).round().astype('int32')
acc = accuracy_score(y_true, y_pred)
# print(round(acc, 3)) # 0.992

# 1-12) f1_score를 구해 반올림하여 소수점 아래 3자리까지 출력
result = f1_score(y_true, y_pred)
# print(round(result, 3)) # 0.993