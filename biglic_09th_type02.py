
# 작업형 제2유형

# 제공된 학습용데이터(9_2_train.csv)는 지역의 특성과 해당지역의 농업유형정보를 포함하고 있다.
# 학습용 데이터를 활용하여 지역의 농업 유형(라벨)을 예측하는 다중분류 모델을 개발하고,
# 가장 우수한 모델을 평가용 데이터(9_2_test.csv)에 적용하여 예측결과를 제출하시오.

import pandas as pd

path = "https://raw.githubusercontent.com/YoungjinBD/data/main/exam/"

XY = pd.read_csv(path + "9_2_train.csv")
X_submission = pd.read_csv(path + "9_2_test.csv")
# print(XY.head())
# print(X_submission.head())
# print(XY.shape, X_submission.shape) # (1680, 6) (720, 6)
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score

pd.set_option('display.width', 100)

# 평가지표 함수
def get_scores(model, x_train, x_test, y_train, y_test):
    A = model.score(x_train, y_train)
    B = model.score(x_test, y_test)
    y_pred1 = model.predict(x_train)
    y_pred2 = model.predict(x_test)
    C = f1_score(y_train, y_pred1, average="macro")
    D = f1_score(y_test, y_pred2, average="macro")
    return f"acc: {A:.4f}, {B:.4f}, F1: {C:.4f}, {D:.4f}"

# XY를 X, Y로 분리하기
X = XY.drop(columns=['ID','라벨'])
Y = XY['라벨']
X_submission = X_submission.drop(columns=['ID','라벨'])

# Y의 분포확인
# print(Y.value_counts(normalize=True)) # 0: 0.875, 1: 0.083, 2: 0.0417

# X와 X_submission을 합치고 EDS 실시
X_all = pd.concat([X, X_submission])
# print(X_all.head())
# print(X_all.info())
# print(X_all.shape) # (2400, 5)
# print(X_all.isna().sum().T) # 결측치 없음

# 전처리: 범주형을 수치형으로 변환
# 범주형 컬럼: 지역, 등급
X_all['지역'] = LabelEncoder().fit_transform(X_all['지역'])
X_all['등급'] = LabelEncoder().fit_transform(X_all['등급'])
# print(X_all.head())

# ID, 지역, 등급, 농업면적, 연도 컬럼의 단위표준화를 위한 스케일링
temp = StandardScaler().fit_transform(X_all)
X_all = pd.DataFrame(temp, columns=X_all.columns)
X = X_all.iloc[:1680]
X_submission = X_all.iloc[1680:]
# print(X.shape, Y.shape, X_submission.shape) # (1680, 5) (1680,) (720, 5)

# 데이터 분할
temp = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=1234)
x_train, x_test, y_train, y_test = temp
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (1176, 5) (504, 5) (1176,) (504,)

# 모델적합
# LogisticRegression
model1 = LogisticRegression().fit(x_train, y_train)
# print(get_scores(model1, x_train, x_test, y_train, y_test)) # acc: 0.8750, 0.8750, F1: 0.3111, 0.3111

# DecisionTreeClassifier
# model2 = DecisionTreeClassifier(random_state=1234).fit(x_train, y_train)
# print(get_scores(model2, x_train, x_test, y_train, y_test)) # acc: 1.0000, 0.7440, F1: 1.0000, 0.3210
# print(model2.get_depth()) # 29
# for d in range(3, 14):
#     model2 = DecisionTreeClassifier(max_depth=d, random_state=1234).fit(x_train, y_train)
#     print(d, get_scores(model2, x_train, x_test, y_train, y_test)) # 3 acc: 0.8801, 0.8770, F1: 0.3601, 0.3539
model2 = DecisionTreeClassifier(max_depth=3, random_state=1234).fit(x_train, y_train)
# print(get_scores(model2, x_train, x_test, y_train, y_test)) # acc: 0.8801, 0.8770, F1: 0.3601, 0.3539

# RandomForestClassifier
# model3 = RandomForestClassifier(random_state=1234).fit(x_train, y_train)
# print(get_scores(model3, x_train, x_test, y_train, y_test)) # acc: 1.0000, 0.8690, F1: 1.0000, 0.3100
# for d in range(3, 14):
#     model3 = RandomForestClassifier(max_depth=d, random_state=1234).fit(x_train, y_train)
#     print(d, get_scores(model3, x_train, x_test, y_train, y_test)) # 4 acc: 0.8750, 0.8750, F1: 0.3111, 0.3111
model3 = RandomForestClassifier(max_depth=4, random_state=1234).fit(x_train, y_train)
# print(get_scores(model3, x_train, x_test, y_train, y_test)) # acc: 0.8750, 0.8750, F1: 0.3111, 0.3111

# AdaBoostClassifier
model4 = AdaBoostClassifier(random_state=0).fit(x_train, y_train)
# print(get_scores(model4, x_train, x_test, y_train, y_test)) # acc: 0.8750, 0.8750, F1: 0.3111, 0.3111

# GradientBoostingClassifier
model5 = GradientBoostingClassifier(random_state=0).fit(x_train, y_train)
# print(get_scores(model5, x_train, x_test, y_train, y_test)) # acc: 0.9150, 0.8710, F1: 0.6382, 0.3392

final_model = model5
y_pred = final_model.predict(X_submission)
pd.DataFrame({'pred': y_pred}).to_csv("result_09th_type2.csv", index=False)

temp = pd.read_csv("result_09th_type2.csv")
print(temp['pred'].value_counts(normalize=True)) # 0: 0.971, 1: 0.021, 2: 0.008