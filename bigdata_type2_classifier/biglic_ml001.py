
# Kaggle Adult income(train, test) dataset
# 이 데이터는 Barry Becker가 1994년 인구조사 데이터베이스에서 추출했습니다.
# 다음 조건들을 만족하는 비교적 깨끗한 레코드 집합을 뽑았습니다:
# ((AAGE > 16) && (AGI > 100) && (AFNLWGT > 1) && (HRSWK > 0))

# 데이터는 미국 인구조사국 데이터베이스에서 추출되었으며, 원본은 아래 링크에서 확인할 수 있습니다:
# http://www.census.gov/ftp/pub/DES/www/welcome.html
# 기부자: Ronny Kohavi, Barry Becker (Silicon Graphics, Data Mining and Visualization)
# 문의: ronnyk@sgi.com

# 데이터는 MLC++의 GenCVFiles를 사용해 학습(train)과 테스트(test) 세트로 무작위 분할되었습니다.
# (2/3은 학습, 1/3은 테스트).
# 전체 인스턴스 수: 48,842개 (학습: 32,561개, 테스트: 16,281개)
# 결측값(unknown) 제거 시: 45,222개 (학습: 30,162개, 테스트: 15,060개)
# 중복 또는 충돌하는 인스턴스: 6개
# 클래스 분포 (adult.all 파일 기준):
# >50K 레이블의 확률: 23.93% / (결측 제거 시 24.78%)
# <=50K 레이블의 확률: 76.07% / (결측 제거 시 75.22%)

# 라이브러리 import
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
pd.set_option('display.max_rows', 100)

# 평가함수(roc_auc_score 사용시)
def get_scores(model, x_train, x_test, y_train, y_test):
    A = model.score(x_train, y_train)
    B = model.score(x_test, y_test)
    y_proba1 = model.predict_proba(x_train)[:, 1]
    y_proba2 = model.predict_proba(x_test)[:, 1]
    C = roc_auc_score(y_train, y_proba1)
    D = roc_auc_score(y_test, y_proba2)
    return f"acc: {A:.4f}, {B:.4f}, AUC: {C:.4f}, {D:.4f}"

# 데이터 불러오기
path = "dataset/adult_income/"
train_csv = path + "adult_train.csv"
test_csv = path + "adult_test.csv"
XY = pd.read_csv(train_csv, na_values="?", skipinitialspace=True)
X_submission = pd.read_csv(test_csv, na_values="?", skipinitialspace=True)
# print(XY.shape, X_submission.shape)
# print(XY.head(100), X_submission.head(100))

# 데이터 탐색
# 독립변수와 종속변수로 나누어 X, Y로 저장
X = XY.drop(columns=['Income'])
Y = XY['Income']
X_submission = X_submission.drop(columns=['Income'])
# print(X.shape, Y.shape) # (30162, 13) (30162,)

# Y의 타입과 값별 수량을 확인
# print(Y.dtypes)
# print(Y.value_counts()) # <=50K: 24720, >50K: 7841

# 데이터 전처리
# X + X_submission 합치기: X_all
# 결측치 확인 후 제거, Encoding, Scaliing
X_all = pd.concat([X, X_submission])
# print(X_all.shape) # (48842, 14)

# 결측치 확인
# print(X_all.isna().sum().T) # age 2799, occupation 2809, natvie-country 857
# 가장 많은 빈도값으로 결측치 대체
X_all['age'] = X_all['age'].fillna('mode')
X_all['occupation'] = X_all['occupation'].fillna('mode')
X_all['native-country'] = X_all['native-country'].fillna('mode')
# print(X_all.isna().sum().sum())

# 컬럼별 범주형 확인 -> Encoding
columns_obj = X_all.select_dtypes(include='object').columns
# print(columns_obj.to_numpy())
# ['age', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
# print(X_all[columns_obj].nunique())
X_all['age'] = LabelEncoder().fit_transform(X_all['age'])
X_all['education'] = LabelEncoder().fit_transform(X_all['education'])
X_all['marital-status'] = LabelEncoder().fit_transform(X_all['marital-status'])
X_all['occupation'] = LabelEncoder().fit_transform(X_all['occupation'])
X_all['relationship'] = LabelEncoder().fit_transform(X_all['relationship'])
X_all['race'] = LabelEncoder().fit_transform(X_all['race'])
X_all['sex'] = LabelEncoder().fit_transform(X_all['sex'])
X_all['native-country'] = LabelEncoder().fit_transform(X_all['native-country'])
# print(X_all.head(3))

# Scaling, X, X_submission 분리
scaler = MinMaxScaler().fit_transform(X_all)
X_all = pd.DataFrame(scaler, columns=X_all.columns)
# print(X_all.head(3))
X = X_all.iloc[:32561, ]
X_submission = X_all.iloc[32561:, ]
# print(X.shape, X_submission.shape)

# 모델링
# 데이터 분할
temp = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=1234)
x_train, x_test, y_train, y_test = temp
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (22792, 13) (9769, 13) (22792,) (9769,)

model1 = LogisticRegression().fit(x_train, y_train)
# print(get_scores(model1, x_train, x_test, y_train, y_test)) # acc: 0.8218, 0.8242, AUC: 0.8408, 0.8384

# model2 = DecisionTreeClassifier().fit(x_train, y_train)
# print(get_scores(model2, x_train, x_test, y_train, y_test)) # acc: 0.9998, 0.8043, AUC: 1.0000, 0.7353
# print(model2.get_depth()) # 62
# for d in range(3, 21):
#     model2 = DecisionTreeClassifier(max_depth=d, random_state=1342).fit(x_train, y_train)
#     print(d, get_scores(model2, x_train, x_test, y_train, y_test)) # 4 acc: 0.8475, 0.8460, AUC: 0.8665, 0.8632
model2 = DecisionTreeClassifier(max_depth=4, random_state=1342).fit(x_train, y_train)
# print(get_scores(model2, x_train, x_test, y_train, y_test)) # acc: 0.8475, 0.8460, AUC: 0.8665, 0.8632

# model3 = RandomForestClassifier(n_estimators=100, random_state=1342).fit(x_train, y_train)
# print(get_scores(model3, x_train, x_test, y_train, y_test)) # acc: 0.9997, 0.8418, AUC: 1.0000, 0.8864
# for d in range(3, 21):
#     model3 = RandomForestClassifier(n_estimators=100, max_depth=d, random_state=1234).fit(x_train, y_train)
#     print(d, get_scores(model3, x_train, x_test, y_train, y_test)) # 3 acc: 0.8347, 0.8352, AUC: 0.8899, 0.8872
model3 = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=1234).fit(x_train, y_train)
# print(get_scores(model3, x_train, x_test, y_train, y_test)) # acc: 0.8347, 0.8352, AUC: 0.8899, 0.8872

model4 = AdaBoostClassifier(random_state=1234).fit(x_train, y_train)
# print(get_scores(model4, x_train, x_test, y_train, y_test)) # acc: 0.8472, 0.8472, AUC: 0.8933, 0.8916

model5 = GradientBoostingClassifier(random_state=1234).fit(x_train, y_train)
# print(get_scores(model5, x_train, x_test, y_train, y_test)) # acc: 0.8657, 0.8606, AUC: 0.9191, 0.9123

# 예측값 구하고 파일로 저장
final_model = model3
y_pred = final_model.predict(X_submission)
pd.DataFrame({'pred': y_pred}).to_csv("result_adult_income.csv", index=False)

# 저장결과 확인, 예측값 비율확인
temp = pd.read_csv("result_adult_income.csv")
print(temp.shape) # (16281, 1)
print(temp['pred'].value_counts(normalize=True)) # <=50K: 0.896935, >50K: 0.103065
print(Y.value_counts(normalize=True)) # <=50K: 0.75919, >50K: 0.24081