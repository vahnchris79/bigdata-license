# 공통
path1 = "https://raw.githubusercontent.com/Soyoung-Yoon/bigdata/main/"
path2 = "https://raw.githubusercontent.com/Soyoung-Yoon/data_01/main/"
path3 = "https://raw.githubusercontent.com/Soyoung-Yoon/data_02/main/"

# 작업형 제1유형
import pandas as pd
# 9-1) 유병률 증가
# worlddata.csv 파일을 사용하여 다음의 사항을 처리합니다.
# 1999년과 2000년 간 유병률이 5배 이상 증가한 국가를 모두 선택합니다.
# 이 중 유병률 증가 비율이 가장 작은 하위 4개 국가의 유병률 증가 비율의 평균을 구해, 반올림하여 소수점 아래 세자리까지 출력합니다.
# 유병률 증가 비율 = 2000년_유병률 / 1999년_유병률
# 단, 1999년의 유병률이 0인 국가는 제외합니다.
df1 = pd.read_csv(path1 + "worlddata.csv")
df1 = df1.set_index('year').T
df1 = df1[df1[1999] !=0 ]
s = df1[2000] / df1[1999]
s = s[s >= 5]
result1 = round(s.sort_values().iloc[:4].mean(), 3)
# print(result1) # 6.458

# 9-2) 시급과 기본급
# salary01.csv를 사용하여 시급이 가장 높은 사람의 기본급을 정수로 구합니다.
# 단, 다음 컬럼들의 '-' 값을 0으로 변경하여 사용합니다.
# totalSalary, specialSalary, numberOfWorker : 정수
# workTime : 실수
# totalSalary 또는 workTime이 0인 데이터는 제외하고 기본급과 시급을 구합니다.
# 기본급 = totalSalary - specialSalary
# 시급 = 기본급 / workTime
df2 = pd.read_csv(path2 + "salary01.csv")
# print(df2.head(3), df2.info(), sep="\n")
for col in ['totalSalary','specialSalary','numberOfWorker','workTime']:
    df2[col] = df2[col].replace('-', 0)
df2 = df2[(df2['totalSalary'].astype(int) > 0) | (df2['workTime'].astype(float) > 0)]
df2['baseSalary'] = df2['totalSalary'].astype(int) - df2['specialSalary'].astype(int)
df2['hourSalary'] = df2['totalSalary'].astype(int) / df2['workTime'].astype(float)
result1 = int(df2[df2['hourSalary'] == df2['hourSalary'].max()]['baseSalary'].iloc[0])
# print(result1) # 4047478

# 작업형 제3유형
# 정규모집단으로부터 추출한 표본이 있다.
# 남자의 키 평균에 대한 일표본t-검정(one sample t-test)을 통해 답하고자 한다. 가설은 아래와 같다.
# 171보다 크거나 같음
# 171보다 작음
# 다음에 대해 유의수준 0.05로 일표본 t-검정한다.
# 성별(gender)은 남자=1, 여자=0의 값을 갖는다.
# 3개의 값을 1개 행에 공백을 구분자로 하여 출력한다.
# 위의 가설을 검정하기 위한 검정통계량을 입력하시오.(반올림하여 소수 넷째자리까지 계산)
# 위의 통계량에 대한 p-값을 구하여 입력하시오. (반올림하여 소수 넷째자리까지 계산)
# 유의수준 0.05 하에서 가설 검정의 결과를 (채택/기각) 중 하나를 선택하여 입력하시오.
import pandas as pd
import scipy.stats
s = pd.Series(dir(scipy.stats))
# print(s[s.str.contains('ttest')]) # ttest_1samp
df1 = pd.read_csv(path3 + "human_traits_sample.csv")
# print(df1.info())
from scipy.stats import ttest_1samp
male_height = df1[df1['gender']==1]['height_cm'].to_numpy()
statistic, pvalues = ttest_1samp(male_height, popmean=171, alternative="less")
result1 = round(statistic, 4)
result2 = round(pvalues, 4)
result3 = "기각" if pvalues <= 0.05 else "채택"
print(f"{result1} {result2} {result3}")
# -2.0328 0.0266 기각

# 2. 이표본 t-검정
# 정규모집단으로부터 추출한 표본이 있다. 
# 여자 그룹의 체중(weight_kg)이 남자 그룹의 체중보다 큰지 이표본 t-검정(two sample t-test)를 통해 답하고자 한다. 
# 가설은 아래와 같다.
# 여자그룹의 체중이 남자그룹의 체중보다 작거나 같다
# 여자그룹의 체중이 남자그룹의 체중보다 크다.
# 다음의 사항을 참고하여 답안을 작성하시오.
# 성별(gender)의 값은 여자=0, 남자=1로 되어 있다.
# 4개의 값을 1개 행에 공백을 구분자로 하여 출력한다.
# 등분산성 검정을 bartlett을 사용하여 수행한다.
# 두 그룹의 등분산 검정 결과를 (등분산/이분산) 중 하나를 선택하여 입력하시오.
# 위의 t-검정 관련 가설을 검정하기 위한 검정통계량을 구하시오. (반올림하여 소수점아래 셋째자리까지 계산)
# 위의 검정통계량에 대한 p-value를 구하라. (반올림하여 소수점아래 셋째자리까지 계산)
# 유의수준 0.05 하에서 가설 검정의 결과를 (채택/기각) 중 하나를 선택하여 입력하시오.
import pandas as pd
import scipy.stats
s = pd.Series(dir(scipy.stats))
from scipy.stats import ttest_ind
from scipy.stats import bartlett
# print(s[s.str.contains('ttest')]) # ttest_ind 
# print(s[s.str.contains('bar')])   # bartlett
df2 = pd.read_csv(path3 + 'human_traits_sample.csv')
# print(df2.head(3))
female_weight = df2[df2['gender']==0]['weight_kg'].to_numpy()
male_weight = df2[df2['gender']==1]['weight_kg'].to_numpy()
statistic, pvalue = bartlett(female_weight, male_weight)
result1 = "등분산" if pvalue > 0.05 else "이분산"
statistic, pvalue = ttest_ind(female_weight, male_weight, alternative="greater",equal_var=True)
result2 = round(statistic, 3)
result3 = round(pvalue, 3)
result4 = "기각" if pvalue <= 0.05 else "채택"
print(f"{result1} {result2} {result3} {result4}")
# 등분산 0.728 0.235 채택

# 3. Paired t-검정
# 정규모집단으로부터 추출한 표본이 있다. 교육 전/후의 시험 점수 평균에 대해 t검정을 실시한다.
# 학생 30명의 교육 전후의 점수가 저장되어 있다. 해당 교육이 효과가 있는지 (즉, 학습 후의 점수가 증가했는지) 쌍체표본 t-검정(paired t-test)를 통해 답하고자 한다. 가설은 아래와 같다
# 𝜇𝑑: (교육 후 점수 - 교육 전 점수)의 평균
# 𝐻0:  𝜇𝑑≤0  (교육 점수 감소 또는 효과 없음)
# 𝐻1:  𝜇𝑑>0 (교육 후 점수 증가)
# 다음에 대해 유의수준 0.05로 Paired t-검정한다.
# 4개의 값을 1개 행에 공백을 구분자로 하여 출력한다.
# 위의 가설을 검정하기 위한 검정통계량을 구하시오. (반올림하여 소수점아래 셋째자리까지 계산)
# 위의 검정통계량에 대한 p-value를 구하시오. (반올림하여 소수점아래 넷째자리까지 계산)
# 유의수준 0.05 하에서 가설검정의 결과를 (채택/기각) 중 하나를 선택하여 입력하시오.
# 가설 검정의 결과 교육 후 시험 점수 변화를 (증가/감소) 중 하나를 선택하여 입력하시오.
import pandas as pd
df3 = pd.read_csv(path3 + "paired_score.csv")
# print(df3.head(3))
score_diff = df3['score_after'] - df3['score_before']
import scipy.stats
s = pd.Series(dir(scipy.stats))
# print(s[s.str.contains('ttest')]) # ttest_rel
from scipy.stats import shapiro, ttest_rel
_, pvalue = shapiro(score_diff)
# print("기각" if pvalue > 0.05 else "채택") # 기각
score_before = df3['score_before'].to_numpy()
score_after = df3['score_after'].to_numpy()
statistic, pvalue = ttest_rel(score_after, score_before, alternative="greater")
result1 = round(statistic, 3)
result2 = round(pvalue, 4)
result3 = "기각 증가" if pvalue <= 0.05 else "채택 감소"
print(f"{result1} {result2} {result3}")
# 3.701 0.0004 기각 증가

# 4. paired t-test
# 주어진 데이터(data/blood_pressure.csv)에는 고혈압 환자 120명의 치료 전후의 혈압이 저장되어 있다. 
# 해당 치료가 효과가 있는지 (즉, 치료 후의 혈압이 감소했는지) 쌍체표본 t-검정(paired t-test)를 통해 답하고자 한다. 가설은 아래와 같다
# 𝜇𝑑  : (치료 후 혈압 - 치료 전 혈압)의 평균
# 𝐻0  :  𝜇𝑑≥0  (치료 효과 없음 또는 증가)
# 𝐻1  :  𝜇𝑑<0 (치료 후 혈압 감소)
# 𝑢𝑑 의 표본 평균을 구하시오 (반올림하여 소수 둘째자리까지 계산)
# 위의 가설을 검정하기 위한 검정통계량을 입력하시오.(반올림하여 소수 넷째자리까지 계산)
# 위의 통계량에 대한 p-값을 구하여 입력하시오. (반올림하여 소수 넷째자리까지 계산)
# 유의수준 0.05 하에서 가설검정의 결과를 (채택/기각) 중 하나를 선택하여 입력하시오.
import pandas as pd
df4 = pd.read_csv(path3 + "blood_pressure.csv")
# print(df4.head(3))
bp_before = df4['bp_before'].to_numpy()
bp_after = df4['bp_after'].to_numpy()
result1 = round((bp_after - bp_before).mean(), 2)
from scipy.stats import ttest_rel
statistic, pvalue = ttest_rel(bp_after, bp_before, alternative="less")
result2 = round(statistic, 3)
result3 = round(pvalue, 4)
result4 = "기각" if pvalue <= 0.05 else "채택"
print(f"{result1} {result2} {result3} {result4}")
# -5.09 -3.337 0.0006 기각

# 작업형 제2유형 제9회 기출복원
path = "https://raw.githubusercontent.com/YoungjinBD/data/main/exam/"
# 학습용 데이터: 9_2_train.csv, 평가용 데이터: 9_2_test.csv
# 지역의 농업유형 예측하는 모델 개발
# 모델평가지표: Macro, F1 Score
# 라이브러리
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score as F1

# 데이터 확인
train = pd.read_csv(path + "9_2_train.csv")
test = pd.read_csv(path + "9_2_test.csv")
# print(train.info(), test.info(), sep="\n") # 1680, 720
# print(train['등급'].unique()) # ['B' 'C' 'A']

# 데이터 전처리
X_all = pd.concat([train, test]).drop(columns=['ID','라벨'])
Y = train['라벨']
X_all['지역'] = LabelEncoder().fit_transform(X_all['지역'])
X_all = pd.get_dummies(X_all, drop_first=True, dtype='int32')

# 데이터 재분할
X = X_all.iloc[:len(train), :]
X_submission = X_all.iloc[len(train):, :]
# print(X.shape, X_submission.shape) # (1680, 5) (720, 5)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=42)
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (1176, 5) (504, 5) (1176,) (504,)

# 모델사전
models = {
    "Logistic": Pipeline([
        ("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=100, tol=0.05, random_state=42))
    ]),
    "DecisionTree": Pipeline([
        ("model", DecisionTreeClassifier(max_depth=20, random_state=42))
    ]),
    "RandomForest": Pipeline([
        ("model", RandomForestClassifier(max_depth=20, random_state=42))
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
    F1_train = F1(y_train, y_pred1, average="macro")
    F1_test = F1(y_test, y_pred2, average="macro")
    return model, F1_train, F1_test

# 모델별 성능평가
results = []
for name, model in models.items():
    model, F1_train, F1_test = get_scores(model, x_train, x_test, y_train, y_test)
    results.append({
        "Model": name, "F1_train": f"{F1_train:.4f}", "F1_test": f"{F1_test:.4f}"
    })
res = pd.DataFrame(results).sort_values("F1_test", ascending=False).reset_index(drop=True)
# print(res) # DecisionTree 0.9837 0.3390

# 모델적합 및 예측
model = models[res.loc[0, "Model"]]
y_pred = model.predict(X_submission)

# 제출파일 생성
# pd.DataFrame({'pred': y_pred}).to_csv("result_9th_type2.csv", index=False)

# 결고확인
# temp = pd.read_csv("result_9th_type2.csv")
# print(Y[:len(test)].value_counts())
# print(Y[:len(test)].value_counts(normalize=True))
# print("=" * 35)
# print(temp['pred'].value_counts())
# print(temp['pred'].value_counts(normalize=True))
