
# 공통: 데이터 경로
import pandas as pd
path1 = "https://raw.githubusercontent.com/Soyoung-Yoon/bigdata/main/"
path2 = "https://raw.githubusercontent.com/Soyoung-Yoon/data_01/main/"
path3 = "https://raw.githubusercontent.com/Soyoung-Yoon/data_02/main/"

# 작업형 제1유형
# 6회-1번) 소방서별 출동 소요시간 분석
# 소방서는 신고를 접수한 후 현장으로 출동한다, 이때 신고 후 출동까지 걸린 시간을 분석하고자 한다.
# 다음 절차에 따라 문제를 해결하시오.
df = pd.read_csv(path1 + "big_6_1_1.csv")
# print(df.head(3))
# 1) 각 사건에 대한 '출동소요시간'을 구하여 컬럼을 추가하시오.
#    > 출동소요시간 = (출동시간 - 신고시간)
for col in df.columns:
    if '시간' in col:
        df[col] = df[col].astype('datetime64[ns]')
df['출동소요시간'] = df['출동시간'] - df['신고시간']
# print(df.head(3))
# 2) 1)의 결과를 사용하여 소방서(소방서명)별 '출동소요시간'의 평균을 구하시오.
s = df.groupby('소방서명')['출동소요시간'].mean()
# print(s)
# 3) 평균 출동소요시간 중 가장 큰 값을 구해, 반올림하여 분 단위로 변환하고, 그 값을 반올림하여 정수로 출력한다.
# print(round(s.max().total_seconds() / 60)) # 3

# 6회-2번) 교사 1인당 학생 수
# 초등학교에서는 학년별 학생 수와 교사 수를 관리하고 있다.
# 학교별로 교사 1명이 맡고 있는 평균 학생 수를 계산하여, 관리가 필요한 학교를 파악하고자 한다.
# 다음 절차에 따라 문제를 해결하시오.
df = pd.read_csv(path1 + "big_6_1_2.csv")
# print(df.head(3))
# 1) 학교별 전체_학생수 및 전체_교사수를 구하시오.
s = df.groupby('학교명', as_index=False)[['학생수', '교사수']].sum()
s.columns = ['학교명', '전체_학생수', '전체_교사수']
# print(s)
# 2) 교사 1인당 학생 수를 계산하시오.
#    > 교사당_학생수 =  전체 학생수 / 전체 교사수
s['교사당학생수'] = s['전체_학생수'] / s['전체_교사수']
# print(s)
# 3) 교사 1인당 학생 수가 가장 많은 학교를 찾고, 해당 학교의 전체 교사 수를 정수로 제출합니다.
s = s.set_index('학교명')
# print(s[s.index == s['교사당학생수'].idxmax()]['전체_교사수'].sum()) # 6

# 6회-3번) 연도별 총 범죄 건수의 월평균 비교
# 범죄 데이터를 분석하여, 연도별로 범죄 발생량의 특징을 파악하고자 한다.
df = pd.read_csv(path1 + "big_6_1_3.csv")
# print(df.head(3))
# 다음 절차에 따라 문제를 해결하시오.
# 1) 각 연도별로 범죄유형별 발생 건수를 모두 합산하여, 연도별 총 범죄 건수를 계산한다.
df['연도'] = df['월별'].str[:4]
df['총범죄건수'] = df.loc[:, '강력범죄':'경범죄'].sum(axis=1)
s = df.groupby('연도')['총범죄건수'].sum().reset_index(name='연도별총범죄건수')
# print(s)
# 2) 1)에서 구한 연도별 총 범죄 건수를 12(개월)로 나누어 월평균 범죄 건수를 구한다.
month = df.groupby('연도')['월별'].nunique().reset_index(name='월수')
s = s.merge(month, "left", "연도")
s['월평균범죄건수'] = s['연도별총범죄건수'] / s['월수']
# print(s)
# 3) 2)의 결과를 사용해 월평균 범죄 건수가 가장 큰 연도를 찾고,
# 해당 연도의 월평균 총 범죄 건수를 반올림하여 정수로 출력한다.
s = s.set_index('연도')
# print(round(s[s.index == s['월평균범죄건수'].idxmax()]['월평균범죄건수'].iloc[0])) # 4329

# 14-4) 시간대별 평균 출동 소요시간 비교
# 소방서는 하루 동안 다양한 시간대에 신고를 접수하고 현장에 출동한다.
# 시간대별로 평균 출동 소요시간을 분석하여, 대응 효율성을 진단하고자 한다.
df = pd.read_csv(path2 + "fire_station02.csv")
# print(df.head(3))
# 다음 절차에 따라 문제를 해결하시오.
# 1) 출동 소요시간은 (출동시간 - 신고시간)으로 계산한다.
for col in df.columns:
    if '시간' in col:
        df[col] = df[col].astype('datetime64[ns]')
df['출동소요시간'] = df['출동시간'] - df['신고시간']
# print(df.head(3))
# 2)  신고시간을 기준으로 다음과 같이 시간대를 분류한다:
#   - 새벽: 00:00 ~ 05:59, 오전: 06:00 ~ 11:59, 오후: 12:00 ~ 17:59, 밤: 18:00 ~ 23:59
df.loc[df['신고시간'].dt.hour < 6, '시간대'] = '새벽'
df.loc[(df['신고시간'].dt.hour >= 6) & (df['신고시간'].dt.hour < 12), '시간대'] = '오전'
df.loc[(df['신고시간'].dt.hour >= 12) & (df['신고시간'].dt.hour < 18), '시간대'] = '오후'
df.loc[df['신고시간'].dt.hour >= 18, '시간대'] = '밤'
# print(df.head(3))
# 3) 각 시간대별 출동 소요시간의 평균을 구한다.
s = df.groupby('시간대')['출동소요시간'].mean()
# print(s)
# 4) 이 중 출동 소요시간 평균이 가장 큰 시간대의 출동 소요시간 평균을 분 단위로 변환하고,
#    그 값을 반올림하여 정수값을 제출하시오.
# print(round(s[s.idxmax()].total_seconds() / 60)) # 7

# 14-5) 계절별 범죄 유형 평균 분석
# 범죄 발생 패턴이 계절에 따라 달라질 수 있다는 점에 착안하여, 월별 범죄 데이터를 계절별로 재구성하고 평균을 분석하여, 
# 계절 간 범죄 발생량 차이가 가장 큰 범죄 유형 및 차이값을 찾고자 한다.
df = pd.read_csv(path1 + 'big_6_1_3.csv')
# print(df.head(3))
# 다음 절차에 따라 문제를 해결하시오.
# 1) '월'값을 기반으로 '계절'을 구분한다.
#    봄 : 3, 4, 5월, 여름 : 6, 7, 8월, 가을 : 9, 10, 11월, 겨울 : 12, 1, 2월
df['월별'] = df['월별'].str[-2:].astype('int')
df['계절'] = df['월별'].map(lambda x: '봄' if x in [3,4,5] else \
                        '여름' if x in [6,7,8] else '가을' if x in [9,10,11] else \
                        '겨울')
# 2) 각 범죄 유형별로 계절별 평균을 계산하시오.
df2 = df.drop(columns="월별").groupby('계절').mean()
# print(df2)
# 3) 각 범죄 유형별로 4개 계절의 평균값 중 최대값과 최소값을 구하고, 그 차이(최대−최소)를 계산하시오.
# 4) 위에서 구한 값들 중 가장 큰 값을 반올림하여 정수로 출력하시오.
# print(round((df2.max() - df2.min()).max())) # 156