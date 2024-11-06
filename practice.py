# 데이터1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from matplotlib import font_manager, rc
from scipy import stats
from scipy.stats import chi2_contingency


# ------------------------------------------------- df : 원본 데이터
# 데이터 불러오기
B1 = pd.read_csv('C:/Users/USER/Documents/LS 빅데이터 스쿨/project5/중합 CPS-8 DB B1.csv')
B2 = pd.read_csv('C:/Users/USER/Documents/LS 빅데이터 스쿨/project5/중합 CPS-8 DB B2.csv')


B1['Timestamp'] = pd.to_datetime(B1['Timestamp'])
B2['Timestamp'] = pd.to_datetime(B2['Timestamp'])


# B1 확인
B1.head()
B2.head()

B1.describe()
B2.describe()

# 자료형 확인
B1.info()
B2.info()


# 자료 행 수 확인
len(B1) # 44641
len(B2) # 43201

# 자료 날짜별 정렬
B1 = B1.sort_values('Timestamp')
B2 = B2.sort_values('Timestamp')


# 합병합고 싶은데 행 수가 달라서, 뭐가 다른지 확인 => 2018-11-09 제거
B1['Timestamp'].unique().isin(B2['Timestamp'].unique()).sum()  # 43201가 동일하게 있음
B1['Timestamp'].unique()[~B1['Timestamp'].unique().isin(B2['Timestamp'].unique())]
len(B1['Timestamp'].unique()[~B1['Timestamp'].unique().isin(B2['Timestamp'].unique())])  # 1439개 

B2['Timestamp'].unique()[~B2['Timestamp'].unique().isin(B1['Timestamp'].unique())]  # 없음

B1[B1['Timestamp']< '2018-11-10 00:05:00']



# B1 2018-11-09 제거
B1_drop1 = B1[B1['Timestamp']>='2018-11-10']

len(B1[B1['Timestamp']>='2018-11-10']) # 43202
len(B2)  # 43201



# 중복행 확인
B1_drop1[B1_drop1.duplicated()]  # 중복행 있음
B2[B2.duplicated()]  # 중복행 없음

B1_drop1.head()  # 인덱스 0, 44640 중복 행임

B1_drop2 = B1_drop1.drop(index=44640)
len(B1_drop2)  # 43201
len(B2)  # 43201


# 중복 변수 확인
B1_drop2.columns[B1_drop2.columns.isin(B2.columns)]  # ['Timestamp']가 중복됨
B2.columns[B2.columns.isin(B1_drop2.columns)]  # ['Timestamp']가 중복됨

len(B1.columns)  # 199
len(B2.columns)  # 213


B1_drop2_T = B1_drop2.transpose()
B1_drop2_T[B1_drop2_T.duplicated()]  # 중복열 있음 : 8FIC_P4260A.PV - Average, 8FIC_P4230B.PV - Average, 8FIC_F4270B.PV - Average, 8TIC_E9340.PV - Average.1
                                     #              ['8FIC_P4260B.PV - Average', '8FIC_P4230B.PV - Average', '8FIC_F4270B.PV - Average', '8TIC_E9340.PV - Average.1'] 제거


B1_drop2[['8FIC_P4260A.PV - Average', '8FIC_P4260B.PV - Average', '8FIC_P4230A.PV - Average', '8FIC_P4230B.PV - Average', '8TIC_E9340.PV - Average', '8TIC_E9340.PV - Average.1', '8FIC_F4270A.PV - Average','8FIC_F4270B.PV - Average']]


B2_T = B2.transpose()
B2_T[B2_T.duplicated()]  # 중복열 없음

B1_drop3 = B1_drop2.drop(columns=['8FIC_P4260B.PV - Average', '8FIC_P4230B.PV - Average', '8FIC_F4270B.PV - Average', '8TIC_E9340.PV - Average.1'])

B1_drop3_T = B1_drop3.transpose()
B1_drop3_T[B1_drop3_T.duplicated()]  # 더이상 중복열 없는거 확인


# 두 데이터 합치기
B = pd.merge(B1_drop3, B2, on='Timestamp')

B.info()
B.describe()

B.columns
B.shape


# 컬럼명 바꾸기
B.columns = B.columns.str.replace(' - Average','')

print(B.columns.to_list())


# 고유값 1개 확인
for i in B.columns:
    print(f"{i}컬럼의 고유값 개수 :",len(B[i].unique()))

# 고유값 1개 : 8PI_P4251A.PV, 8FI_D8502.PV, 8FI_F4270B.PV, LAB_8CHIP_TIO2, LAB_8CHIP_HEAT
# 고유값 1개인 변수 제거
B_drop1 = B.drop(columns=['8PI_P4251A.PV', '8FI_D8502.PV', '8FI_F4270B.PV', 'LAB_8CHIP_TIO2', 'LAB_8CHIP_HEAT'])

for i in B2.columns:
    if len(B2[i].unique()) == 1:
        print(f"{i}컬럼의 고유값 개수 :",len(B2[i].unique()))


# 확인
B1[['8PI_P4251A.PV - Average','8FI_D8502.PV - Average','8FI_F4270B.PV - Average']]
B2[['LAB_8CHIP_TIO2 - Average','LAB_8CHIP_HEAT - Average']]

B[['8PI_P4251A.PV', '8FI_D8502.PV', '8FI_F4270B.PV', 'LAB_8CHIP_TIO2', 'LAB_8CHIP_HEAT']]



B_drop1.shape

len(B_drop1.columns) # 402개

# 고유값이 15개 이하인 컬럼의 고유값 확인
for i in B.columns:
    if len(B[i].unique()) <=15:
        print(f"{i}컬럼의 고유값 : ", B[i].unique())

for i in B_drop1.columns:
    if len(B[i].unique()) <=15:
        print(f"{i}컬럼의 고유값 개수 :",len(B[i].unique()))


# 날짜 컬럼을 인덱스로 바꾸기 : 교차 상관분석 하기전에 해야함
B_index = B_drop1.set_index('Timestamp')



# ----------------------------------------- 시각화
B_tindex = B_drop1.reset_index()
B_tindex['dayofyear'] = B_tindex['Timestamp'].dt.dayofyear
B_tindex['dayofyear'].info()

B_tindex.columns

info = dict(df=B_tindex, x_time='Timestamp', y='8PIC_D4210.PV', x_time_term = None, x_time_start = 'W')
plt.figure(figsize=(7,4))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.lineplot(data=info['df'], x=info['x_time'], y=info['y'])
plt.axhline(info['df'][info['y']].mean(), color='red', linestyle='--', label='평균값') 
unique_dates = info['df'][info['x_time']].unique()
if (pd.api.types.is_numeric_dtype(info['df'][info['x_time']])) & (info['x_time_term'] != None):
    plt.xticks(np.arange(unique_dates.min(), unique_dates.max()+1, step=info['x_time_term']).astype(int))
elif (pd.api.types.is_datetime64_any_dtype(info['df'][info['x_time']])) & (info['x_time_start'] != None):  
    if info['x_time_start'] == 'W':
        ticks_term = pd.date_range(start=info['df'][info['x_time']].min(), end=info['df'][info['x_time']].max(), freq=info['x_time_start'])
        plt.xticks(ticks_term, [tick.strftime('%m-%d') for tick in ticks_term])  # 월-일 형식으로 설정
    else:
        ticks_term = pd.date_range(start=info['df'][info['x_time']].min(), end=info['df'][info['x_time']].max(), freq=info['x_time_start'])
        plt.xticks(ticks_term, ticks_term.strftime('%Y-%m-%d'))
plt.title(f'날짜별 {info['y']}', fontsize=15)
std_dev = np.round(info['df'][info['y']].std(),3)
plt.xlabel(f'{info['x_time']}  [ {info['y']}의 std : {std_dev} ]')
plt.tight_layout()  
plt.show()



B_tindex[B_tindex['Timestamp']<'2018-11-14']
info = dict(df=B_tindex[B_tindex['Timestamp']<'2018-11-14'], x_time='Timestamp', y='8PIC_D4210.PV', x_time_term = None, x_time_start = 'W',window=1440)
plt.figure(figsize=(7,4))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
df_copy = info['df'].copy()
df_copy['moving_avg'] = info['df'][info['y']].rolling(window=info['window']).mean()
plt.subplot(1,2,1)
sns.lineplot(data=info['df'], x=info['x_time'], y=info['y'])
sns.lineplot(data=df_copy, x=info['x_time'], y='moving_avg', label=f'{info['window']} 이동 평균', color='orange')
plt.axhline(info['df'][info['y']].mean(), color='red', linestyle='--', label='평균값') 
unique_dates = info['df'][info['x_time']].unique()
if (pd.api.types.is_numeric_dtype(info['df'][info['x_time']])) & (info['x_time_term'] != None):
    plt.xticks(np.arange(unique_dates.min(), unique_dates.max()+1, step=info['x_time_term']).astype(int))
elif (pd.api.types.is_datetime64_any_dtype(info['df'][info['x_time']])) & (info['x_time_start'] != None):  
    if info['x_time_start'] == 'W':
        ticks_term = pd.date_range(start=info['df'][info['x_time']].min(), end=info['df'][info['x_time']].max(), freq=info['x_time_start'])
        plt.xticks(ticks_term, [tick.strftime('%m-%d') for tick in ticks_term])  # 월-일 형식으로 설정
    else:
        ticks_term = pd.date_range(start=info['df'][info['x_time']].min(), end=info['df'][info['x_time']].max(), freq=info['x_time_start'])
        plt.xticks(ticks_term, ticks_term.strftime('%Y-%m-%d'))
plt.title(f'날짜별 {info['y']}', fontsize=15)
std_dev = np.round(info['df'][info['y']].std(),3)
plt.xlabel(f'{info['x_time']}  [ {info['y']}의 std : {std_dev} ]')

if info['window'] != None:
    plt.subplot(1,2,2)
    sns.lineplot(data=df_copy, x=info['x_time'], y='moving_avg', label=f'{info['window']} 이동 평균', color='orange')
    plt.axhline(info['df'][info['y']].mean(), color='red', linestyle='--', label='평균값') 
    if (pd.api.types.is_numeric_dtype(info['df'][info['x_time']])) & (info['x_time_term'] != None):
        plt.xticks(np.arange(unique_dates.min(), unique_dates.max()+1, step=info['x_time_term']).astype(int))
    elif (pd.api.types.is_datetime64_any_dtype(info['df'][info['x_time']])) & (info['x_time_start'] != None):  
        if info['x_time_start'] == 'W':
            ticks_term = pd.date_range(start=info['df'][info['x_time']].min(), end=info['df'][info['x_time']].max(), freq=info['x_time_start'])
            plt.xticks(ticks_term, [tick.strftime('%m-%d') for tick in ticks_term])  # 월-일 형식으로 설정
        else:
            ticks_term = pd.date_range(start=info['df'][info['x_time']].min(), end=info['df'][info['x_time']].max(), freq=info['x_time_start'])
            plt.xticks(ticks_term, ticks_term.strftime('%Y-%m-%d'))
plt.tight_layout()  
plt.show()




info = dict(df=B_tindex, x_time='dayofyear', y='8PIC_D4210.PV', x_time_term = 7, x_time_start = None)
plt.figure(figsize=(7,4))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.lineplot(data=info['df'], x=info['x_time'], y=info['y'])
plt.axhline(info['df'][info['y']].mean(), color='red', linestyle='--', label='평균값') 
unique_dates = info['df'][info['x_time']].unique()
if (pd.api.types.is_numeric_dtype(info['df'][info['x_time']])) & (info['x_time_term'] != None):
    plt.xticks(np.arange(unique_dates.min(), unique_dates.max()+1, step=info['x_time_term']).astype(int))
elif (pd.api.types.is_datetime64_any_dtype(info['df'][info['x_time']])) & (info['x_time_start'] != None):  
    if info['x_time_start'] == 'W':
        ticks_term = pd.date_range(start=info['df'][info['x_time']].min(), end=info['df'][info['x_time']].max(), freq=info['x_time_start'])
        plt.xticks(ticks_term, [tick.strftime('%m-%d') for tick in ticks_term])  # 월-일 형식으로 설정
    else:
        ticks_term = pd.date_range(start=info['df'][info['x_time']].min(), end=info['df'][info['x_time']].max(), freq=info['x_time_start'])
        plt.xticks(ticks_term, ticks_term.strftime('%Y-%m-%d'))
plt.title(f'날짜별 {info['y']}', fontsize=15)
std_dev = np.round(info['df'][info['y']].std(),3)
plt.xlabel(f'{info['x_time']}  [ {info['y']}의 std : {std_dev} ]')
plt.tight_layout()  
plt.show()



# 


#타켓변수
info = dict(df=B_drop1, x_time='Timestamp', y='LAB_8CHIP_L', x_time_term = None, x_time_start = 'W')
plt.figure(figsize=(7,4))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.lineplot(data=info['df'], x=info['x_time'], y=info['y'])
plt.axvline(pd.to_datetime('2018-12-08 07:00:00'), color='red', linestyle='--', label='12-08 이상 발생')
plt.legend()
unique_dates = info['df'][info['x_time']].unique()
if (pd.api.types.is_numeric_dtype(info['df'][info['x_time']])) & (info['x_time_term'] != None):
    plt.xticks(np.arange(unique_dates.min(), unique_dates.max()+1, step=info['x_time_term']).astype(int))
elif (pd.api.types.is_datetime64_any_dtype(info['df'][info['x_time']])) & (info['x_time_start'] != None):  
    if info['x_time_start'] == 'W':
        ticks_term = pd.date_range(start=info['df'][info['x_time']].min(), end=info['df'][info['x_time']].max(), freq=info['x_time_start'])
        plt.xticks(ticks_term, [tick.strftime('%m-%d') for tick in ticks_term])  # 월-일 형식으로 설정
    else:
        ticks_term = pd.date_range(start=info['df'][info['x_time']].min(), end=info['df'][info['x_time']].max(), freq=info['x_time_start'])
        plt.xticks(ticks_term, ticks_term.strftime('%Y-%m-%d'))
plt.title(f'날짜별 {info['y']}', fontsize=15)




info = dict(df=B_drop1[B_drop1['Timestamp']>='2018-12-08'], x_time='Timestamp', y='LAB_8CHIP_L', x_time_term = None, x_time_start = '3H')
plt.figure(figsize=(9,4))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.lineplot(data=info['df'], x=info['x_time'], y=info['y'])
plt.axvline(pd.to_datetime('2018-12-08 06:00:00'), color='red', linestyle='--', label='12-08 이상 발생')
plt.legend()
unique_dates = info['df'][info['x_time']].unique()
if (pd.api.types.is_numeric_dtype(info['df'][info['x_time']])) & (info['x_time_term'] != None):
    plt.xticks(np.arange(unique_dates.min(), unique_dates.max()+1, step=info['x_time_term']).astype(int))
elif (pd.api.types.is_datetime64_any_dtype(info['df'][info['x_time']])) & (info['x_time_start'] != None):  
    if info['x_time_start'] == 'W':
        ticks_term = pd.date_range(start=info['df'][info['x_time']].min(), end=info['df'][info['x_time']].max(), freq=info['x_time_start'])
        plt.xticks(ticks_term, [tick.strftime('%m-%d') for tick in ticks_term])  # 월-일 형식으로 설정
    elif (info['x_time_start'] == 'H') | (info['x_time_start'] == '3H'):
        ticks_term = pd.date_range(start=info['df'][info['x_time']].min(), end=info['df'][info['x_time']].max(), freq=info['x_time_start'])
        plt.xticks(ticks_term, [tick.strftime('%m-%d %H') for tick in ticks_term], rotation=45)
    else:
        ticks_term = pd.date_range(start=info['df'][info['x_time']].min(), end=info['df'][info['x_time']].max(), freq=info['x_time_start'])
        plt.xticks(ticks_term, ticks_term.strftime('%Y-%m-%d'))
plt.title(f'날짜별 {info['y']}', fontsize=15)




info = dict(df=B_drop1, x_time='Timestamp', y='8TIC_E4312.PV', x_time_term = None, x_time_start = 'W')
plt.figure(figsize=(7,4))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.lineplot(data=info['df'], x=info['x_time'], y=info['y'])
plt.axvline(pd.to_datetime('2018-12-08 07:00:00'), color='red', linestyle='--', label='12-08 이상 발생')
plt.legend()
unique_dates = info['df'][info['x_time']].unique()
if (pd.api.types.is_numeric_dtype(info['df'][info['x_time']])) & (info['x_time_term'] != None):
    plt.xticks(np.arange(unique_dates.min(), unique_dates.max()+1, step=info['x_time_term']).astype(int))
elif (pd.api.types.is_datetime64_any_dtype(info['df'][info['x_time']])) & (info['x_time_start'] != None):  
    if info['x_time_start'] == 'W':
        ticks_term = pd.date_range(start=info['df'][info['x_time']].min(), end=info['df'][info['x_time']].max(), freq=info['x_time_start'])
        plt.xticks(ticks_term, [tick.strftime('%m-%d') for tick in ticks_term])  # 월-일 형식으로 설정
    else:
        ticks_term = pd.date_range(start=info['df'][info['x_time']].min(), end=info['df'][info['x_time']].max(), freq=info['x_time_start'])
        plt.xticks(ticks_term, ticks_term.strftime('%Y-%m-%d'))
plt.title(f'날짜별 {info['y']}', fontsize=15)

info = dict(df=B_drop1, x_time='Timestamp', y='8TIC_E4312.PV', x_time_term = None, x_time_start = 'W')
plt.figure(figsize=(7,4))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.lineplot(data=info['df'], x=info['x_time'], y=info['y'])
plt.axvline(pd.to_datetime('2018-12-08 07:00:00'), color='red', linestyle='--', label='12-08 이상 발생')
plt.axvline(pd.to_datetime('2018-12-06 06:00:00'), color='green', linestyle='--', label='이상 발생 2일 전')
plt.legend()
unique_dates = info['df'][info['x_time']].unique()
if (pd.api.types.is_numeric_dtype(info['df'][info['x_time']])) & (info['x_time_term'] != None):
    plt.xticks(np.arange(unique_dates.min(), unique_dates.max()+1, step=info['x_time_term']).astype(int))
elif (pd.api.types.is_datetime64_any_dtype(info['df'][info['x_time']])) & (info['x_time_start'] != None):  
    if info['x_time_start'] == 'W':
        ticks_term = pd.date_range(start=info['df'][info['x_time']].min(), end=info['df'][info['x_time']].max(), freq=info['x_time_start'])
        plt.xticks(ticks_term, [tick.strftime('%m-%d') for tick in ticks_term])  # 월-일 형식으로 설정
    else:
        ticks_term = pd.date_range(start=info['df'][info['x_time']].min(), end=info['df'][info['x_time']].max(), freq=info['x_time_start'])
        plt.xticks(ticks_term, ticks_term.strftime('%Y-%m-%d'))
plt.title(f'날짜별 {info['y']}', fontsize=15)


info = dict(df=B_drop1, x_time='Timestamp', y='8TI_D9210B.PV', y2='' x_time_term = None, x_time_start = 'W')
plt.figure(figsize=(7,4))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.lineplot(data=info['df'], x=info['x_time'], y=info['y'])
sns.lineplot(data=info['df'], x=info['x_time'], y=info['y'])
plt.axvline(pd.to_datetime('2018-12-08 07:00:00'), color='red', linestyle='--', label='12-08 이상 발생')
plt.axvline(pd.to_datetime('2018-12-06 06:00:00'), color='green', linestyle='--', label='이상 발생 2일 전')
plt.legend()
unique_dates = info['df'][info['x_time']].unique()
if (pd.api.types.is_numeric_dtype(info['df'][info['x_time']])) & (info['x_time_term'] != None):
    plt.xticks(np.arange(unique_dates.min(), unique_dates.max()+1, step=info['x_time_term']).astype(int))
elif (pd.api.types.is_datetime64_any_dtype(info['df'][info['x_time']])) & (info['x_time_start'] != None):  
    if info['x_time_start'] == 'W':
        ticks_term = pd.date_range(start=info['df'][info['x_time']].min(), end=info['df'][info['x_time']].max(), freq=info['x_time_start'])
        plt.xticks(ticks_term, [tick.strftime('%m-%d') for tick in ticks_term])  # 월-일 형식으로 설정
    else:
        ticks_term = pd.date_range(start=info['df'][info['x_time']].min(), end=info['df'][info['x_time']].max(), freq=info['x_time_start'])
        plt.xticks(ticks_term, ticks_term.strftime('%Y-%m-%d'))
plt.title(f'날짜별 {info['y']}', fontsize=15)


info = dict(df=B_drop1, x_time='Timestamp', y='8TIC_E4312.PV', x_time_term = None, x_time_start = 'W')
plt.figure(figsize=(7,4))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.lineplot(data=info['df'], x=info['x_time'], y=info['y'])
plt.axvline(pd.to_datetime('2018-12-08 07:00:00'), color='red', linestyle='--', label='12-08 이상 발생')
plt.axvline(pd.to_datetime('2018-11-22 06:00:00'), color='green', linestyle='--', label='이상 발생 16일 전')
plt.legend()
unique_dates = info['df'][info['x_time']].unique()
if (pd.api.types.is_numeric_dtype(info['df'][info['x_time']])) & (info['x_time_term'] != None):
    plt.xticks(np.arange(unique_dates.min(), unique_dates.max()+1, step=info['x_time_term']).astype(int))
elif (pd.api.types.is_datetime64_any_dtype(info['df'][info['x_time']])) & (info['x_time_start'] != None):  
    if info['x_time_start'] == 'W':
        ticks_term = pd.date_range(start=info['df'][info['x_time']].min(), end=info['df'][info['x_time']].max(), freq=info['x_time_start'])
        plt.xticks(ticks_term, [tick.strftime('%m-%d') for tick in ticks_term])  # 월-일 형식으로 설정
    else:
        ticks_term = pd.date_range(start=info['df'][info['x_time']].min(), end=info['df'][info['x_time']].max(), freq=info['x_time_start'])
        plt.xticks(ticks_term, ticks_term.strftime('%Y-%m-%d'))
plt.title(f'날짜별 {info['y']}', fontsize=15)









info = dict(df=B_drop1, x_time='Timestamp', y='8PIC_D4210.PV', x_time_term = None, x_time_start = 'W')
plt.figure(figsize=(7,4))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.lineplot(data=info['df'], x=info['x_time'], y=info['y'])
plt.axvline(pd.to_datetime('2018-12-08 06:00:00'), color='red', linestyle='--', label='12-08 이상 발생')
plt.legend()
unique_dates = info['df'][info['x_time']].unique()
if (pd.api.types.is_numeric_dtype(info['df'][info['x_time']])) & (info['x_time_term'] != None):
    plt.xticks(np.arange(unique_dates.min(), unique_dates.max()+1, step=info['x_time_term']).astype(int))
elif (pd.api.types.is_datetime64_any_dtype(info['df'][info['x_time']])) & (info['x_time_start'] != None):  
    if info['x_time_start'] == 'W':
        ticks_term = pd.date_range(start=info['df'][info['x_time']].min(), end=info['df'][info['x_time']].max(), freq=info['x_time_start'])
        plt.xticks(ticks_term, [tick.strftime('%m-%d') for tick in ticks_term])  # 월-일 형식으로 설정
    else:
        ticks_term = pd.date_range(start=info['df'][info['x_time']].min(), end=info['df'][info['x_time']].max(), freq=info['x_time_start'])
        plt.xticks(ticks_term, ticks_term.strftime('%Y-%m-%d'))
plt.title(f'날짜별 {info['y']}', fontsize=15)



info = dict(df=B_drop1, x_time='Timestamp', y='8DIC_P4240.PV', x_time_term = None, x_time_start = 'W')
plt.figure(figsize=(7,4))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.lineplot(data=info['df'], x=info['x_time'], y=info['y'])
plt.axvline(pd.to_datetime('2018-12-08 06:00:00'), color='red', linestyle='--', label='12-08 이상 발생')
plt.legend()
unique_dates = info['df'][info['x_time']].unique()
if (pd.api.types.is_numeric_dtype(info['df'][info['x_time']])) & (info['x_time_term'] != None):
    plt.xticks(np.arange(unique_dates.min(), unique_dates.max()+1, step=info['x_time_term']).astype(int))
elif (pd.api.types.is_datetime64_any_dtype(info['df'][info['x_time']])) & (info['x_time_start'] != None):  
    if info['x_time_start'] == 'W':
        ticks_term = pd.date_range(start=info['df'][info['x_time']].min(), end=info['df'][info['x_time']].max(), freq=info['x_time_start'])
        plt.xticks(ticks_term, [tick.strftime('%m-%d') for tick in ticks_term])  # 월-일 형식으로 설정
    else:
        ticks_term = pd.date_range(start=info['df'][info['x_time']].min(), end=info['df'][info['x_time']].max(), freq=info['x_time_start'])
        plt.xticks(ticks_term, ticks_term.strftime('%Y-%m-%d'))
plt.title(f'날짜별 {info['y']}', fontsize=15)


info = dict(df=B_drop1, x_time='Timestamp', y='8DIC_P4240.PV', x_time_term = None, x_time_start = 'W')
plt.figure(figsize=(7,4))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.lineplot(data=info['df'], x=info['x_time'], y=info['y'])
plt.axvline(pd.to_datetime('2018-12-08 06:00:00'), color='red', linestyle='--', label='12-08 이상 발생')
plt.legend()
unique_dates = info['df'][info['x_time']].unique()
if (pd.api.types.is_numeric_dtype(info['df'][info['x_time']])) & (info['x_time_term'] != None):
    plt.xticks(np.arange(unique_dates.min(), unique_dates.max()+1, step=info['x_time_term']).astype(int))
elif (pd.api.types.is_datetime64_any_dtype(info['df'][info['x_time']])) & (info['x_time_start'] != None):  
    if info['x_time_start'] == 'W':
        ticks_term = pd.date_range(start=info['df'][info['x_time']].min(), end=info['df'][info['x_time']].max(), freq=info['x_time_start'])
        plt.xticks(ticks_term, [tick.strftime('%m-%d') for tick in ticks_term])  # 월-일 형식으로 설정
    else:
        ticks_term = pd.date_range(start=info['df'][info['x_time']].min(), end=info['df'][info['x_time']].max(), freq=info['x_time_start'])
        plt.xticks(ticks_term, ticks_term.strftime('%Y-%m-%d'))
plt.title(f'날짜별 {info['y']}', fontsize=15)


info = dict(df=B_drop1, x_time='Timestamp', y='8II_P9120B.PV', x_time_term = None, x_time_start = 'W')
plt.figure(figsize=(7,4))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.lineplot(data=info['df'], x=info['x_time'], y=info['y'])
plt.axvline(pd.to_datetime('2018-12-08 06:00:00'), color='red', linestyle='--', label='12-08 이상 발생')
plt.axvline(pd.to_datetime('2018-12-01 06:00:00'), color='green', linestyle='--', label='이상 발생 1주 전')
plt.legend()
unique_dates = info['df'][info['x_time']].unique()
if (pd.api.types.is_numeric_dtype(info['df'][info['x_time']])) & (info['x_time_term'] != None):
    plt.xticks(np.arange(unique_dates.min(), unique_dates.max()+1, step=info['x_time_term']).astype(int))
elif (pd.api.types.is_datetime64_any_dtype(info['df'][info['x_time']])) & (info['x_time_start'] != None):  
    if info['x_time_start'] == 'W':
        ticks_term = pd.date_range(start=info['df'][info['x_time']].min(), end=info['df'][info['x_time']].max(), freq=info['x_time_start'])
        plt.xticks(ticks_term, [tick.strftime('%m-%d') for tick in ticks_term])  # 월-일 형식으로 설정
    else:
        ticks_term = pd.date_range(start=info['df'][info['x_time']].min(), end=info['df'][info['x_time']].max(), freq=info['x_time_start'])
        plt.xticks(ticks_term, ticks_term.strftime('%Y-%m-%d'))
plt.title(f'날짜별 {info['y']}', fontsize=15)






info = dict(df=B_drop1[B_drop1['Timestamp']>'2018-11-22 00:00:00'], x_time='Timestamp', y='8II_E9130B.PV', x_time_term = None, x_time_start = 'W')
plt.figure(figsize=(5,6))
plt.subplot(2,1,1)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.lineplot(data=info['df'], x=info['x_time'], y=info['y'], linewidth=0.5)
plt.axvline(pd.to_datetime('2018-12-07 10:00:00'), color='orange', linestyle='--', label='상관이 높은 시점', linewidth=2)
plt.axvline(pd.to_datetime('2018-12-08 06:00:00'), color='red', linestyle='--', label='12-08 이상 발생', linewidth=2)
plt.legend()
unique_dates = info['df'][info['x_time']].unique()
if (pd.api.types.is_numeric_dtype(info['df'][info['x_time']])) & (info['x_time_term'] != None):
    plt.xticks(np.arange(unique_dates.min(), unique_dates.max()+1, step=info['x_time_term']).astype(int))
elif (pd.api.types.is_datetime64_any_dtype(info['df'][info['x_time']])) & (info['x_time_start'] != None):  
    if info['x_time_start'] == 'W':
        ticks_term = pd.date_range(start=info['df'][info['x_time']].min(), end=info['df'][info['x_time']].max(), freq=info['x_time_start'])
        plt.xticks(ticks_term, [tick.strftime('%m-%d') for tick in ticks_term])  # 월-일 형식으로 설정
    else:
        ticks_term = pd.date_range(start=info['df'][info['x_time']].min(), end=info['df'][info['x_time']].max(), freq=info['x_time_start'])
        plt.xticks(ticks_term, ticks_term.strftime('%Y-%m-%d'))
plt.title(f'날짜별 {info['y']}', fontsize=15)

info = dict(df=B_drop1[B_drop1['Timestamp']>'2018-11-22 00:00:00'], x_time='Timestamp', y='LAB_8CHIP_L', x_time_term = None, x_time_start = 'W')
plt.subplot(2,1,2)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.lineplot(data=info['df'], x=info['x_time'], y=info['y'], linewidth=0.5)
plt.axvline(pd.to_datetime('2018-12-08 06:00:00'), color='red', linestyle='--', label='12-08 이상 발생', linewidth=2)
plt.legend()
unique_dates = info['df'][info['x_time']].unique()
if (pd.api.types.is_numeric_dtype(info['df'][info['x_time']])) & (info['x_time_term'] != None):
    plt.xticks(np.arange(unique_dates.min(), unique_dates.max()+1, step=info['x_time_term']).astype(int))
elif (pd.api.types.is_datetime64_any_dtype(info['df'][info['x_time']])) & (info['x_time_start'] != None):  
    if info['x_time_start'] == 'W':
        ticks_term = pd.date_range(start=info['df'][info['x_time']].min(), end=info['df'][info['x_time']].max(), freq=info['x_time_start'])
        plt.xticks(ticks_term, [tick.strftime('%m-%d') for tick in ticks_term])  # 월-일 형식으로 설정
    else:
        ticks_term = pd.date_range(start=info['df'][info['x_time']].min(), end=info['df'][info['x_time']].max(), freq=info['x_time_start'])
        plt.xticks(ticks_term, ticks_term.strftime('%Y-%m-%d'))
plt.title(f'날짜별 {info['y']}', fontsize=15)
plt.tight_layout()  #  plt.show() 전에 있어야 적용됨.
plt.show()  # for문 안에 있으면 그래프 1개씩 보여줌





#
info = dict(df=B_drop1[B_drop1['Timestamp']>'2018-11-22 00:00:00'], x_time='Timestamp', y='8TIC_E4312.PV', x_time_term = None, x_time_start = 'W')
plt.figure(figsize=(5,6))
plt.subplot(2,1,1)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.lineplot(data=info['df'], x=info['x_time'], y=info['y'], linewidth=0.5)
plt.axvline(pd.to_datetime('2018-11-22 14:32:00'), color='orange', linestyle='--', label='상관이 높은 시점', linewidth=2)
plt.axvline(pd.to_datetime('2018-12-08 06:00:00'), color='red', linestyle='--', label='12-08 이상 발생', linewidth=2)
plt.legend()
unique_dates = info['df'][info['x_time']].unique()
if (pd.api.types.is_numeric_dtype(info['df'][info['x_time']])) & (info['x_time_term'] != None):
    plt.xticks(np.arange(unique_dates.min(), unique_dates.max()+1, step=info['x_time_term']).astype(int))
elif (pd.api.types.is_datetime64_any_dtype(info['df'][info['x_time']])) & (info['x_time_start'] != None):  
    if info['x_time_start'] == 'W':
        ticks_term = pd.date_range(start=info['df'][info['x_time']].min(), end=info['df'][info['x_time']].max(), freq=info['x_time_start'])
        plt.xticks(ticks_term, [tick.strftime('%m-%d') for tick in ticks_term])  # 월-일 형식으로 설정
    else:
        ticks_term = pd.date_range(start=info['df'][info['x_time']].min(), end=info['df'][info['x_time']].max(), freq=info['x_time_start'])
        plt.xticks(ticks_term, ticks_term.strftime('%Y-%m-%d'))
plt.title(f'날짜별 {info['y']}', fontsize=15)

info = dict(df=B_drop1[B_drop1['Timestamp']>'2018-11-22 00:00:00'], x_time='Timestamp', y='LAB_8CHIP_L', x_time_term = None, x_time_start = 'W')
plt.subplot(2,1,2)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.lineplot(data=info['df'], x=info['x_time'], y=info['y'], linewidth=0.5)
plt.axvline(pd.to_datetime('2018-12-08 06:00:00'), color='red', linestyle='--', label='12-08 이상 발생', linewidth=2)
plt.legend()
unique_dates = info['df'][info['x_time']].unique()
if (pd.api.types.is_numeric_dtype(info['df'][info['x_time']])) & (info['x_time_term'] != None):
    plt.xticks(np.arange(unique_dates.min(), unique_dates.max()+1, step=info['x_time_term']).astype(int))
elif (pd.api.types.is_datetime64_any_dtype(info['df'][info['x_time']])) & (info['x_time_start'] != None):  
    if info['x_time_start'] == 'W':
        ticks_term = pd.date_range(start=info['df'][info['x_time']].min(), end=info['df'][info['x_time']].max(), freq=info['x_time_start'])
        plt.xticks(ticks_term, [tick.strftime('%m-%d') for tick in ticks_term])  # 월-일 형식으로 설정
    else:
        ticks_term = pd.date_range(start=info['df'][info['x_time']].min(), end=info['df'][info['x_time']].max(), freq=info['x_time_start'])
        plt.xticks(ticks_term, ticks_term.strftime('%Y-%m-%d'))
plt.title(f'날짜별 {info['y']}', fontsize=15)
plt.tight_layout()  #  plt.show() 전에 있어야 적용됨.
plt.show()  # for문 안에 있으면 그래프 1개씩 보여줌



#
info = dict(df=B_drop1[(B_drop1['Timestamp']>'2018-12-05 00:00:00')], x_time='Timestamp', y='8TI_P4350A.PV', x_time_term = None, x_time_start = 'D')
plt.figure(figsize=(5,7))
plt.subplot(2,1,1)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.lineplot(data=info['df'], x=info['x_time'], y=info['y'], linewidth=0.5)
plt.axvline(pd.to_datetime('2018-12-08 05:09:00'), color='orange', linestyle='--', label='상관이 높은 시점', linewidth=2)
plt.axvline(pd.to_datetime('2018-12-08 06:00:00'), color='red', linestyle='--', label='12-08 이상 발생', linewidth=2)
plt.legend()
unique_dates = info['df'][info['x_time']].unique()
if (pd.api.types.is_numeric_dtype(info['df'][info['x_time']])) & (info['x_time_term'] != None):
    plt.xticks(np.arange(unique_dates.min(), unique_dates.max()+1, step=info['x_time_term']).astype(int))
elif (pd.api.types.is_datetime64_any_dtype(info['df'][info['x_time']])) & (info['x_time_start'] != None):  
    if info['x_time_start'] == 'W':
        ticks_term = pd.date_range(start=info['df'][info['x_time']].min(), end=info['df'][info['x_time']].max(), freq=info['x_time_start'])
        plt.xticks(ticks_term, [tick.strftime('%m-%d') for tick in ticks_term])  # 월-일 형식으로 설정
    elif (info['x_time_start'] == 'H') | (info['x_time_start'] == '3H'):
        ticks_term = pd.date_range(start=info['df'][info['x_time']].min(), end=info['df'][info['x_time']].max(), freq=info['x_time_start'])
        plt.xticks(ticks_term, [tick.strftime('%m-%d %H') for tick in ticks_term], rotation=45)
    else:
        ticks_term = pd.date_range(start=info['df'][info['x_time']].min(), end=info['df'][info['x_time']].max(), freq=info['x_time_start'])
        plt.xticks(ticks_term, ticks_term.strftime('%Y-%m-%d'))
plt.title(f'날짜별 {info['y']}', fontsize=15)

info = dict(df=B_drop1[(B_drop1['Timestamp']>'2018-12-05 00:00:00')], x_time='Timestamp', y='LAB_8CHIP_L', x_time_term = None, x_time_start = 'D')
plt.subplot(2,1,2)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.lineplot(data=info['df'], x=info['x_time'], y=info['y'], linewidth=0.5)
plt.axvline(pd.to_datetime('2018-12-08 06:00:00'), color='red', linestyle='--', label='12-08 이상 발생', linewidth=2)
plt.legend()
unique_dates = info['df'][info['x_time']].unique()
if (pd.api.types.is_numeric_dtype(info['df'][info['x_time']])) & (info['x_time_term'] != None):
    plt.xticks(np.arange(unique_dates.min(), unique_dates.max()+1, step=info['x_time_term']).astype(int))
elif (pd.api.types.is_datetime64_any_dtype(info['df'][info['x_time']])) & (info['x_time_start'] != None):  
    if info['x_time_start'] == 'W':
        ticks_term = pd.date_range(start=info['df'][info['x_time']].min(), end=info['df'][info['x_time']].max(), freq=info['x_time_start'])
        plt.xticks(ticks_term, [tick.strftime('%m-%d') for tick in ticks_term])  # 월-일 형식으로 설정
    elif (info['x_time_start'] == 'H') | (info['x_time_start'] == '3H'):
        ticks_term = pd.date_range(start=info['df'][info['x_time']].min(), end=info['df'][info['x_time']].max(), freq=info['x_time_start'])
        plt.xticks(ticks_term, [tick.strftime('%m-%d %H') for tick in ticks_term], rotation=45)
    else:
        ticks_term = pd.date_range(start=info['df'][info['x_time']].min(), end=info['df'][info['x_time']].max(), freq=info['x_time_start'])
        plt.xticks(ticks_term, ticks_term.strftime('%Y-%m-%d'))
plt.title(f'날짜별 {info['y']}', fontsize=15)
plt.tight_layout()  #  plt.show() 전에 있어야 적용됨.
plt.show()  # for문 안에 있으면 그래프 1개씩 보여줌






# -----------------------------  ADF 검정
from statsmodels.tsa.stattools import adfuller

non_stationary = []
for i in B_index.columns:
    result = adfuller(B_index[i])

    # 결과 출력
    print("-"*30,"\n",f'{i}컬럼의 ADF Statistic:', result[0])
    print(f'{i}컬럼의 p-value:', result[1])
    print(f'{i}컬럼의 Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value}')  # statistic 값이 임계값(value)보다 작으면 귀무가설 기각 (정상성 만족)
    if result[1] < 0.05:
        print(f'{i}컬럼은 정상적이다')
    elif result[1] >= 0.05:
        print(f'{i}컬럼은 비정상적이다')
        non_stationary.append(i)


i = '8PIC_J9130.PV'
result = adfuller(B_index[i])

# 결과 출력
print("-"*30,"\n",f'{i}컬럼의 ADF Statistic:', result[0])
print(f'{i}컬럼의 p-value:', result[1])
print(f'{i}컬럼의 Critical Values:')
for key, value in result[4].items():
    print(f'   {key}: {value}')  # statistic 값이 임계값(value)보다 작으면 귀무가설 기각 (정상성 만족)
if result[1] < 0.05:
    print(f'{i}컬럼은 정상적이다 \n',"-"*30)
elif result[1] >= 0.05:
    print(f'{i}컬럼은 비정상적이다 \n',"-"*30)



i = '8PIC_D4210.PV'
result = adfuller(B_index[i])

# 결과 출력
print("-"*30,"\n",f'{i}컬럼의 ADF Statistic:', result[0])
print(f'{i}컬럼의 p-value:', result[1])
print(f'{i}컬럼의 Critical Values:')
for key, value in result[4].items():
    print(f'   {key}: {value}')  # statistic 값이 임계값(value)보다 작으면 귀무가설 기각 (정상성 만족)
if result[1] < 0.05:
    print(f'{i}컬럼은 정상적이다 \n',"-"*30)
elif result[1] >= 0.05:
    print(f'{i}컬럼은 비정상적이다 \n',"-"*30)


i = 'LAB_8CHIP_L'
result = adfuller(B_index[i])

# 결과 출력
print("-"*30,"\n",f'{i}컬럼의 ADF Statistic:', result[0])
print(f'{i}컬럼의 p-value:', result[1])
print(f'{i}컬럼의 Critical Values:')
for key, value in result[4].items():
    print(f'   {key}: {value}')  # statistic 값이 임계값(value)보다 작으면 귀무가설 기각 (정상성 만족)
if result[1] < 0.05:
    print(f'{i}컬럼은 정상적이다 \n',"-"*30)
elif result[1] >= 0.05:
    print(f'{i}컬럼은 비정상적이다 \n',"-"*30)







# 비정상

non_stationary = \
['8FIC_P4250A.PV', '8PIC_E9260.PV', '8PIC_R4260.PV', '8TI_R4260C.PV', '8TI_R4260D.PV', '8TI_P9270A.PV', \
 '8TI_P9270B.PV', '8TI_T4260E.PV', '8TI_T4260G.PV', '8TIC_E4262.PV', '8TIC_E9221A.PV', '8TIC_R4310.PV',\
 '8PIC_R4310A.PV', '8TIC_E4332.PV', '8II_P4311B.PV', '8II_P4311A.PV', '8PI_R4310B.PV', '8PI_R4310A.PV',\
 '8PI_T4311B.PV', '8FI_E4385.PV', '8TI_J4387.PV', '8TI_J4388.PV', '8LIC_R4330.PV', '8TI_P4340F.PV',\
 '8TI_P4340G.PV', '8TI_P4340E.PV', '8TI_P4340D.PV', '8TI_K4330C.PV', '8TI_P4340.PV', '8TIC_R4330.PV',\
 '8SIC_P4340.PV', '8II_P4340.PV', '8PI_P4340.PV', '8PI_P4340C.PV', '8PI_P4340A.PV', '8PI_P4340B.PV',\
 '8TI_Z4342A.PV', '8TI_Z4342B.PV', '8TI_E4340A.PV', '8TI_P4334.PV', '8FI_P4332.PV', '8PI_E4340.PV',\
 '8TIC_Z4343.PV', '8TI_P4361B.PV', '8TI_P4360C.PV', '8TI_E9360A.PV', '8PIC_E9360.PV', '8TI_E9330A.PV',\
 '8PIC_E9330.PV', '8TI_P9370F.PV', '8TI_P9370A.PV', '8TI_P9370D.PV', '8TI_P9362E.PV', '8TI_P9362C.PV',\
 '8TI_P9362D.PV', '8TI_P9362B.PV', '8TI_P9362A.PV', '8TI_P9362F.PV', '8TI_P9360D.PV', '8TI_P9360A.PV',\
 '8TI_P9361A.PV', '8TI_P9361D.PV', '8TI_P9361E.PV', '8TI_P9361B.PV', '8TI_P9361F.PV', '8TI_P9360G.PV',\
 '8TI_P9360E.PV', '8TI_E9360B.PV', '8TI_P9360C.PV', '8TI_P9360F.PV', '8TI_R9311B.PV', '8TI_E9261A.PV',\
 '8TI_R9260D.PV', '8TI_Z9350F.PV', '8TI_Z9350A.PV', '8TI_E9351B.PV', '8II_E9130A.PV', '8PIC_J9130.PV',\
 '8PI_J9130.PV', '8TI_E9140.PV', '8TI_D9140.PV', '8LI_D9140.PV', '8II_P9130B.PV', '8TI_E9270A.PV',\
 '8TI_E9270B.PV', '8TI_F9270C.PV', '8TI_Z9270A.PV', '8TI_Z9270B.PV', '8TI_Z9270C.PV', '8II_P9120B.PV',\
 '8II_P9120A.PV', '8II_P9111B.PV', '8II_P9111A.PV', '8LI_D8465.PV', '8TI_R9330A.PV', '8TI_R9330B.PV',\
 '8TI_R9330C.PV', '8TI_R9330D.PV', '8TI_R9330E.PV', '8TI_R9330F.PV', '8TI_E9330F.PV', '8TI_E9330E.PV',\
 '8TI_E9330D.PV', '8TI_E9330C.PV', '8TI_E9330B.PV', '8TI_E9330G.PV', 'LAB_8CHIP_B',\
 'LAB_8CHIP_DEG', 'LAB_8CHIP_COOH', 'LAB_8CHIP_TM', 'LAB_8CHIP_SIZE']
stationary = B_drop1.columns[~B_drop1.columns.isin(non_stationary)]


B_drop1[non_stationary]



info = dict(df=B_drop1, x_time='Timestamp', y='8FIC_P4250A.PV', x_time_term = None, x_time_start = 'W',window=1440)

def trend_timeline(df, x_time, x_time_term=None, x_time_start=None, window=None):
    n=len(df.columns)*2
    plt.figure(figsize=(13, n*2.5))
    for index, col_u in enumerate(df.columns.drop(x_time), 1):
        
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
        
        df_copy = df.copy()
        df_copy['moving_avg'] = df[col_u].rolling(window=window).mean()
        plt.subplot(n, 4, index*2-1)
        sns.lineplot(data=df, x=x_time, y=col_u)
        sns.lineplot(data=df_copy, x=x_time, y='moving_avg', label=f'{window} 이동 평균', color='orange')
        plt.axhline(df[col_u].mean(), color='red', linestyle='--', label='전체 평균값') 
        plt.legend()
        unique_dates = df[x_time].unique()
        if (pd.api.types.is_numeric_dtype(df[x_time])) & (x_time_term != None):
            plt.xticks(np.arange(unique_dates.min(), unique_dates.max()+1, step=x_time_term).astype(int))
        elif (pd.api.types.is_datetime64_any_dtype(df[x_time])) & (x_time_start != None):  
            if x_time_start == 'W':
                ticks_term = pd.date_range(start=df[x_time].min(), end=df[x_time].max(), freq=x_time_start)
                plt.xticks(ticks_term, [tick.strftime('%m-%d') for tick in ticks_term])  # 월-일 형식으로 설정
            else:
                ticks_term = pd.date_range(start=df[x_time].min(), end=df[x_time].max(), freq=x_time_start)
                plt.xticks(ticks_term, ticks_term.strftime('%Y-%m-%d'))
        plt.title(f'날짜별 {col_u}', fontsize=15)
        std_dev = np.round(df[col_u].std(),3)
        plt.xlabel(f'{x_time}  [ {col_u}의 std : {std_dev} ]')
        
        if window != None:
            plt.subplot(n, 4, index*2)
            sns.lineplot(data=df_copy, x=x_time, y='moving_avg', label=f'{window} 이동 평균', color='orange')
            plt.axhline(df[col_u].mean(), color='red', linestyle='--', label='전체 평균값') 
            if (pd.api.types.is_numeric_dtype(df[x_time])) & (x_time_term != None):
                plt.xticks(np.arange(unique_dates.min(), unique_dates.max()+1, step=x_time_term).astype(int))
            elif (pd.api.types.is_datetime64_any_dtype(df[x_time])) & (x_time_start != None):  
                if x_time_start == 'W':
                    ticks_term = pd.date_range(start=df[x_time].min(), end=df[x_time].max(), freq=x_time_start)
                    plt.xticks(ticks_term, [tick.strftime('%m-%d') for tick in ticks_term])  # 월-일 형식으로 설정
                else:
                    ticks_term = pd.date_range(start=df[x_time].min(), end=df[x_time].max(), freq=x_time_start)
                    plt.xticks(ticks_term, ticks_term.strftime('%Y-%m-%d'))
            plt.legend()
    plt.tight_layout()  
    plt.show()

trend_timeline(df=B_drop1[non_stationary+['Timestamp']], x_time='Timestamp', x_time_start='W', window=1440)
trend_timeline(df=B_drop1[non_stationary[:2]], x_time='Timestamp', x_time_start='W', window=1440)




info = dict(df=B_drop1, x_time='Timestamp', y='8PIC_D4210.PV', x_time_term = None, x_time_start = 'W',window=1440)
plt.figure(figsize=(7,3))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
df_copy = info['df'].copy()
df_copy['moving_avg'] = info['df'][info['y']].rolling(window=info['window']).mean()
plt.subplot(1,2,1)
sns.lineplot(data=info['df'], x=info['x_time'], y=info['y'])
sns.lineplot(data=df_copy, x=info['x_time'], y='moving_avg', label=f'{info['window']} 이동 평균', color='orange')
plt.axhline(info['df'][info['y']].mean(), color='red', linestyle='--', label='평균값') 
plt.legend()
unique_dates = info['df'][info['x_time']].unique()
if (pd.api.types.is_numeric_dtype(info['df'][info['x_time']])) & (info['x_time_term'] != None):
    plt.xticks(np.arange(unique_dates.min(), unique_dates.max()+1, step=info['x_time_term']).astype(int))
elif (pd.api.types.is_datetime64_any_dtype(info['df'][info['x_time']])) & (info['x_time_start'] != None):  
    if info['x_time_start'] == 'W':
        ticks_term = pd.date_range(start=info['df'][info['x_time']].min(), end=info['df'][info['x_time']].max(), freq=info['x_time_start'])
        plt.xticks(ticks_term, [tick.strftime('%m-%d') for tick in ticks_term])  # 월-일 형식으로 설정
    else:
        ticks_term = pd.date_range(start=info['df'][info['x_time']].min(), end=info['df'][info['x_time']].max(), freq=info['x_time_start'])
        plt.xticks(ticks_term, ticks_term.strftime('%Y-%m-%d'))
plt.title(f'날짜별 {info['y']}', fontsize=15)
std_dev = np.round(info['df'][info['y']].std(),3)
plt.xlabel(f'{info['x_time']}  [ {info['y']}의 std : {std_dev} ]')

if info['window'] != None:
    plt.subplot(1,2,2)
    sns.lineplot(data=df_copy, x=info['x_time'], y='moving_avg', label=f'{info['window']} 이동 평균', color='orange')
    plt.axhline(info['df'][info['y']].mean(), color='red', linestyle='--', label='평균값') 
    if (pd.api.types.is_numeric_dtype(info['df'][info['x_time']])) & (info['x_time_term'] != None):
        plt.xticks(np.arange(unique_dates.min(), unique_dates.max()+1, step=info['x_time_term']).astype(int))
    elif (pd.api.types.is_datetime64_any_dtype(info['df'][info['x_time']])) & (info['x_time_start'] != None):  
        if info['x_time_start'] == 'W':
            ticks_term = pd.date_range(start=info['df'][info['x_time']].min(), end=info['df'][info['x_time']].max(), freq=info['x_time_start'])
            plt.xticks(ticks_term, [tick.strftime('%m-%d') for tick in ticks_term])  # 월-일 형식으로 설정
        else:
            ticks_term = pd.date_range(start=info['df'][info['x_time']].min(), end=info['df'][info['x_time']].max(), freq=info['x_time_start'])
            plt.xticks(ticks_term, ticks_term.strftime('%Y-%m-%d'))
plt.legend()
plt.tight_layout()  
plt.show()



info = dict(df=B_drop1, x_time='Timestamp', y='LAB_8CHIP_L', x_time_term = None, x_time_start = 'W',window=1440)
plt.figure(figsize=(7,3))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
df_copy = info['df'].copy()
df_copy['moving_avg'] = info['df'][info['y']].rolling(window=info['window']).mean()
plt.subplot(1,2,1)
sns.lineplot(data=info['df'], x=info['x_time'], y=info['y'])
sns.lineplot(data=df_copy, x=info['x_time'], y='moving_avg', label=f'{info['window']} 이동 평균', color='orange')
plt.axhline(info['df'][info['y']].mean(), color='red', linestyle='--', label='평균값') 
plt.legend()
unique_dates = info['df'][info['x_time']].unique()
if (pd.api.types.is_numeric_dtype(info['df'][info['x_time']])) & (info['x_time_term'] != None):
    plt.xticks(np.arange(unique_dates.min(), unique_dates.max()+1, step=info['x_time_term']).astype(int))
elif (pd.api.types.is_datetime64_any_dtype(info['df'][info['x_time']])) & (info['x_time_start'] != None):  
    if info['x_time_start'] == 'W':
        ticks_term = pd.date_range(start=info['df'][info['x_time']].min(), end=info['df'][info['x_time']].max(), freq=info['x_time_start'])
        plt.xticks(ticks_term, [tick.strftime('%m-%d') for tick in ticks_term])  # 월-일 형식으로 설정
    else:
        ticks_term = pd.date_range(start=info['df'][info['x_time']].min(), end=info['df'][info['x_time']].max(), freq=info['x_time_start'])
        plt.xticks(ticks_term, ticks_term.strftime('%Y-%m-%d'))
plt.title(f'날짜별 {info['y']}', fontsize=15)
std_dev = np.round(info['df'][info['y']].std(),3)
plt.xlabel(f'{info['x_time']}  [ {info['y']}의 std : {std_dev} ]')

if info['window'] != None:
    plt.subplot(1,2,2)
    sns.lineplot(data=df_copy, x=info['x_time'], y='moving_avg', label=f'{info['window']} 이동 평균', color='orange')
    plt.axhline(info['df'][info['y']].mean(), color='red', linestyle='--', label='평균값') 
    if (pd.api.types.is_numeric_dtype(info['df'][info['x_time']])) & (info['x_time_term'] != None):
        plt.xticks(np.arange(unique_dates.min(), unique_dates.max()+1, step=info['x_time_term']).astype(int))
    elif (pd.api.types.is_datetime64_any_dtype(info['df'][info['x_time']])) & (info['x_time_start'] != None):  
        if info['x_time_start'] == 'W':
            ticks_term = pd.date_range(start=info['df'][info['x_time']].min(), end=info['df'][info['x_time']].max(), freq=info['x_time_start'])
            plt.xticks(ticks_term, [tick.strftime('%m-%d') for tick in ticks_term])  # 월-일 형식으로 설정
        else:
            ticks_term = pd.date_range(start=info['df'][info['x_time']].min(), end=info['df'][info['x_time']].max(), freq=info['x_time_start'])
            plt.xticks(ticks_term, ticks_term.strftime('%Y-%m-%d'))
plt.legend()
plt.tight_layout()  
plt.show()




def kde(df, palette='dark', alpha=0.5):
	numeric_cols = df.select_dtypes(include=['number']).columns   # number에는 boolean도 포함됨
	n = int(np.ceil(len(numeric_cols)/4))
	plt.clf()
	plt.figure(figsize=(12, 2*n))
	for index, col in enumerate(numeric_cols, 1):
		plt.rcParams['font.family'] = 'Malgun Gothic'
		plt.rcParams['axes.unicode_minus'] = False
		plt.subplot(n, 4, index)
		sns.kdeplot(data=df, x=col, fill=True , palette=palette, alpha=alpha)
		plt.title(f'{col}의 확률밀도', fontsize=16)
	plt.tight_layout()  #  plt.show() 전에 있어야 적용됨.
	plt.show()  # for문 안에 있으면 그래프 1개씩 보여줌

kde(box_B)

info = dict(data_frame=B_drop1, col='8PIC_J9130.PV', palette='dark', alpha=0.5)   # 얘만 바꾸면 됨.
plt.clf()
plt.figure(figsize=(6, 4))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.kdeplot(data=info['data_frame'], x=info['col'], fill=True , palette=info['palette'], alpha=info['alpha'])
plt.title(f'{info['col']}의 확률밀도', fontsize=20)
plt.tight_layout()  #  plt.show() 전에 있어야 적용됨.
plt.show()  # for문 안에 있으면 그래프 1개씩 보여줌


info = dict(data_frame=box_B, col='log_8PIC_J9130.PV', palette='dark', alpha=0.5)   # 얘만 바꾸면 됨.
plt.clf()
plt.figure(figsize=(6, 4))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.kdeplot(data=info['data_frame'], x=info['col'], fill=True , palette=info['palette'], alpha=info['alpha'])
plt.title(f'{info['col']}의 확률밀도', fontsize=20)
plt.tight_layout()  #  plt.show() 전에 있어야 적용됨.
plt.show()  # for문 안에 있으면 그래프 1개씩 보여줌


info = dict(data_frame=B_drop1, col='8PIC_J9130.PV', palette='dark', alpha=0.5)   # 얘만 바꾸면 됨.
plt.clf()
plt.figure(figsize=(6, 4))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.kdeplot(data=info['data_frame'], x=info['col'], fill=True , palette=info['palette'], alpha=info['alpha'])
plt.title(f'{info['col']}의 확률밀도', fontsize=20)
plt.tight_layout()  #  plt.show() 전에 있어야 적용됨.
plt.show()  # for문 안에 있으면 그래프 1개씩 보여줌



# 이산형은 교차상관분석할 때 제외하기 (혹은 해석할 때 제외하기)

# boxcox
box_B = B_drop1[non_stationary[:-5]+['Timestamp']]
boxcox_vars = []

for column in box_B.columns[:-1]:
    transformed_variable, lambda_value = stats.boxcox(box_B[column] + 1)
    box_B[f'boxcox_{column}'] = transformed_variable
    boxcox_vars.append((column, lambda_value))

for original, lambda_val in boxcox_vars:
    print(f"{original}의 최적의 Box-Cox 변환 λ 값: {lambda_val}")


# 로그 변환
for column in box_B.columns[:106]:
    box_B[f'log_{column}'] = np.log1p(box_B[column])  # log(1 + value) 적용

for column in box_B.columns[:106]:
    print(box_B[column][box_B[column] == box_B[f'log_{column}']])


# 제곱근 변환
for column in box_B.columns[:106]:
    box_B[f'sqrt_{column}'] = np.sqrt(box_B[column])



# 차분
for column in box_B.columns[:106]:
    box_B[f'diff_{column}'] = box_B[column].diff()




box_B.columns.to_list().index('Timestamp')  # 106


kde(box_B.iloc[:,:106])  # 원래 컬럼들  # 106번째가 날짜 컬럼
kde(box_B.iloc[:,107:213])  # boxcox   # 고유값 1개인 애들때문인지 에러남
kde(box_B.iloc[:,213:319])   # log
kde(box_B.iloc[:,319:425])   # sqrt
kde(box_B.iloc[:,425:])    # 차분

box_B.iloc[:,425:].isna().sum()
box_B.iloc[:,425:].head()


len(box_B.iloc[:,107:].columns)


box_B.iloc[:,107:].isnull().sum()

box_B.iloc[:,107:].isinf().sum()
np.isinf(box_B.iloc[:,107:]).sum()


for i in box_B.iloc[:,107:].columns:
    print(f'{i}컬럼의 고유값 개수 :', len(box_B.iloc[:,107:][i].unique()))



# log 변환에 대해서 ADF 검정
from statsmodels.tsa.stattools import adfuller

log_non_stationary = []
for i in box_B.iloc[:,213:].columns:
    result = adfuller(box_B[i])

    # 결과 출력
    print("-"*30,"\n",f'{i}컬럼의 ADF Statistic:', result[0])
    print(f'{i}컬럼의 p-value:', result[1])
    print(f'{i}컬럼의 Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value}')  # statistic 값이 임계값(value)보다 작으면 귀무가설 기각 (정상성 만족)
    if result[1] < 0.05:
        print(f'{i}컬럼은 정상적이다')
    elif result[1] >= 0.05:
        print(f'{i}컬럼은 비정상적이다')
        log_non_stationary.append(i)



sqrt_non_stationary = []
for i in box_B.iloc[:,319:425].columns:
    result = adfuller(box_B[i])

    # 결과 출력
    print("-"*30,"\n",f'{i}컬럼의 ADF Statistic:', result[0])
    print(f'{i}컬럼의 p-value:', result[1])
    print(f'{i}컬럼의 Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value}')  # statistic 값이 임계값(value)보다 작으면 귀무가설 기각 (정상성 만족)
    if result[1] < 0.05:
        print(f'{i}컬럼은 정상적이다')
    elif result[1] >= 0.05:
        print(f'{i}컬럼은 비정상적이다')
        sqrt_non_stationary.append(i)


diff_non_stationary = []
for i in box_B.iloc[1:,425:].columns:
    result = adfuller(box_B.loc[1:,i])

    # 결과 출력
    print("-"*30,"\n",f'{i}컬럼의 ADF Statistic:', result[0])
    print(f'{i}컬럼의 p-value:', result[1])
    print(f'{i}컬럼의 Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value}')  # statistic 값이 임계값(value)보다 작으면 귀무가설 기각 (정상성 만족)
    if result[1] < 0.05:
        print(f'{i}컬럼은 정상적이다')
    elif result[1] >= 0.05:
        print(f'{i}컬럼은 비정상적이다')
        diff_non_stationary.append(i)





# ---------------- 
non_stationary = \
['8FIC_P4250A.PV', '8PIC_E9260.PV', '8PIC_R4260.PV', '8TI_R4260C.PV', '8TI_R4260D.PV', '8TI_P9270A.PV', \
 '8TI_P9270B.PV', '8TI_T4260E.PV', '8TI_T4260G.PV', '8TIC_E4262.PV', '8TIC_E9221A.PV', '8TIC_R4310.PV',\
 '8PIC_R4310A.PV', '8TIC_E4332.PV', '8II_P4311B.PV', '8II_P4311A.PV', '8PI_R4310B.PV', '8PI_R4310A.PV',\
 '8PI_T4311B.PV', '8FI_E4385.PV', '8TI_J4387.PV', '8TI_J4388.PV', '8LIC_R4330.PV', '8TI_P4340F.PV',\
 '8TI_P4340G.PV', '8TI_P4340E.PV', '8TI_P4340D.PV', '8TI_K4330C.PV', '8TI_P4340.PV', '8TIC_R4330.PV',\
 '8SIC_P4340.PV', '8II_P4340.PV', '8PI_P4340.PV', '8PI_P4340C.PV', '8PI_P4340A.PV', '8PI_P4340B.PV',\
 '8TI_Z4342A.PV', '8TI_Z4342B.PV', '8TI_E4340A.PV', '8TI_P4334.PV', '8FI_P4332.PV', '8PI_E4340.PV',\
 '8TIC_Z4343.PV', '8TI_P4361B.PV', '8TI_P4360C.PV', '8TI_E9360A.PV', '8PIC_E9360.PV', '8TI_E9330A.PV',\
 '8PIC_E9330.PV', '8TI_P9370F.PV', '8TI_P9370A.PV', '8TI_P9370D.PV', '8TI_P9362E.PV', '8TI_P9362C.PV',\
 '8TI_P9362D.PV', '8TI_P9362B.PV', '8TI_P9362A.PV', '8TI_P9362F.PV', '8TI_P9360D.PV', '8TI_P9360A.PV',\
 '8TI_P9361A.PV', '8TI_P9361D.PV', '8TI_P9361E.PV', '8TI_P9361B.PV', '8TI_P9361F.PV', '8TI_P9360G.PV',\
 '8TI_P9360E.PV', '8TI_E9360B.PV', '8TI_P9360C.PV', '8TI_P9360F.PV', '8TI_R9311B.PV', '8TI_E9261A.PV',\
 '8TI_R9260D.PV', '8TI_Z9350F.PV', '8TI_Z9350A.PV', '8TI_E9351B.PV', '8II_E9130A.PV', '8PIC_J9130.PV',\
 '8PI_J9130.PV', '8TI_E9140.PV', '8TI_D9140.PV', '8LI_D9140.PV', '8II_P9130B.PV', '8TI_E9270A.PV',\
 '8TI_E9270B.PV', '8TI_F9270C.PV', '8TI_Z9270A.PV', '8TI_Z9270B.PV', '8TI_Z9270C.PV', '8II_P9120B.PV',\
 '8II_P9120A.PV', '8II_P9111B.PV', '8II_P9111A.PV', '8LI_D8465.PV', '8TI_R9330A.PV', '8TI_R9330B.PV',\
 '8TI_R9330C.PV', '8TI_R9330D.PV', '8TI_R9330E.PV', '8TI_R9330F.PV', '8TI_E9330F.PV', '8TI_E9330E.PV',\
 '8TI_E9330D.PV', '8TI_E9330C.PV', '8TI_E9330B.PV', '8TI_E9330G.PV', 'LAB_8CHIP_B',\
 'LAB_8CHIP_DEG', 'LAB_8CHIP_COOH', 'LAB_8CHIP_TM', 'LAB_8CHIP_SIZE']

stationary = B_drop1.columns[~B_drop1.columns.isin(non_stationary)]
B_stationary = B_drop1[stationary]
B_stationary = B_drop1[['Timestamp']]

for column in non_stationary:
    B_stationary[f'diff_{column}'] = B_drop1[column].diff()

trend_timeline(df=B_stationary.iloc[1:,:], x_time='Timestamp', x_time_start='W', window=1440)
trend_timeline(df=box_B.loc[:,box_B.columns[213:319].to_list()+['Timestamp']], x_time='Timestamp', x_time_start='W', window=1440)



# 확인
B_drop1[['Timestamp', 'LAB_8CHIP_L', '8PIC_D4210.PV']]




# ------------------------------------ 이산형 교차 상관 분석

# diff 차분 포함 교차상관 분석

defect_time = pd.to_datetime("2018-12-08 06:00:00")
start_date = B_drop1['Timestamp'].min()  # 전체 기간의 시작일
end_date = B_drop1['Timestamp'].max()    # 전체 기간의 종료일
a_week_ago = defect_time - pd.Timedelta(minutes=60*24*7) # 결함 7일 전
two_weeks_ago = defect_time - pd.Timedelta(minutes=60*24*7*2) # 결함 14일 전

# 이상이 발생하기 전 데이터만 추출
# 이산형 열 제거
B_drop1.shape
B_drop1_a_week_ago = B_drop1[B_drop1['Timestamp']>= a_week_ago]
B_drop1_two_weeks_ago = B_drop1[B_drop1['Timestamp']>= two_weeks_ago]

########################################## 교차 상관 ccf 패키지 이용
from statsmodels.tsa.stattools import ccf

# 교차 상관계수 계산
def calculate_ccf(df):
    #discrete_columns = list(df.columns[(df.nunique() < 20) & (df.columns != 'LAB_8CHIP_L')])
    #df = df.drop(discrete_columns, axis=1)
    
    cross_corr_results = {}

    for col in df.columns:
        if col != 'LAB_8CHIP_L':
            cross_corr_values = ccf(df['LAB_8CHIP_L'], df[col], adjusted=False)
            cross_corr_results[col] = cross_corr_values

    # 결과를 데이터프레임으로 변환
    cross_corr_df = pd.DataFrame(cross_corr_results)

    # lag 값 추가
    cross_corr_df.index.name = 'lag'
    cross_corr_df = cross_corr_df.reset_index()

    # # 각 열의 절대값 최대값과 해당 lag을 찾기
    max_abs_corr = cross_corr_df.abs().max()  # 절대값 최대값
    max_lags = cross_corr_df.abs().idxmax()   # 최대값의 인덱스(lag)
                    
    # # 각 열의 절대값 최대값의 원래 값을 가져오기
    max_original_values = [cross_corr_df[col][idx] for col, idx in max_lags.items()]

    # # 결과를 데이터프레임으로 정리
    result_df = pd.DataFrame({
        'max_abs_corr': max_abs_corr,
        'original_value': max_original_values,
        'lag': max_lags
    })

    # # 절대값이 높은 순서로 정렬
    result_df = result_df.sort_values(by='max_abs_corr', ascending=False).reset_index()
    result_top10 = result_df[1:11]
    top_10_variable_lag = dict(zip(result_top10['index'], result_top10['lag']))
    top_10_variables = list(top_10_variable_lag.keys())
        
    plt.rcParams['font.family'] ='Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] =False
    for col in top_10_variables:
        sns.lineplot(data=cross_corr_df, x='lag', y=col)
        plt.ylabel('TOP 10 변수')  
        
    return result_top10
    # top_10_variable_lag = dict(zip(result_df[1:11]['index'], result_df[1:11]['lag']))
    # top_10_variables = list(top_10_variable_lag.keys())



stationary = B_drop1.columns[~B_drop1.columns.isin(non_stationary)]


diff_B = pd.merge(B_drop1[stationary] , box_B.loc[:,box_B.columns[425:].to_list()+['Timestamp']], on='Timestamp')
diff_df = diff_B.iloc[1:,:].drop(columns='Timestamp')

df_B_ccf_top10 = calculate_ccf(diff_df)

# ------ 이상 발생 2일 전
lab = ['LAB_8CHIP_IV', 'LAB_8CHIP_B', 'LAB_8CHIP_DEG', 'LAB_8CHIP_COOH', 'LAB_8CHIP_TM', 'LAB_8CHIP_SIZE']

B_lab = B_drop1[lab+['LAB_8CHIP_L']]
B_lab_2 = B_drop1[lab+['LAB_8CHIP_L','Timestamp']][B_lab['Timestamp']>='2018-12-06 06:00:00'].drop(columns='Timestamp')
B_lab_16 = B_drop1[lab+['LAB_8CHIP_L','Timestamp']][B_lab['Timestamp']>='2018-11-22 06:00:00'].drop(columns='Timestamp')

calculate_ccf(B_lab)
calculate_ccf(B_lab_2)
calculate_ccf(B_lab_16)
