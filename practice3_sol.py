import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 파일 불러오기
# df_B1_raw = pd.read_excel('중합 CPS-8 DB B1.xlsx')
# df_B2_raw = pd.read_excel('중합 CPS-8 DB B2.xlsx')

# df_B1_raw.to_csv('중합 CPS-8 DB B1.csv', index=False)
# df_B2_raw.to_csv('중합 CPS-8 DB B2.csv', index=False)

df_B1 = pd.read_csv('중합 CPS-8 DB B1.csv')
df_B2 = pd.read_csv('중합 CPS-8 DB B2.csv')

df_B1.shape, df_B2.shape
# 결측치 확인
sum(df_B1.isna().sum())
sum(df_B2.isna().sum())

# sns.heatmap(df_B1.isnull())
# sns.heatmap(df_B2.isnull())

###############################

df_B1.info()
df_B1.head()
df_B1['Timestamp'] = pd.to_datetime(df_B1['Timestamp'])
df_B2['Timestamp'] = pd.to_datetime(df_B2['Timestamp'])

## 행 제거
# day_of_year 추가
df_B1['day_of_year'] = df_B1['Timestamp'].dt.day_of_year
df_B2['day_of_year'] = df_B2['Timestamp'].dt.day_of_year

# day_of_year별 중복 확인
b1_days = df_B1.groupby('day_of_year',as_index=False).agg(n=('day_of_year','count'))
b2_days = df_B2.groupby('day_of_year',as_index=False).agg(n=('day_of_year','count'))
b1_b2_days_diff = b1_days.merge(b2_days, how='left', on='day_of_year')
b1_b2_days_diff['diff'] = b1_b2_days_diff['n_x']-b1_b2_days_diff['n_y']
b1_b2_days_diff[b1_b2_days_diff['diff'] !=0] # day_of_year가 313,314

# day_of_year 가 313: B1_단독 행(day_of_year == 313)
df_B1 = df_B1[df_B1['day_of_year'] != 313]

# day_of_year가 314: 중복된 시간 확인
count_TImestamp = df_B1.groupby('Timestamp',as_index=False).agg(n=('Timestamp','count'))
count_TImestamp[count_TImestamp['n'] > 1] # 1439
count_TImestamp.iloc[1438:1441]
df_B1['Timestamp'].iloc[1435:1440]
df_B1.query('Timestamp == "2018-11-10 00:00:00"') # 0행과 44640 행이 동일
same_Time = df_B1.query('Timestamp == "2018-11-10 00:00:00"').reset_index(drop=True)
sum( same_Time.iloc[0] != same_Time.iloc[1] ) # 0

# day_of_year 가 314 : B1_중복 행(Timestamp == "2018-11-10 00:00:00) 삭제(맨 마지막 44640행)
df_B1 = df_B1.iloc[:-1]
df_B1

############################ 데이터프레임 합치기 B1+B2
## 열 제거
df_B_columns = list(df_B1.columns) + list(df_B2.columns) 
df_B_columns

# 열이름 중복 확인
from collections import Counter
# 각 요소의 빈도 계산
count = Counter(df_B_columns)
# 중복된 요소와 개수 확인
duplicates = {key: value for key, value in count.items() if value > 1}
print("중복된 요소와 개수:", duplicates)

# 값의 중복 확인
df_B = df_B1.merge(df_B2, how='left', on='Timestamp')
df_B = df_B.drop(['day_of_year_x','day_of_year_y'], axis=1)
corr_matrix = df_B.corr()

# 상관계수 값이 1인 행과 열 이름 찾기 (자기 자신 및 중복 제외)
high_corr = [(row, col) for row in corr_matrix.index for col in corr_matrix.columns 
             if corr_matrix.loc[row, col] == 1 and row < col]
drop_list = []
for col in high_corr:
    i,j = col
    drop_list.append(j)
    print(f'{i}와 {j}의 값이 같다')    
# 8FIC_P4260A.PV - Average와 8FIC_P4260B.PV - Average의 값이 같다
# 8FIC_P4230A.PV - Average와 8FIC_P4230B.PV - Average의 값이 같다
# 8TIC_E9340.PV - Average와 8TIC_E9340.PV - Average.1의 값이 같다
# 8FIC_F4270A.PV - Average와 8FIC_F4270B.PV - Average의 값이 같다

# 문자 위치 확인
for target in drop_list:
    if target in df_B1:
        print(f"'{target}'은(는) df_B1 리스트에 있습니다.")
    elif target in df_B2:
        print(f"'{target}'은(는) df_B2 리스트에 있습니다.")
    else:
        print(f"'{target}'은(는) df_B1나 df_B2 리스트에 없습니다.")
        
# 중복 열 제거
df_B = df_B.drop(drop_list, axis=1)
df_B

######################################## 열이름 변경

df_B.columns[df_B.isna().sum()>0] # 결측치 없음

# 모든 열 이름에서 ' - Average' 제거
df_B.columns = df_B.columns.str.replace(' - Average', '', regex=False)
df_B

# 열 이름을 첫 번째 단어로 그룹화하여 딕셔너리로 변환
grouped_dict = {}
for col in df_B.columns:
    key = col.split('_')[0]  # 첫 번째 단어 추출
    if key not in grouped_dict:
        grouped_dict[key] = []  # 키가 없으면 빈 리스트 생성
    grouped_dict[key].append(col)  # 열 이름 추가

print(grouped_dict.keys())
print("그룹화된 딕셔너리:")
for i in grouped_dict:
    print(f'{i}:{grouped_dict[i]}')


########################################## 단일 값 가지는 열 제거
df_B.columns[df_B.isna().sum()>0]
same_value_columns = df_B.columns[df_B.nunique()==1]
same_value_columns
df_B = df_B.drop(same_value_columns, axis=1)
df_B.shape

df_B.describe()

########################################### 이산형 열 제거
discrete_columns = list(df_B.columns[(df_B.nunique() <= 20) & (df_B.columns != 'LAB_8CHIP_L')])
discrete_columns
df_B = df_B.drop(discrete_columns, axis=1)

########################################## 시계열 그래프
# target값인 LAB_8CHIP_L의 시계열 그래프
df_B.columns
LAB_8CHIP_list = [col for col in df_B.columns if 'LAB_8CHIP' in col]
sns.lineplot(data=df_B, x='Timestamp', y='LAB_8CHIP_L')
plt.xticks(rotation=45)
# 날짜 확인
df_B.query('(LAB_8CHIP_L > 84.25) & (Timestamp > "2018-12-05")')['Timestamp'].max()

# 12/8일
defect_day = df_B.query('(Timestamp >= "2018-12-08") & (Timestamp < "2018-12-09")')
sns.lineplot(data=defect_day, x='Timestamp', y='LAB_8CHIP_L') 
plt.xticks(rotation=90)

########################################### ccf구하기
# 정상화
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
 '8TI_E9330D.PV', '8TI_E9330C.PV', '8TI_E9330B.PV', '8TI_E9330G.PV']

stationary = df_B.columns[~df_B.columns.isin(non_stationary)]
df_B_stationary = df_B[stationary]
for column in non_stationary:
    df_B_stationary[f'diff_{column}'] = df_B[column].diff()
df_B_stationary = df_B_stationary.dropna()
df_B_stationary.columns[df_B_stationary.isna().sum()>0]
df_B_stationary.isna().sum()>0
df_B_stationary

# 기간설정
defect_time = pd.to_datetime("2018-12-08 06:00:00")
start_date = df_B['Timestamp'].min()  # 전체 기간의 시작일
end_date = df_B['Timestamp'].max()    # 전체 기간의 종료일
a_week_ago = defect_time - pd.Timedelta(minutes=60*24*7) # 결함 7일 전
two_weeks_ago = defect_time - pd.Timedelta(minutes=60*24*7*2) # 결함 14일 전
sixteen_ago = defect_time - pd.Timedelta(minutes=60*24*16)
two_days_ago = defect_time - pd.Timedelta(minutes=60*24*2)

# 기간에 따른 데이터 프레임 나누기
df_B_a_week_ago = df_B_stationary[df_B_stationary['Timestamp']>= a_week_ago]
df_B_two_weeks_ago = df_B_stationary[df_B_stationary['Timestamp']>= two_weeks_ago]
df_B_sixteen_ago = df_B_stationary[df_B_stationary['Timestamp']>= sixteen_ago]
df_B_two_days_ago = df_B_stationary[df_B_stationary['Timestamp']>= two_days_ago]

# 단일 값 확인
df_B.nunique().sort_values()
df_B_a_week_ago.nunique().sort_values()
df_B_two_weeks_ago.nunique().sort_values()
df_B_sixteen_ago.nunique().sort_values()
df_B_two_days_ago.nunique().sort_values()

# df_B_a_week_ago에 단일값 존재
df_B[df_B['8II_P9120B.PV']==0] # 2018-11-27 15:42:00
sns.lineplot(data=df_B, x='Timestamp',y='8II_P9120B.PV')
sns.lineplot(data=df_B, x='Timestamp',y='LAB_8CHIP_L')
plt.xticks(rotation=90)
lag_8II_P9120B_PV = pd.Timestamp('2018-12-08 06:00:00') - pd.Timestamp('2018-11-27 15:42:00') 
lag_8II_P9120B_PV # 10 days 14:18:00

# ccf 계산을 위해 단일값 가지는 diff_8II_P9120B.PV열 제거
df_B_a_week_ago = df_B_a_week_ago.drop('diff_8II_P9120B.PV',axis=1)
df_B_stationary.columns[df_B_stationary.isna().sum()>0]
df_B_a_week_ago.columns[df_B_a_week_ago.isna().sum()>0]
df_B_two_weeks_ago.columns[df_B_two_weeks_ago.isna().sum()>0]
df_B_sixteen_ago.columns[df_B_sixteen_ago.isna().sum()>0]
df_B_two_days_ago.columns[df_B_two_days_ago.isna().sum()>0]

# df_B_two_days_ago에 diff_8II_P9120B.PV 단일값 존재
df_B_two_days_ago = df_B_two_days_ago.drop('diff_8II_P9120B.PV',axis=1)

from statsmodels.tsa.stattools import ccf

# 교차 상관계수 계산
def calculate_ccf(df):
    cross_corr_results = {}

    for col in df.columns[1:]:
        if col != 'LAB_8CHIP_L':
            cross_corr_values = ccf(df['LAB_8CHIP_L'], df[col], adjusted=False)
            cross_corr_results[col] = cross_corr_values

    # 결과를 데이터프레임으로 변환
    cross_corr_df = pd.DataFrame(cross_corr_results)

    # lag 값 추가
    cross_corr_df.index.name = 'lag'
    cross_corr_df = cross_corr_df.reset_index()

    # 각 열의 절대값 최대값과 해당 lag을 찾기
    max_abs_corr = cross_corr_df.abs().max()  # 절대값 최대값
    max_lags = cross_corr_df.abs().idxmax()   # 최대값의 인덱스(lag)
                    
    # 각 열의 절대값 최대값의 원래 값을 가져오기
    max_original_values = [cross_corr_df[col][idx] for col, idx in max_lags.items()]

    # 결과를 데이터프레임으로 정리
    result_df = pd.DataFrame({
        'max_abs_corr': max_abs_corr,
        'original_value': max_original_values,
        'lag': max_lags
    })

    # 절대값이 높은 순서로 정렬
    result_df = result_df.sort_values(by='max_abs_corr', ascending=False).reset_index()
    result_top30 = result_df[1:31]
    result_top10 = result_df[1:11]
    top_10_variable_lag = dict(zip(result_top30['index'], result_top30['lag']))
    top_10_variables = list(top_10_variable_lag.keys())
        
    plt.rcParams['font.family'] ='Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] =False
    for col in top_10_variables:
        sns.lineplot(data=cross_corr_df, x='lag', y=col)
        plt.ylabel('TOP 10 변수')  
        
    return cross_corr_df, result_top30

a = calculate_ccf(df_B_sixteen_ago)[0]
a_Test = a[['lag','8TI_P4350A.PV']]
sns.lineplot(data=a_Test,x='lag', y='8TI_P4350A.PV')
a_Test.sort_values('8TI_P4350A.PV',ascending=False).head(20)
######################################### 회귀분석으로 변수 유의성 확인
# 회귀분석을 위한 top변수에 lag 적용한 새로운 데이터 프레임 생성
import statsmodels.api as sm

def check_ols(df, ccf_top30):
    new_df = df[['Timestamp','LAB_8CHIP_L']]
    new_df

    for col, lag in zip(ccf_top30['index'], ccf_top30['lag']):
        new_df[f'{col}_lag{lag}'] = df[col].shift(lag)
        
    important_variables = []
    summarys = []
    
    # 새로운 데이터프레임으로 회귀
    for col in new_df.drop(['Timestamp','LAB_8CHIP_L'],axis=1).columns:
        X = new_df[col]
        X = sm.add_constant(X)  # statsmodels는 상수항을 포함하지 않기 때문에 추가해야 함
        y = new_df['LAB_8CHIP_L']

        # 회귀 모델 적합
        model = sm.OLS(y, X, missing='drop')
        results = model.fit()

        # 결과 요약 출력
        summarys.append(results.summary())

        # p-value를 기준으로 유의성 판단
        p_values = results.pvalues
        for variable, p_value in p_values.items():
            if (variable !='const') * (p_value < 0.05) :
                important_variables.append(variable)
    return important_variables, summarys

########################################### 통계검정 통과한 주요변수와 타켓변수 시계열 시각화

def create_graph(df, variable_lag):
    
    used_variables=[]
        
    for str in variable_lag:
        col, lag = str.split('_lag')
        lag = int(lag)
        
        if lag > 0:
            if len(used_variables) < 3:  # 3개까지만 추가
                used_variables.append(str)
            else:
                break  # used_variables가 3개 이상일 경우 반복 종료

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            plt.subplots_adjust(wspace=0.4)  # 서브플롯 간격 조정
            lag_time = defect_time - pd.Timedelta(minutes=lag)
            two_lag_ago = defect_time - pd.Timedelta(minutes=lag*2)

            # 전체 기간
            ax1 = axes[0]
            sns.lineplot(data=df, x='Timestamp', y=col, label=col, ax=ax1, color='b')
            ax2 = ax1.twinx()
            sns.lineplot(data=df, x='Timestamp', y='LAB_8CHIP_L', ax=ax2, color='orange')
            ax1.axvline(x=lag_time, color='red', linestyle='--', linewidth=1, label=f"lag{lag}")
            ax1.axvline(x=defect_time, color='black', linestyle='--', linewidth=1, label="Defect Time")
            ax1.set_xlim(start_date, end_date)
            ax1.set_title('전체 기간')
            ax1.tick_params(axis='x', rotation=90)
            ax1.set_ylabel(f'{col} 단위', color='b')
            ax2.set_ylabel('LAB_8CHIP_L 단위', color='orange')
            ax2.legend(['LAB_8CHIP_L'], loc='upper right')
            

            # 결함 16일 전
            ax1 = axes[1]
            sns.lineplot(data=df, x='Timestamp', y=col, label=col, ax=ax1, color='b')
            ax2 = ax1.twinx()
            sns.lineplot(data=df, x='Timestamp', y='LAB_8CHIP_L', ax=ax2, color='orange')
            ax1.axvline(x=lag_time, color='red', linestyle='--', linewidth=1, label=f"lag{lag}")
            ax1.axvline(x=defect_time, color='black', linestyle='--', linewidth=1, label="Defect Time")
            ax1.set_xlim(sixteen_ago, end_date)
            ax1.set_title('결함 16일 전')
            ax1.tick_params(axis='x', rotation=90)
            ax1.set_ylabel(f'{col} 단위', color='b')
            ax2.set_ylabel('LAB_8CHIP_L 단위', color='orange')
            ax2.legend(['LAB_8CHIP_L'], loc='upper right')

            # 결함 2일 전
            ax1 = axes[2]
            sns.lineplot(data=df, x='Timestamp', y=col, label=col, ax=ax1, color='b')
            ax2 = ax1.twinx()
            sns.lineplot(data=df, x='Timestamp', y='LAB_8CHIP_L', ax=ax2, color='orange')
            ax1.axvline(x=lag_time, color='red', linestyle='--', linewidth=1, label=f"lag{lag}")
            ax1.axvline(x=defect_time, color='black', linestyle='--', linewidth=1, label="Defect Time")
            ax1.set_xlim(two_days_ago, end_date)
            ax1.set_title('결함 2일 전')
            ax1.tick_params(axis='x', rotation=90)
            ax1.set_ylabel(f'{col} 단위', color='b')
            ax2.set_ylabel('LAB_8CHIP_L 단위', color='orange')
            ax2.legend(['LAB_8CHIP_L'], loc='upper right')

            # 범례 추가 (왼쪽 축에만 추가)
            axes[0].legend(loc='upper left')
            axes[1].legend(loc='upper left')
            axes[2].legend(loc='upper left')

            plt.tight_layout()
            plt.show()
            
    return used_variables

################################### 
 
def check_important_variables(df, df_part):
       df_ccf = calculate_ccf(df_part)
       df_ccf_ols = check_ols(df, df_ccf)
       df_final_variables = create_graph(df, df_ccf_ols[0])
       return df_ccf, df_ccf_ols, df_final_variables
   
sixteen_ago_ccf, sixteen_ago_ccf_ols, sixteen_ago_fianl_variables = check_important_variables(df_B, df_B_sixteen_ago)
# 상위 3개 변수 추출
sixteen_ago_fianl_variables

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def create_graph_one_axis(variables_list, start_time):
    # 1행 3열의 서브플롯 생성
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(wspace=0.4)  # 서브플롯 간격 조정

    # variables_list의 첫 3개 변수에 대해 서브플롯에 각각 그리기
    for i, str in enumerate(variables_list):
        col, lag = str.split('_lag')
        lag = int(lag)
        lag_time = defect_time - pd.Timedelta(minutes=lag)

        # 변수별 전체 기간 그래프
        sns.lineplot(data=df_B, x='Timestamp', y=col, ax=axes[i], label=col, color='b')
        sns.lineplot(data=df_B, x='Timestamp', y='LAB_8CHIP_L', ax=axes[i], label='LAB_8CHIP_L', color='orange')
        axes[i].axvline(x=lag_time, color='red', linestyle='--', linewidth=1, label=f"lag{lag}")
        axes[i].axvline(x=defect_time, color='black', linestyle='--', linewidth=1, label="Defect Time")
        axes[i].tick_params(axis='x', rotation=90)
        axes[i].set_xlim(start_time, end_date)
        axes[i].set_title(f'{col} 전체 기간')
        axes[i].legend()

    # 레이아웃 조정
    plt.tight_layout()
    plt.show()

create_graph_one_axis(sixteen_ago_fianl_variables, start_date)     

