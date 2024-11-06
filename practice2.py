
#!pip install --upgrade matplotlib
from statsmodels.tsa.stattools import ccf
# 

# 교차 상관 분석
# 예를 들어, '8PIC_D4210.PV - Average'와 '8DIC_P4240.PV - Average'의 교차 상관 분석


# 교차 상관 계산


B.columns
series1 = B['LAB_8CHIP_L'] # target
series2 = B['8DIC_P4240.PV']
cross_corr = ccf(series1, series2)   # 현재 series1와 과거 series2의 상관 관계를 구한걸 확인함

for lag, value in enumerate(cross_corr):
    print(f'Lag {lag}: {value:.4f}')  # 소수점 4자리까지 출력


# 교차 상관 시각화
plt.figure(figsize=(10, 5))
plt.stem(range(len(cross_corr)), cross_corr)  
plt.title(f'Cross-Correlation between {series1} and {series2}')
plt.xlabel('Lag')
plt.ylabel('Cross-Correlation')
plt.grid()
plt.show()





series1 = B['LAB_8CHIP_L'] # target
series2 = B['8DIC_P4240.PV']
cross_corr = ccf(series2, series1)   #과거 series2를 이용해서 현재 series1을 구한걸 확인함

for lag, value in enumerate(cross_corr):
    print(f'Lag {lag}: {value:.4f}')  # 소수점 4자리까지 출력


# 교차 상관 시각화
plt.figure(figsize=(10, 5))
plt.stem(range(len(cross_corr)), cross_corr)  
plt.title(f'Cross-Correlation between {series1} and {series2}')
plt.xlabel('Lag')
plt.ylabel('Cross-Correlation')
plt.grid()
plt.show()





# lag 1일 때 사용된 데이터 행 출력 (과거 series2를 이용해서 현재 series1을 구한걸 확인함)
lag = 1
for i in range(lag, len(B)):  # lag부터 시작
    print(f'Row for Lag {lag} - series1: {B["LAB_8CHIP_L"].iloc[i]}, series2: {B["8DIC_P4240.PV"].iloc[i-lag]}')

B[["LAB_8CHIP_L", "8DIC_P4240.PV"]]

1.2909823060035706
1.2910899718602498






# 이때의 회귀 분석은 타켓 변수와 lag 시간차를 고려한 (과거값을 끌고온) 설명변수를 회귀분석함
# 과거값을 끌고오면서 lag 시간차 만큼 첫 행 부분은 nan 값이 생기는데
# 전체 행 4만 중 nan값이 6천 정도라면 제거하고 돌리고 (+ 첫 행 부분은 과거 정보라서 별로 안 중요하다고 생각하면)
# 첫 행 부분이 중요하고, 전체 데이터 중에서 너무 많은 부분을 차지하고 있으면, 보간을 하던가, 평균값으로 대체함





B_index2 = B_index.asfreq('T')
B_index2.index.freq

B_index2.index = pd.date_range(start=B_index2.index[0], periods=len(B_index2), freq='T')

B_index2.index.freq = 'T'


B_index2.index = pd.date_range(start=B_index2.index[0], periods=len(B_index2), freq='T')
B_index2.index.freq = pd.tseries.frequencies.Frequency('T')  # 분(minute) 단위로 명시적으로 설정

import statsmodels.api as sm
# 시계열 분해 (additive 또는 multiplicative)
decomposition = sm.tsa.seasonal_decompose(B_index['8PIC_D4210.PV'], model='additive')  # 에러
decomposition = sm.tsa.seasonal_decompose(B_index2['8PIC_D4210.PV'], model='additive')

# 결과 플롯
fig = decomposition.plot()
plt.show()

