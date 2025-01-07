import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from Battery.secrets import show_heatmapFile

# CSV 파일 읽기
file_path = f"../PostProcess/11column4sample/{show_heatmapFile}"  # CSV 파일 경로
df = pd.read_csv(file_path)

columns_to_exclude = ['Unnamed: 0', 'Relative']
filtered_df = df.drop(columns=columns_to_exclude, errors='ignore')


# 상관계수 계산
corr = filtered_df.corr()

# 히트맵 그리기
plt.figure(figsize=(10, 8))  # 그래프 크기 조정
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()
