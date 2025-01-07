import pandas as pd
from Battery.secrets import samplingFile

# CSV 파일 읽기
file_path = f"11column/{samplingFile}"  # 파일 경로를 입력하세요.
data = pd.read_csv(file_path)

# 4개 행마다 1개씩 샘플링
sampled_data = data.iloc[::4]  # 매 4번째 행 선택

# 샘플링된 데이터 저장
output_path = f"11column4sample/{samplingFile}"
sampled_data.to_csv(output_path, index=False)