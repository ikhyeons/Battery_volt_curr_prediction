import pandas as pd
import numpy as np
from Battery.secrets import col_list, col_list_without_volt, columns_to_normalize, drop_column
from Battery.secrets import volt_column, curr_column, press_column

def data_post_process(df, outputname):
    result = df.copy()
    for dc in drop_column:
        result = result.drop(columns=[dc])

    # 컬럼명 양쪽 공백 제거
    result.columns = result.columns.str.strip()
    # 데이터를 모두 숫자로 취급
    result = result.apply(pd.to_numeric, errors='coerce')
    # 데이터의 모든 공백 값을 np.nan으로 수정
    result.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    # 양 끝에 값 생성(volt만 제외)
    result = create_start_and_end(result)
    # spline 보간(volt만 제외)
    result = interpolation_nan(result)
    # volt에 값이 없는 행 제거
    result = result.dropna()
    # 전압, 전류, 압력 데이터 정규화
    result[columns_to_normalize] = result[columns_to_normalize].apply(
        lambda x: (x - x.min()) / (x.max() - x.min()))
    # t-1시점 데이터열 생성 (만약 모든값이 비어있을 경우 모든 데이터가 날라갈 수 있음)
    result = create_t_minus_1(result)
    # 전압, 전류 이전시간 변화량 생성
    # result = create_delta_value(result) // 성능 저하로 일단 제외
    result.to_csv(f"./11column/{outputname}.csv")


def create_delta_value(df):
    print("create delta value")
    result = df.copy()
    result['voltage_diff'] = result[volt_column].shift(1) - result[volt_column].shift(2)
    result['current_diff'] = result[curr_column].shift(1) - result[curr_column].shift(2)
    result = result.dropna()
    return result


def create_t_minus_1(df):
    print("create -1")
    result = df.copy()
    result['S3PDU-AVOLT_t_minus_1'] = result[volt_column].shift(1)
    result['S3PDU-ACURR_t_minus_1'] = result[curr_column].shift(1)
    result = result.dropna()
    return result


def interpolation_nan(df):
    print("Start to interpolate")
    result = df.copy()

    for column in col_list:
        if column == volt_column:
            None
        elif column == curr_column or column == press_column:
            print(f"interpolate {column}")
            result[column] = result[column].interpolate()
        else:
            print(f"fill {column}")
            result[column] = result[column].ffill()

    return result


def create_start_and_end(df):
    print("Start to create first and last value")
    result = df.copy()
    for column in col_list_without_volt:
        # 컬럼에서 데이터를 순회하면서 제일 처음 만난 데이터가 nan이 아닐 경우 맨 앞 데이터를 해당 값으로 변경
        for value in result[column]:
            if not np.isnan(value):
                result.loc[0, column] = value
                break

        # 컬럼에서 데이터를 역순회하면서 제일 처음 만난 데이터가 nan이 아닐 경우 맨 뒤 데이터를 해당 값으로 변경
        for value in result[column].iloc[::-1]:
            if not np.isnan(value):
                result.loc[result.index[-1], column] = value
                break
    return result


def create_post_processed_data(file_name):
    row_data = pd.read_csv(f'../2rawData/{file_name}.csv', sep=';', dtype=str)
    data_post_process(row_data, f"{file_name}")


# 전처리할 파일 이름
# 전처리할 파일은 2rawData폴더 안에 있어야 함
fileName = "s_ikhyeon_nor"
create_post_processed_data(fileName)

