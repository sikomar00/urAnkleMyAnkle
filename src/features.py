"""
파생변수 생성 함수 모음.

노트북(notebooks/ai4i_model.ipynb)과 Dash 앱
(app/pages/data_lookup.py) 양쪽에서 똑같이 이 함수를 불러다
씁니다. 학습 때와 앱에서 전처리 방식이 달라지지 않도록 하기
위함입니다 (자세한 이유는 src/폴더설명.txt 참고).
"""
import numpy as np
import pandas as pd


def make_ai4i_features(df: pd.DataFrame) -> pd.DataFrame:
    """AI4I 원본 데이터프레임에 파생변수를 추가해 반환한다.

    TODO:
    - temp_diff = process_k - air_k
    - power_w = torque * rpm * 2 * pi / 60
    - strain = tool_wear * torque
    """
    raise NotImplementedError("TODO: 파생변수 로직 구현")
