"""
파생변수 생성 함수 모음.

노트북(notebooks/ai4i_model.ipynb)과 Dash 앱
(app/pages/data_lookup.py) 양쪽에서 똑같이 이 함수를 불러다
씁니다. 학습 때와 앱에서 전처리 방식이 달라지지 않도록 하기
위함입니다 (자세한 이유는 src/README.md 참고).
"""
import numpy as np
import pandas as pd

# AI4I 원본 CSV의 컬럼명 -> 프로젝트 표준 snake_case
#
# 원본을 그대로 읽으면 컬럼명에 공백·대괄호가 들어 있어서
# df.air_k 같은 접근이 안 되고, src/validate.py의 REQUIRED_COLUMNS
# 와도 맞지 않습니다. 로드 직후 한 번만 적용하세요:
#     df = pd.read_csv(path).rename(columns=COLUMN_RENAME_MAP)
COLUMN_RENAME_MAP = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "type",
    "Air temperature [K]": "air_k",
    "Process temperature [K]": "process_k",
    "Rotational speed [rpm]": "rpm",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "tool_wear",
    "Machine failure": "failure",
    "TWF": "twf", "HDF": "hdf", "PWF": "pwf",
    "OSF": "osf", "RNF": "rnf",
}

# 모델 입력에서 제외할 컬럼과 그 근거 (발표에서 설명할 항목)
#   udi, product_id : 단순 식별자. 학습하면 행 번호를 외우는 꼴
#   twf~rnf         : 고장의 '원인 유형' = 타깃이 정해진 뒤에야 알 수 있음
#                     -> 누수(leakage). 반드시 제외
LEAKAGE_COLUMNS = ["udi", "product_id", "twf", "hdf", "pwf", "osf", "rnf"]


def make_ai4i_features(df: pd.DataFrame) -> pd.DataFrame:
    """AI4I 데이터프레임에 파생변수를 추가해 **새 객체로** 반환한다.

    입력은 COLUMN_RENAME_MAP 적용 후 상태를 가정한다.

    TODO:
    - temp_diff = process_k - air_k
    - power_w = torque * rpm * 2 * pi / 60
    - strain = tool_wear * torque
    - 원본 df를 수정하지 말고 df.copy()로 시작할 것
    """
    raise NotImplementedError("TODO: 파생변수 로직 구현")
