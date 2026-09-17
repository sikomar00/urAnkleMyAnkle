"""
업로드된 CSV를 검증하는 함수.

app/pages/data_lookup.py 의 CSV 업로드 기능에서 사용합니다.
"""
import pandas as pd

REQUIRED_COLUMNS = {
    "type", "air_k", "process_k", "rpm", "torque", "tool_wear",
}


def validate_ai4i_csv(df: pd.DataFrame) -> list[str]:
    """검증 후 문제점 목록(빈 리스트면 문제 없음)을 반환한다.

    TODO:
    - 필수 컬럼 누락 확인
    - 중복 행 확인
    - 범위를 벗어난 값 확인 (예: rpm <= 0)
    """
    raise NotImplementedError("TODO: 검증 로직 구현")
