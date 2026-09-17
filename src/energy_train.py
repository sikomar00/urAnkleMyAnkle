"""
에너지 트랙: 전처리 → 회귀(기준 모델) → Prophet 예측 → 결과 저장.

실행:
    python src/energy_train.py

생성되는 파일:
    outputs/energy_forecast.csv
    outputs/energy_scores.csv
"""

def main():
    # TODO 1. data/raw/의 전력·기상 원본 로드
    # TODO 2. 일 단위 집계, 결측/중복/범위 검사
    # TODO 3. 시간순으로 train/test 분할 (섞지 않음)
    # TODO 4. 기준 모델(baseline) 예측
    # TODO 5. Prophet 학습·예측
    # TODO 6. outputs/energy_forecast.csv, energy_scores.csv 저장
    #         (컬럼 구조는 docs/data_contract.md 참고)
    raise NotImplementedError("TODO: 학습 파이프라인 구현")


if __name__ == "__main__":
    main()
