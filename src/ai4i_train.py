"""
AI4I 트랙: 전처리 → 분류(DT/RF) → K-Means 이상탐지 → 결과 저장.

실행:
    python src/ai4i_train.py

생성되는 파일:
    outputs/ai4i_metrics.csv
    outputs/ai4i_scored.csv
    models/rf.joblib
"""

def main():
    # TODO 1. data/raw/의 AI4I 원본 로드
    # TODO 2. src.features.make_ai4i_features로 파생변수 생성
    # TODO 3. UDI, Product ID, TWF~RNF 제외 근거 확인 후 제외
    # TODO 4. stratify 분할
    # TODO 5. baseline / DecisionTree / RandomForest 학습·평가
    # TODO 6. K-Means 군집화 + 거리 기반 이상치 탐지
    # TODO 7. outputs/ai4i_metrics.csv, ai4i_scored.csv 저장
    #         models/rf.joblib 저장
    #         (컬럼 구조는 docs/data_contract.md 참고)
    raise NotImplementedError("TODO: 학습 파이프라인 구현")


if __name__ == "__main__":
    main()
