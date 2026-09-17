[notebooks] 폴더 설명
---------------------------------
용도: 데이터를 탐색하고 실험하는 주피터 노트북 폴더입니다.

규칙:
- energy_eda.ipynb 는 에너지 트랙, ai4i_model.ipynb 는 AI4I 트랙
  전용입니다. 두 사람이 같은 노트북 파일을 동시에 수정하면
  Git에서 거의 항상 충돌이 나니, 절대 파일을 합치지 마세요.
- 노트북은 "탐색용"입니다. 여러 사람이 같이 쓸 함수(전처리,
  검증 로직 등)는 노트북에 남기지 말고 src/ 로 옮겨서
  `from src.features import ...` 형태로 불러다 쓰세요.
- 노트북을 열기 전에 `nbstripout --install`을 꼭 한 번
  실행해두세요 (가이드.txt 4번 항목 참고). 실행 결과(출력, 그림)가
  Git에 남지 않게 해줘서 병합 충돌을 크게 줄여줍니다.

파일:
- energy_eda.ipynb : 전력·기상 데이터 전처리, EDA, KPI, 회귀,
  Prophet 예측 실험
- ai4i_model.ipynb : AI4I 데이터 전처리, 분류 모델(DT/RF) 비교,
  K-Means 이상탐지 실험
