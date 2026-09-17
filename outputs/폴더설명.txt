[outputs] 폴더 설명 (가장 중요한 폴더 중 하나)
---------------------------------
용도: 분석 결과를 저장하는 곳이자, Dash 앱이 "읽기만" 하는
"데이터 계약(contract)" 파일들이 모이는 곳입니다.

핵심 규칙:
- 여기 있는 파일들의 이름과 컬럼 구조는 docs/data_contract.md에
  이미 정해져 있습니다. 컬럼을 추가/변경하고 싶으면 반드시
  data_contract.md를 먼저 수정하고 팀 전체에 공유한 다음
  바꾸세요. 안 그러면 Dash 코드가 갑자기 깨집니다.
- 지금 이 폴더의 csv 파일들은 전부 "더미(가짜) 데이터"입니다.
  Dash 담당이 실제 모델 결과를 기다리지 않고 먼저 화면을
  만들 수 있게 하기 위한 것입니다.
- 실제 모델을 다 돌리고 나면, 같은 파일명 · 같은 컬럼 구조를
  유지한 채로 진짜 결과로 "덮어쓰기"만 하면 됩니다.
  (src/energy_train.py, src/ai4i_train.py 실행 시 자동 생성됨)

파일 (상세 컬럼은 docs/data_contract.md 참고):
- energy_forecast.csv : 실제값·예측값·예측구간 (Prophet 결과)
- energy_scores.csv   : 모델별 MAE·RMSE·R2 비교표
- ai4i_metrics.csv    : 모델별 Precision·Recall·F1·혼동행렬
- ai4i_scored.csv     : 개별 데이터에 대한 고장확률·군집·이상여부
