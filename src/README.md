[src] 폴더 설명
---------------------------------
용도: 노트북과 Dash 앱(app/) 양쪽에서 "똑같이" 써야 하는 파이썬
함수를 모아두는 곳입니다.

왜 필요한가:
- 만약 전처리 코드를 노트북에만 써두면, Dash 쪽에서 새 CSV를
  업로드했을 때 노트북과 "다른 방식"으로 전처리하게 될 위험이
  있습니다. 학습할 때와 실전(앱)에서 전처리가 달라지면 모델이
  이상하게 작동합니다. 이걸 막기 위해 공용 함수를 이 폴더에
  한 번만 작성하고, 양쪽에서 똑같이 불러다 씁니다.

파일:
- features.py     : 파생변수 생성 함수 (temp_diff, power_w 등).
                     노트북과 app/pages/data_lookup.py가 같이 사용
- validate.py      : 업로드된 CSV의 필수 컬럼·이상값을 검사하는
                     함수. app/pages/data_lookup.py에서 사용
- energy_train.py  : 에너지 회귀·Prophet 모델을 학습하고
                     outputs/energy_*.csv를 생성하는 스크립트
- ai4i_train.py    : AI4I 분류(DT/RF)·K-Means를 학습하고
                     outputs/ai4i_*.csv와 models/rf.joblib을
                     생성하는 스크립트

사용법 예시:
  from src.features import make_features
  from src.validate import validate_ai4i_csv
