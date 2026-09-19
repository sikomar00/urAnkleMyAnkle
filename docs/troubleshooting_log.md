# 트러블슈팅 로그

막혔던 문제가 생길 때마다 기록하세요. 이 로그가 포트폴리오의
"트러블슈팅" 섹션(평가자가 가장 유심히 보는 항목)의 재료가 됩니다.

## 이슈 템플릿 (복사해서 사용)

### 이슈: (제목)
- 문제 상황:
- 원인 분석:
- 시도한 것들:
- AI 제안을 받았다면 — 제안 내용 / 신뢰했는지·수정했는지:
- 결과 (Before → After):
- 배운 점:


### 이슈: Prophet이 pip 설치 후 fit()에서 AttributeError 발생
- 문제 상황: `pip install -r requirements.txt` 후 Prophet().fit()이
  `AttributeError: 'Prophet' object has no attribute 'stan_backend'`로 실패
- 원인 분석: prophet==1.1.6은 wheel에 cmdstan 2.33.1을 번들하지만
  makefile은 포함하지 않음. cmdstanpy 1.3.0부터 cmdstan 경로 검증 시
  makefile 존재를 요구하도록 바뀌어서 버전 미스매치 발생 (1.2.5→1.3.0
  사이 변경, 중간 릴리스 없음). requirements.txt에 cmdstanpy 버전을
  안 고정해서 최신이 설치된 게 근본 원인
- 시도한 것들: (1) cmdstanpy.install_cmdstan() — SSL 인증서 에러,
  디스크 쿼터 에러로 실패 (2) conda-forge로 우회 설치 — 성공했지만
  플랫폼마다 다른 절차 필요 (3) prophet 소스 코드(models.py,
  forecaster.py) 직접 읽어 예외가 어디서 삼켜지는지 추적 →
  cmdstanpy.set_cmdstan_path()의 validate_cmdstan_path()에서
  ValueError 발생 확인
- AI 제안을 받았다면: Claude가 처음엔 "wheel에 번들돼 있으니
  install_cmdstan 불필요"라고 예측했으나 실제 테스트에서 틀림
  → 소스 코드 직접 읽어서 진짜 원인(cmdstanpy 버전 검증 로직 변경)
  찾음. AI 가설을 검증 없이 믿지 않고 실제로 재현·확인한 사례
- 결과 (Before → After): conda 우회(플랫폼별 절차 필요) →
  requirements.txt에 `cmdstanpy==1.2.5` 한 줄 고정으로 Mac/Windows
  공통 해결
- 배운 점: 에러 메시지가 안내하는 해결책(install_cmdstan)이 항상
  올바른 처방은 아님. 버전 미스매치가 원인일 때는 오히려 하위
  호환 버전 고정이 정답