# 일별 Top 3 읽기용 결과 (A2, 12점 기준)

원본 `asset_risk_predictions.csv`는 약 87MB라 앱 미리보기에서 열리지 않을 수 있어,
A2 모델·12점 기준만 추려 장비·날짜당 한 행으로 저장한 파일입니다.

- 행 수: **10,950개**
- 기간: **2022-01-03 ~ 2025-01-01**
- 장비 수: **10개**
- `top3_probability_parts`: 고장확률 상위 3개
- `top3_contribution_parts`: 중요도 가중 점수 기여도 상위 3개

## 기계별 요약

| machine_type | 장비일수 | 평균예상점수 | 평균_12점이상확률 | 고위험예측일수 |
| --- | --- | --- | --- | --- |
| Belt Conveyor | 2190 | 4.5547 | 0.0882 | 4 |
| CNC Lathe | 2190 | 4.8408 | 0.1041 | 14 |
| EOT Crane | 2190 | 4.9572 | 0.1138 | 10 |
| Hydraulic Press | 2190 | 5.7461 | 0.1600 | 13 |
| Screw Compressor | 2190 | 5.3920 | 0.1379 | 8 |

## 앞쪽 10개 기록

| transaction_date | asset_tag | machine_type | expected_failure_points | prob_ge_12 | actual_severity | predicted_severity | top3_probability_parts | top3_contribution_parts | top3_display_label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022-01-03 | AST-1041 | CNC Lathe | 6.8330 | 0.1460 | risk | risk | MRO-10045;MRO-10046;MRO-10047 | MRO-10045;MRO-10046;MRO-10047 | 우선 점검 후보 |
| 2022-01-03 | AST-1042 | CNC Lathe | 6.8102 | 0.1458 | high_risk | risk | MRO-10045;MRO-10046;MRO-10047 | MRO-10045;MRO-10046;MRO-10047 | 우선 점검 후보 |
| 2022-01-03 | AST-2017 | Hydraulic Press | 9.0711 | 0.2989 | high_risk | risk | MRO-10045;MRO-10046;MRO-10047 | MRO-10045;MRO-10046;MRO-10047 | 우선 점검 후보 |
| 2022-01-03 | AST-2031 | Hydraulic Press | 7.9188 | 0.2157 | risk | risk | MRO-80001;MRO-10046;MRO-10047 | MRO-80001;MRO-10046;MRO-10047 | 우선 점검 후보 |
| 2022-01-03 | AST-3008 | Belt Conveyor | 6.4603 | 0.1258 | caution | risk | MRO-10045;MRO-10046;MRO-10047 | MRO-10045;MRO-10046;MRO-10047 | 우선 점검 후보 |
| 2022-01-03 | AST-3019 | Belt Conveyor | 5.9001 | 0.0977 | caution | risk | MRO-10045;MRO-10046;MRO-10047 | MRO-10045;MRO-10046;MRO-10047 | 우선 점검 후보 |
| 2022-01-03 | AST-4055 | Screw Compressor | 8.0512 | 0.2223 | risk | risk | MRO-10045;MRO-10046;MRO-10047 | MRO-10045;MRO-10046;MRO-10047 | 우선 점검 후보 |
| 2022-01-03 | AST-4061 | Screw Compressor | 8.8601 | 0.2815 | high_risk | risk | MRO-10046;MRO-10045;MRO-10047 | MRO-10046;MRO-10045;MRO-10047 | 우선 점검 후보 |
| 2022-01-03 | AST-5003 | EOT Crane | 9.4758 | 0.3254 | risk | risk | MRO-20033;MRO-10045;MRO-10046 | MRO-10045;MRO-10046;MRO-10047 | 우선 점검 후보 |
| 2022-01-03 | AST-5007 | EOT Crane | 5.6488 | 0.0830 | caution | caution | MRO-10045;MRO-10046;MRO-10047 | MRO-10045;MRO-10046;MRO-10047 | 우선 점검 후보 |

## 마지막 10개 기록

| transaction_date | asset_tag | machine_type | expected_failure_points | prob_ge_12 | actual_severity | predicted_severity | top3_probability_parts | top3_contribution_parts | top3_display_label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025-01-01 | AST-1041 | CNC Lathe | 7.6350 | 0.1930 | risk | risk | MRO-20031;MRO-10045;MRO-10047 | MRO-20031;MRO-10045;MRO-10047 | 우선 점검 후보 |
| 2025-01-01 | AST-1042 | CNC Lathe | 7.2612 | 0.1713 | risk | risk | MRO-40022;MRO-10045;MRO-20031 | MRO-10045;MRO-20031;MRO-80001 | 우선 점검 후보 |
| 2025-01-01 | AST-2017 | Hydraulic Press | 7.3052 | 0.1712 | risk | risk | MRO-60016;MRO-80001;MRO-50009 | MRO-80001;MRO-50009;MRO-20031 | 우선 점검 후보 |
| 2025-01-01 | AST-2031 | Hydraulic Press | 7.1038 | 0.1640 | high_risk | risk | MRO-50009;MRO-10045;MRO-20031 | MRO-50009;MRO-10045;MRO-20031 | 우선 점검 후보 |
| 2025-01-01 | AST-3008 | Belt Conveyor | 7.0400 | 0.1582 | risk | risk | MRO-10047;MRO-10046;MRO-20031 | MRO-10047;MRO-10046;MRO-20031 | 우선 점검 후보 |
| 2025-01-01 | AST-3019 | Belt Conveyor | 6.1667 | 0.1101 | risk | risk | MRO-10046;MRO-30012;MRO-50009 | MRO-10046;MRO-50009;MRO-80001 | 우선 점검 후보 |
| 2025-01-01 | AST-4055 | Screw Compressor | 9.5160 | 0.3260 | risk | risk | MRO-20032;MRO-10046;MRO-50009 | MRO-10046;MRO-50009;MRO-80001 | 우선 점검 후보 |
| 2025-01-01 | AST-4061 | Screw Compressor | 7.7950 | 0.2025 | risk | risk | MRO-10046;MRO-10047;MRO-30011 | MRO-10046;MRO-10047;MRO-80001 | 우선 점검 후보 |
| 2025-01-01 | AST-5003 | EOT Crane | 8.2635 | 0.2372 | risk | risk | MRO-10046;MRO-10045;MRO-10047 | MRO-10046;MRO-10045;MRO-10047 | 우선 점검 후보 |
| 2025-01-01 | AST-5007 | EOT Crane | 6.7540 | 0.1417 | risk | risk | MRO-10045;MRO-50009;MRO-10046 | MRO-10045;MRO-50009;MRO-10046 | 우선 점검 후보 |

주의: Top 3는 점검 우선순위 후보이며 실제 고장 부품 확정 결과가 아닙니다.
