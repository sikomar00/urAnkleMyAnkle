# 산업 장비 당일 Family 이상·심각 진단 결과

## 세 줄 요약

1. 장비·날짜 센서로 9개 Family의 `affected`와 `severe`를 독립 진단했습니다.
2. 성능은 Accuracy만이 아니라 Family별 Average Precision과 다중라벨 Macro·Micro 지표로 판단합니다.
3. 순차 모델 판정은 **진행 보류**이며, 높은 Precision 정책 불가 항목은 24건입니다.

## 한 페이지 요약

이 결과는 당일 센서와 과거 이력으로 어느 Family를 점검할지 범위를 줄이는 실험입니다. `affected`는 구성품 하나 이상 표시, `severe`는 두 개 이상 또는 A등급 부품 표시입니다. 출력은 Family 점검 우선순위이며 실제 장비 고장 확정이 아닙니다. 모델과 임계값은 검증 구간에서만 선택했고 테스트 구간은 최종 평가에만 사용했습니다. 낮은 성능과 사용할 수 없는 Precision 정책도 그대로 기록합니다.

## 1. Family별 발생률

| period | part_family | row_count | affected_rate | severe_rate | all_failed_rate |
| --- | --- | --- | --- | --- | --- |
| overall | Bearing | 10950 | 0.3177 | 0.3177 | 0.0053 |
| overall | Seal & Gasket | 10950 | 0.2510 | 0.1322 | 0.0017 |
| overall | Drive Belt | 10950 | 0.1805 | 0.0143 | 0.0143 |
| overall | Filter | 10950 | 0.1824 | 0.0127 | 0.0127 |
| overall | Electrical | 10950 | 0.2046 | 0.1272 | 0.0162 |
| overall | Coupling | 10950 | 0.1712 | 0.0102 | 0.0102 |
| overall | Lubrication | 10950 | 0.1348 | 0.0061 | 0.0061 |
| overall | Sensor | 10950 | 0.2065 | 0.1277 | 0.0176 |
| overall | Fastener | 10950 | 0.1399 | 0.0086 | 0.0086 |
| train | Bearing | 7280 | 0.3144 | 0.3144 | 0.0059 |
| train | Seal & Gasket | 7280 | 0.2536 | 0.1312 | 0.0015 |
| train | Drive Belt | 7280 | 0.1738 | 0.0141 | 0.0141 |
| train | Filter | 7280 | 0.1870 | 0.0126 | 0.0126 |
| train | Electrical | 7280 | 0.2036 | 0.1253 | 0.0170 |
| train | Coupling | 7280 | 0.1710 | 0.0102 | 0.0102 |
| train | Lubrication | 7280 | 0.1338 | 0.0069 | 0.0069 |
| train | Sensor | 7280 | 0.2052 | 0.1275 | 0.0176 |
| train | Fastener | 7280 | 0.1380 | 0.0084 | 0.0084 |
| valid | Bearing | 1820 | 0.3165 | 0.3165 | 0.0038 |
| valid | Seal & Gasket | 1820 | 0.2495 | 0.1324 | 0.0022 |
| valid | Drive Belt | 1820 | 0.1901 | 0.0132 | 0.0132 |
| valid | Filter | 1820 | 0.1747 | 0.0115 | 0.0115 |
| valid | Electrical | 1820 | 0.2154 | 0.1368 | 0.0143 |
| valid | Coupling | 1820 | 0.1714 | 0.0110 | 0.0110 |
| valid | Lubrication | 1820 | 0.1335 | 0.0060 | 0.0060 |
| valid | Sensor | 1820 | 0.2022 | 0.1225 | 0.0209 |
| valid | Fastener | 1820 | 0.1451 | 0.0077 | 0.0077 |
| test | Bearing | 1850 | 0.3319 | 0.3319 | 0.0043 |
| test | Seal & Gasket | 1850 | 0.2422 | 0.1362 | 0.0022 |
| test | Drive Belt | 1850 | 0.1978 | 0.0162 | 0.0162 |
| test | Filter | 1850 | 0.1719 | 0.0141 | 0.0141 |
| test | Electrical | 1850 | 0.1978 | 0.1254 | 0.0146 |
| test | Coupling | 1850 | 0.1719 | 0.0097 | 0.0097 |
| test | Lubrication | 1850 | 0.1400 | 0.0032 | 0.0032 |
| test | Sensor | 1850 | 0.2157 | 0.1335 | 0.0146 |
| test | Fastener | 1850 | 0.1422 | 0.0103 | 0.0103 |

## 2. A·B·C 비교

| part_family | target | feature_set | model | average_precision | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Bearing | affected | A | logistic_regression | 0.4989 | 0.4526 | 0.9618 | 0.6156 |
| Bearing | affected | A | random_forest | 0.4974 | 0.4454 | 0.9983 | 0.6160 |
| Bearing | affected | A | hist_gradient_boosting | 0.4892 | 0.4434 | 1.0000 | 0.6144 |
| Bearing | affected | B | logistic_regression | 0.5136 | 0.4470 | 0.9878 | 0.6155 |
| Bearing | affected | B | random_forest | 0.4622 | 0.4444 | 0.9983 | 0.6150 |
| Bearing | affected | B | hist_gradient_boosting | 0.4840 | 0.4434 | 1.0000 | 0.6144 |
| Bearing | affected | C | logistic_regression | 0.5042 | 0.4441 | 0.9931 | 0.6137 |
| Bearing | affected | C | random_forest | 0.4747 | 0.4429 | 0.9965 | 0.6132 |
| Bearing | affected | C | hist_gradient_boosting | 0.4743 | 0.4440 | 0.9983 | 0.6146 |
| Bearing | severe | A | logistic_regression | 0.4989 | 0.4526 | 0.9618 | 0.6156 |
| Bearing | severe | A | random_forest | 0.4974 | 0.4454 | 0.9983 | 0.6160 |
| Bearing | severe | A | hist_gradient_boosting | 0.4892 | 0.4434 | 1.0000 | 0.6144 |
| Bearing | severe | B | logistic_regression | 0.5136 | 0.4470 | 0.9878 | 0.6155 |
| Bearing | severe | B | random_forest | 0.4622 | 0.4444 | 0.9983 | 0.6150 |
| Bearing | severe | B | hist_gradient_boosting | 0.4840 | 0.4434 | 1.0000 | 0.6144 |
| Bearing | severe | C | logistic_regression | 0.5042 | 0.4441 | 0.9931 | 0.6137 |
| Bearing | severe | C | random_forest | 0.4747 | 0.4429 | 0.9965 | 0.6132 |
| Bearing | severe | C | hist_gradient_boosting | 0.4743 | 0.4440 | 0.9983 | 0.6146 |
| Seal & Gasket | affected | A | logistic_regression | 0.3726 | 0.3596 | 0.9449 | 0.5209 |
| Seal & Gasket | affected | A | random_forest | 0.3725 | 0.3524 | 0.9890 | 0.5197 |
| Seal & Gasket | affected | A | hist_gradient_boosting | 0.3523 | 0.3510 | 0.9912 | 0.5184 |
| Seal & Gasket | affected | B | logistic_regression | 0.3788 | 0.3484 | 1.0000 | 0.5168 |
| Seal & Gasket | affected | B | random_forest | 0.3604 | 0.3490 | 1.0000 | 0.5174 |
| Seal & Gasket | affected | B | hist_gradient_boosting | 0.3688 | 0.3495 | 0.9978 | 0.5177 |
| Seal & Gasket | affected | C | logistic_regression | 0.3706 | 0.3491 | 0.9934 | 0.5166 |
| Seal & Gasket | affected | C | random_forest | 0.3585 | 0.3461 | 0.9956 | 0.5136 |
| Seal & Gasket | affected | C | hist_gradient_boosting | 0.3741 | 0.3509 | 0.9956 | 0.5189 |
| Seal & Gasket | severe | A | logistic_regression | 0.1970 | 0.1970 | 0.8838 | 0.3222 |
| Seal & Gasket | severe | A | random_forest | 0.2039 | 0.1982 | 0.8091 | 0.3184 |
| Seal & Gasket | severe | A | hist_gradient_boosting | 0.1935 | 0.1914 | 0.9627 | 0.3193 |
| Seal & Gasket | severe | B | logistic_regression | 0.1989 | 0.1879 | 0.9668 | 0.3147 |
| Seal & Gasket | severe | B | random_forest | 0.1968 | 0.1865 | 0.9959 | 0.3141 |
| Seal & Gasket | severe | B | hist_gradient_boosting | 0.2089 | 0.1887 | 0.9834 | 0.3166 |
| Seal & Gasket | severe | C | logistic_regression | 0.2001 | 0.1845 | 0.9876 | 0.3109 |
| Seal & Gasket | severe | C | random_forest | 0.2001 | 0.1865 | 0.9253 | 0.3104 |
| Seal & Gasket | severe | C | hist_gradient_boosting | 0.2163 | 0.1865 | 0.9959 | 0.3141 |
| Drive Belt | affected | A | logistic_regression | 0.3048 | 0.2657 | 0.9884 | 0.4189 |
| Drive Belt | affected | A | random_forest | 0.2611 | 0.2662 | 0.9971 | 0.4202 |
| Drive Belt | affected | A | hist_gradient_boosting | 0.2688 | 0.2741 | 0.9364 | 0.4241 |
| Drive Belt | affected | B | logistic_regression | 0.2823 | 0.2699 | 0.9798 | 0.4232 |
| Drive Belt | affected | B | random_forest | 0.2745 | 0.2786 | 0.9422 | 0.4301 |
| Drive Belt | affected | B | hist_gradient_boosting | 0.2736 | 0.2662 | 1.0000 | 0.4204 |
| Drive Belt | affected | C | logistic_regression | 0.2769 | 0.2736 | 0.9624 | 0.4261 |
| Drive Belt | affected | C | random_forest | 0.2813 | 0.2692 | 0.9220 | 0.4167 |
| Drive Belt | affected | C | hist_gradient_boosting | 0.2758 | 0.2673 | 0.9942 | 0.4213 |
| Drive Belt | severe | A | logistic_regression | 0.0222 | 0.0714 | 0.0833 | 0.0769 |
| Drive Belt | severe | A | random_forest | 0.0161 | 0.0208 | 0.4583 | 0.0397 |
| Drive Belt | severe | A | hist_gradient_boosting | 0.0173 | 0.0256 | 0.2500 | 0.0465 |
| Drive Belt | severe | B | logistic_regression | 0.0342 | 0.0938 | 0.1250 | 0.1071 |
| Drive Belt | severe | B | random_forest | 0.0188 | 0.0230 | 0.5833 | 0.0443 |
| Drive Belt | severe | B | hist_gradient_boosting | 0.0268 | 0.0429 | 0.2917 | 0.0749 |
| Drive Belt | severe | C | logistic_regression | 0.0204 | 0.0337 | 0.2500 | 0.0594 |
| Drive Belt | severe | C | random_forest | 0.0155 | 0.0294 | 0.0417 | 0.0345 |
| Drive Belt | severe | C | hist_gradient_boosting | 0.0198 | 0.0265 | 0.3750 | 0.0496 |
| Filter | affected | A | logistic_regression | 0.2742 | 0.2474 | 0.9591 | 0.3933 |
| Filter | affected | A | random_forest | 0.2485 | 0.2465 | 0.9843 | 0.3942 |
| Filter | affected | A | hist_gradient_boosting | 0.2360 | 0.2446 | 1.0000 | 0.3931 |
| Filter | affected | B | logistic_regression | 0.2537 | 0.2513 | 0.9245 | 0.3952 |
| Filter | affected | B | random_forest | 0.2380 | 0.2440 | 0.9969 | 0.3921 |
| Filter | affected | B | hist_gradient_boosting | 0.2374 | 0.2476 | 0.9874 | 0.3960 |
| Filter | affected | C | logistic_regression | 0.2480 | 0.2488 | 0.9874 | 0.3975 |
| Filter | affected | C | random_forest | 0.2355 | 0.2436 | 0.9811 | 0.3902 |
| Filter | affected | C | hist_gradient_boosting | 0.2481 | 0.2477 | 0.9969 | 0.3967 |
| Filter | severe | A | logistic_regression | 0.0247 | 0.0833 | 0.0476 | 0.0606 |
| Filter | severe | A | random_forest | 0.0250 | 0.0458 | 0.2857 | 0.0789 |
| Filter | severe | A | hist_gradient_boosting | 0.0250 | 0.0435 | 0.2857 | 0.0755 |
| Filter | severe | B | logistic_regression | 0.0209 | 0.0441 | 0.1429 | 0.0674 |
| Filter | severe | B | random_forest | 0.0201 | 0.0435 | 0.1429 | 0.0667 |
| Filter | severe | B | hist_gradient_boosting | 0.0173 | 0.0476 | 0.0476 | 0.0476 |
| Filter | severe | C | logistic_regression | 0.0177 | 0.0213 | 0.4762 | 0.0407 |
| Filter | severe | C | random_forest | 0.0212 | 0.1111 | 0.0476 | 0.0667 |
| Filter | severe | C | hist_gradient_boosting | 0.0184 | 0.0833 | 0.0476 | 0.0606 |
| Electrical | affected | A | logistic_regression | 0.3244 | 0.3032 | 0.9770 | 0.4628 |
| Electrical | affected | A | random_forest | 0.3279 | 0.3099 | 0.9337 | 0.4654 |
| Electrical | affected | A | hist_gradient_boosting | 0.3295 | 0.3048 | 0.9719 | 0.4641 |
| Electrical | affected | B | logistic_regression | 0.3228 | 0.3074 | 0.9668 | 0.4665 |
| Electrical | affected | B | random_forest | 0.3200 | 0.3006 | 1.0000 | 0.4623 |
| Electrical | affected | B | hist_gradient_boosting | 0.3104 | 0.3037 | 0.9847 | 0.4642 |
| Electrical | affected | C | logistic_regression | 0.3020 | 0.3052 | 0.9949 | 0.4671 |
| Electrical | affected | C | random_forest | 0.3138 | 0.3016 | 0.9872 | 0.4621 |
| Electrical | affected | C | hist_gradient_boosting | 0.3071 | 0.3037 | 0.9847 | 0.4642 |
| Electrical | severe | A | logistic_regression | 0.2159 | 0.2064 | 0.8072 | 0.3287 |
| Electrical | severe | A | random_forest | 0.2080 | 0.1922 | 0.9839 | 0.3215 |
| Electrical | severe | A | hist_gradient_boosting | 0.2046 | 0.2000 | 0.8594 | 0.3245 |
| Electrical | severe | B | logistic_regression | 0.2257 | 0.1951 | 0.9920 | 0.3261 |
| Electrical | severe | B | random_forest | 0.1947 | 0.1889 | 0.9719 | 0.3163 |
| Electrical | severe | B | hist_gradient_boosting | 0.1815 | 0.1925 | 0.9960 | 0.3227 |
| Electrical | severe | C | logistic_regression | 0.1946 | 0.1962 | 0.9438 | 0.3248 |
| Electrical | severe | C | random_forest | 0.1797 | 0.1812 | 0.9237 | 0.3030 |
| Electrical | severe | C | hist_gradient_boosting | 0.1774 | 0.2007 | 0.9398 | 0.3307 |
| Coupling | affected | A | logistic_regression | 0.2700 | 0.2640 | 0.8750 | 0.4056 |
| Coupling | affected | A | random_forest | 0.2633 | 0.2406 | 1.0000 | 0.3878 |
| Coupling | affected | A | hist_gradient_boosting | 0.2435 | 0.2407 | 1.0000 | 0.3881 |
| Coupling | affected | B | logistic_regression | 0.2580 | 0.2412 | 0.9936 | 0.3882 |
| Coupling | affected | B | random_forest | 0.2568 | 0.2534 | 0.9487 | 0.4000 |
| Coupling | affected | B | hist_gradient_boosting | 0.2470 | 0.2489 | 0.9071 | 0.3906 |
| Coupling | affected | C | logistic_regression | 0.2513 | 0.2494 | 0.9295 | 0.3932 |
| Coupling | affected | C | random_forest | 0.2697 | 0.2371 | 0.9840 | 0.3821 |
| Coupling | affected | C | hist_gradient_boosting | 0.2407 | 0.2407 | 1.0000 | 0.3881 |
| Coupling | severe | A | logistic_regression | 0.0143 | 0.0161 | 0.9000 | 0.0316 |
| Coupling | severe | A | random_forest | 0.0169 | 0.0476 | 0.1500 | 0.0723 |
| Coupling | severe | A | hist_gradient_boosting | 0.0199 | 0.0380 | 0.1500 | 0.0606 |
| Coupling | severe | B | logistic_regression | 0.0163 | 0.0193 | 0.8000 | 0.0378 |
| Coupling | severe | B | random_forest | 0.0122 | 0.0250 | 0.0500 | 0.0333 |
| Coupling | severe | B | hist_gradient_boosting | 0.0110 | 0.0154 | 0.9500 | 0.0304 |
| Coupling | severe | C | logistic_regression | 0.0146 | 0.0172 | 0.8500 | 0.0336 |
| Coupling | severe | C | random_forest | 0.0101 | 0.0110 | 1.0000 | 0.0217 |
| Coupling | severe | C | hist_gradient_boosting | 0.0144 | 0.0184 | 0.1500 | 0.0328 |
| Lubrication | affected | A | logistic_regression | 0.1891 | 0.1939 | 0.9218 | 0.3205 |
| Lubrication | affected | A | random_forest | 0.1837 | 0.1913 | 0.9588 | 0.3190 |
| Lubrication | affected | A | hist_gradient_boosting | 0.1837 | 0.1931 | 0.9630 | 0.3216 |
| Lubrication | affected | B | logistic_regression | 0.1882 | 0.1892 | 0.9547 | 0.3159 |
| Lubrication | affected | B | random_forest | 0.2068 | 0.1876 | 1.0000 | 0.3160 |
| Lubrication | affected | B | hist_gradient_boosting | 0.1726 | 0.1893 | 0.9918 | 0.3179 |
| Lubrication | affected | C | logistic_regression | 0.1955 | 0.1853 | 0.9877 | 0.3121 |
| Lubrication | affected | C | random_forest | 0.1906 | 0.1831 | 0.9630 | 0.3077 |
| Lubrication | affected | C | hist_gradient_boosting | 0.1918 | 0.1887 | 1.0000 | 0.3174 |
| Lubrication | severe | A | logistic_regression | 0.0080 | 0.0103 | 0.9091 | 0.0204 |
| Lubrication | severe | A | random_forest | 0.0076 | 0.0096 | 0.4545 | 0.0188 |
| Lubrication | severe | A | hist_gradient_boosting | 0.0091 | 0.0125 | 0.4545 | 0.0244 |
| Lubrication | severe | B | logistic_regression | 0.0066 | 0.0082 | 0.5455 | 0.0162 |
| Lubrication | severe | B | random_forest | 0.0084 | 0.0097 | 0.6364 | 0.0192 |
| Lubrication | severe | B | hist_gradient_boosting | 0.0082 | 0.0121 | 0.2727 | 0.0233 |
| Lubrication | severe | C | logistic_regression | 0.0050 | 0.0065 | 0.9091 | 0.0129 |
| Lubrication | severe | C | random_forest | 0.0077 | 0.0142 | 0.1818 | 0.0263 |
| Lubrication | severe | C | hist_gradient_boosting | 0.0107 | 0.0192 | 0.0909 | 0.0317 |
| Sensor | affected | A | logistic_regression | 0.3207 | 0.2869 | 0.9457 | 0.4402 |
| Sensor | affected | A | random_forest | 0.3318 | 0.2837 | 1.0000 | 0.4420 |
| Sensor | affected | A | hist_gradient_boosting | 0.3258 | 0.2933 | 0.9212 | 0.4449 |
| Sensor | affected | B | logistic_regression | 0.3244 | 0.2885 | 0.9728 | 0.4450 |
| Sensor | affected | B | random_forest | 0.3102 | 0.2834 | 0.9973 | 0.4414 |
| Sensor | affected | B | hist_gradient_boosting | 0.3184 | 0.2831 | 1.0000 | 0.4412 |
| Sensor | affected | C | logistic_regression | 0.3174 | 0.2893 | 0.9647 | 0.4451 |
| Sensor | affected | C | random_forest | 0.2994 | 0.2902 | 0.9620 | 0.4458 |
| Sensor | affected | C | hist_gradient_boosting | 0.3250 | 0.2856 | 0.9973 | 0.4440 |
| Sensor | severe | A | logistic_regression | 0.2035 | 0.1858 | 0.8430 | 0.3045 |
| Sensor | severe | A | random_forest | 0.2098 | 0.2032 | 0.7444 | 0.3192 |
| Sensor | severe | A | hist_gradient_boosting | 0.1839 | 0.1820 | 0.7892 | 0.2958 |
| Sensor | severe | B | logistic_regression | 0.2184 | 0.2121 | 0.6278 | 0.3171 |
| Sensor | severe | B | random_forest | 0.1876 | 0.1706 | 0.9955 | 0.2913 |
| Sensor | severe | B | hist_gradient_boosting | 0.2075 | 0.2042 | 0.6592 | 0.3118 |
| Sensor | severe | C | logistic_regression | 0.2092 | 0.1774 | 0.9776 | 0.3003 |
| Sensor | severe | C | random_forest | 0.1979 | 0.1858 | 0.8072 | 0.3020 |
| Sensor | severe | C | hist_gradient_boosting | 0.1783 | 0.1759 | 0.9641 | 0.2976 |
| Fastener | affected | A | logistic_regression | 0.2268 | 0.2086 | 0.9583 | 0.3426 |
| Fastener | affected | A | random_forest | 0.2130 | 0.2042 | 1.0000 | 0.3391 |
| Fastener | affected | A | hist_gradient_boosting | 0.2173 | 0.2062 | 0.9886 | 0.3412 |
| Fastener | affected | B | logistic_regression | 0.2417 | 0.2116 | 0.9280 | 0.3446 |
| Fastener | affected | B | random_forest | 0.2361 | 0.2026 | 0.9886 | 0.3363 |
| Fastener | affected | B | hist_gradient_boosting | 0.2400 | 0.2153 | 0.8750 | 0.3455 |
| Fastener | affected | C | logistic_regression | 0.2546 | 0.2313 | 0.6667 | 0.3434 |
| Fastener | affected | C | random_forest | 0.2207 | 0.1995 | 0.9735 | 0.3312 |
| Fastener | affected | C | hist_gradient_boosting | 0.2195 | 0.2206 | 0.7803 | 0.3439 |
| Fastener | severe | A | logistic_regression | 0.0229 | 0.0833 | 0.0714 | 0.0769 |
| Fastener | severe | A | random_forest | 0.0156 | 0.0364 | 0.1429 | 0.0580 |
| Fastener | severe | A | hist_gradient_boosting | 0.0157 | 0.0288 | 0.2143 | 0.0508 |
| Fastener | severe | B | logistic_regression | 0.0177 | 0.0306 | 0.2143 | 0.0536 |
| Fastener | severe | B | random_forest | 0.0361 | 0.1538 | 0.1429 | 0.1481 |
| Fastener | severe | B | hist_gradient_boosting | 0.0216 | 0.0909 | 0.0714 | 0.0800 |
| Fastener | severe | C | logistic_regression | 0.0243 | 0.1000 | 0.0714 | 0.0833 |
| Fastener | severe | C | random_forest | 0.0095 | 0.0323 | 0.0714 | 0.0444 |
| Fastener | severe | C | hist_gradient_boosting | 0.0305 | 0.0800 | 0.1429 | 0.1026 |

## 3. Precision 0.70·0.80 가용성

검증에서 목표 Precision을 달성하지 못해 `unavailable`로 기록된 행은 24건입니다.

| part_family | target | threshold_policy | status | probability_cutoff |
| --- | --- | --- | --- | --- |
| Bearing | affected | min_precision_70 | ok | 0.8310 |
| Bearing | affected | min_precision_80 | ok | 0.8462 |
| Bearing | severe | min_precision_70 | identical_target | 0.8310 |
| Bearing | severe | min_precision_80 | identical_target | 0.8462 |
| Seal & Gasket | affected | min_precision_70 | unavailable | - |
| Seal & Gasket | affected | min_precision_80 | unavailable | - |
| Seal & Gasket | severe | min_precision_70 | unavailable | - |
| Seal & Gasket | severe | min_precision_80 | unavailable | - |
| Drive Belt | affected | min_precision_70 | ok | 0.8852 |
| Drive Belt | affected | min_precision_80 | ok | 0.8852 |
| Drive Belt | severe | min_precision_70 | unavailable | - |
| Drive Belt | severe | min_precision_80 | unavailable | - |
| Filter | affected | min_precision_70 | unavailable | - |
| Filter | affected | min_precision_80 | unavailable | - |
| Filter | severe | min_precision_70 | unavailable | - |
| Filter | severe | min_precision_80 | unavailable | - |
| Electrical | affected | min_precision_70 | unavailable | - |
| Electrical | affected | min_precision_80 | unavailable | - |
| Electrical | severe | min_precision_70 | unavailable | - |
| Electrical | severe | min_precision_80 | unavailable | - |
| Coupling | affected | min_precision_70 | ok | 0.8771 |
| Coupling | affected | min_precision_80 | ok | 0.8771 |
| Coupling | severe | min_precision_70 | unavailable | - |
| Coupling | severe | min_precision_80 | unavailable | - |
| Lubrication | affected | min_precision_70 | ok | 0.4000 |
| Lubrication | affected | min_precision_80 | ok | 0.4000 |
| Lubrication | severe | min_precision_70 | unavailable | - |
| Lubrication | severe | min_precision_80 | unavailable | - |
| Sensor | affected | min_precision_70 | ok | 0.9218 |
| Sensor | affected | min_precision_80 | ok | 0.9218 |
| Sensor | severe | min_precision_70 | unavailable | - |
| Sensor | severe | min_precision_80 | unavailable | - |
| Fastener | affected | min_precision_70 | unavailable | - |
| Fastener | affected | min_precision_80 | unavailable | - |
| Fastener | severe | min_precision_70 | unavailable | - |
| Fastener | severe | min_precision_80 | unavailable | - |

## 4. 다중라벨 전체 결과

| target | threshold_policy | macro_average_precision | macro_f1 | micro_average_precision | micro_f1 | hamming_loss | exact_match_ratio | excluded_families |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| affected | f1 | 0.2993 | 0.4262 | 0.3075 | 0.4367 | 0.4867 | 0.2735 | - |
| severe | f1 | 0.1364 | 0.1900 | 0.2375 | 0.3873 | 0.2314 | 0.2881 | - |

## 5. 기계 종류·장비별 결과

현재 저장 지표에 기계 종류 또는 장비 범위 열이 있으면 아래 표로 표시합니다. 전체 범위 실행만 했다면 별도 범위 결과가 없다는 뜻입니다.

| scope_kind | scope_name | part_family | target | average_precision | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| machine_type | CNC Lathe | Bearing | affected | 0.4131 | 0.4436 | 1.0000 | 0.6146 |
| machine_type | Hydraulic Press | Bearing | affected | 0.5551 | 0.4943 | 0.9924 | 0.6599 |
| machine_type | Belt Conveyor | Bearing | affected | 0.4246 | 0.4183 | 0.9910 | 0.5882 |
| machine_type | Screw Compressor | Bearing | affected | 0.5473 | 0.5227 | 0.9928 | 0.6849 |
| machine_type | EOT Crane | Bearing | affected | 0.5212 | 0.4247 | 0.9649 | 0.5898 |
| machine_type | CNC Lathe | Bearing | severe | 0.4131 | 0.4436 | 1.0000 | 0.6146 |
| machine_type | Hydraulic Press | Bearing | severe | 0.5551 | 0.4943 | 0.9924 | 0.6599 |
| machine_type | Belt Conveyor | Bearing | severe | 0.4246 | 0.4183 | 0.9910 | 0.5882 |
| machine_type | Screw Compressor | Bearing | severe | 0.5473 | 0.5227 | 0.9928 | 0.6849 |
| machine_type | EOT Crane | Bearing | severe | 0.5212 | 0.4247 | 0.9649 | 0.5898 |
| machine_type | CNC Lathe | Seal & Gasket | affected | 0.3309 | 0.3140 | 0.9643 | 0.4737 |
| machine_type | Hydraulic Press | Seal & Gasket | affected | 0.3941 | 0.3626 | 0.9694 | 0.5278 |
| machine_type | Belt Conveyor | Seal & Gasket | affected | 0.3466 | 0.2951 | 0.8780 | 0.4417 |
| machine_type | Screw Compressor | Seal & Gasket | affected | 0.3864 | 0.3798 | 0.9899 | 0.5490 |
| machine_type | EOT Crane | Seal & Gasket | affected | 0.3714 | 0.3265 | 0.9412 | 0.4848 |
| machine_type | CNC Lathe | Seal & Gasket | severe | 0.2052 | 0.1786 | 0.9574 | 0.3010 |
| machine_type | Hydraulic Press | Seal & Gasket | severe | 0.1860 | 0.2015 | 1.0000 | 0.3354 |
| machine_type | Belt Conveyor | Seal & Gasket | severe | 0.2035 | 0.1875 | 0.9412 | 0.3127 |
| machine_type | Screw Compressor | Seal & Gasket | severe | 0.2478 | 0.2085 | 0.9643 | 0.3429 |
| machine_type | EOT Crane | Seal & Gasket | severe | 0.1627 | 0.1705 | 1.0000 | 0.2913 |
| machine_type | CNC Lathe | Drive Belt | affected | 0.1767 | 0.2105 | 1.0000 | 0.3478 |
| machine_type | Hydraulic Press | Drive Belt | affected | 0.3233 | 0.3037 | 1.0000 | 0.4659 |
| machine_type | Belt Conveyor | Drive Belt | affected | 0.2339 | 0.2101 | 0.9818 | 0.3462 |
| machine_type | Screw Compressor | Drive Belt | affected | 0.4071 | 0.3797 | 1.0000 | 0.5504 |
| machine_type | EOT Crane | Drive Belt | affected | 0.3417 | 0.2689 | 0.9861 | 0.4226 |
| machine_type | CNC Lathe | Drive Belt | severe | 0.0449 | 0.0000 | 0.0000 | 0.0000 |
| machine_type | Hydraulic Press | Drive Belt | severe | 0.0247 | 0.0000 | 0.0000 | 0.0000 |
| machine_type | Belt Conveyor | Drive Belt | severe | 0.0126 | 0.0000 | 0.0000 | 0.0000 |
| machine_type | Screw Compressor | Drive Belt | severe | 0.0335 | 0.0000 | 0.0000 | 0.0000 |
| machine_type | EOT Crane | Drive Belt | severe | 0.0630 | 0.0909 | 0.2000 | 0.1250 |
| machine_type | CNC Lathe | Filter | affected | 0.2201 | 0.1571 | 0.9762 | 0.2706 |
| machine_type | Hydraulic Press | Filter | affected | 0.2803 | 0.2692 | 1.0000 | 0.4242 |
| machine_type | Belt Conveyor | Filter | affected | 0.2388 | 0.2559 | 0.9701 | 0.4050 |
| machine_type | Screw Compressor | Filter | affected | 0.2538 | 0.2548 | 0.9429 | 0.4012 |
| machine_type | EOT Crane | Filter | affected | 0.2470 | 0.2586 | 0.9855 | 0.4096 |
| machine_type | CNC Lathe | Filter | severe | 0.0081 | 0.0000 | 0.0000 | 0.0000 |
| machine_type | Hydraulic Press | Filter | severe | 0.0155 | 0.0000 | 0.0000 | 0.0000 |
| machine_type | Belt Conveyor | Filter | severe | 0.0238 | 0.0000 | 0.0000 | 0.0000 |
| machine_type | Screw Compressor | Filter | severe | 0.0382 | 0.0303 | 0.1429 | 0.0500 |
| machine_type | EOT Crane | Filter | severe | 0.0262 | 0.0000 | 0.0000 | 0.0000 |
| machine_type | CNC Lathe | Electrical | affected | 0.3138 | 0.2566 | 1.0000 | 0.4084 |
| machine_type | Hydraulic Press | Electrical | affected | 0.3474 | 0.3270 | 0.9885 | 0.4914 |
| machine_type | Belt Conveyor | Electrical | affected | 0.2499 | 0.2672 | 1.0000 | 0.4217 |
| machine_type | Screw Compressor | Electrical | affected | 0.2646 | 0.2692 | 0.9722 | 0.4217 |
| machine_type | EOT Crane | Electrical | affected | 0.3611 | 0.2615 | 0.9855 | 0.4134 |
| machine_type | CNC Lathe | Electrical | severe | 0.1702 | 0.1852 | 0.9375 | 0.3093 |
| machine_type | Hydraulic Press | Electrical | severe | 0.2109 | 0.2205 | 0.9825 | 0.3601 |
| machine_type | Belt Conveyor | Electrical | severe | 0.1728 | 0.1511 | 0.8500 | 0.2566 |
| machine_type | Screw Compressor | Electrical | severe | 0.2089 | 0.1867 | 0.9783 | 0.3136 |
| machine_type | EOT Crane | Electrical | severe | 0.2032 | 0.1690 | 0.8780 | 0.2835 |
| machine_type | CNC Lathe | Coupling | affected | 0.2054 | 0.1935 | 0.8400 | 0.3146 |
| machine_type | Hydraulic Press | Coupling | affected | 0.3266 | 0.2762 | 0.8571 | 0.4177 |
| machine_type | Belt Conveyor | Coupling | affected | 0.2522 | 0.1839 | 0.7111 | 0.2922 |
| machine_type | Screw Compressor | Coupling | affected | 0.2518 | 0.2636 | 0.8630 | 0.4038 |
| machine_type | EOT Crane | Coupling | affected | 0.2924 | 0.2850 | 0.8356 | 0.4251 |
| machine_type | CNC Lathe | Coupling | severe | 0.0027 | 0.0000 | 0.0000 | 0.0000 |
| machine_type | Hydraulic Press | Coupling | severe | 0.0386 | 0.0625 | 0.2000 | 0.0952 |
| machine_type | Belt Conveyor | Coupling | severe | 0.0385 | 0.0323 | 0.2500 | 0.0571 |
| machine_type | Screw Compressor | Coupling | severe | 0.2470 | 0.1053 | 0.4000 | 0.1667 |
| machine_type | EOT Crane | Coupling | severe | 0.0077 | 0.0000 | 0.0000 | 0.0000 |
| machine_type | CNC Lathe | Lubrication | affected | 0.1653 | 0.1679 | 0.9778 | 0.2866 |
| machine_type | Hydraulic Press | Lubrication | affected | 0.2886 | 0.2313 | 1.0000 | 0.3758 |
| machine_type | Belt Conveyor | Lubrication | affected | 0.2699 | 0.1767 | 1.0000 | 0.3003 |
| machine_type | Screw Compressor | Lubrication | affected | 0.1971 | 0.1977 | 1.0000 | 0.3302 |
| machine_type | EOT Crane | Lubrication | affected | 0.2292 | 0.1978 | 1.0000 | 0.3302 |
| machine_type | CNC Lathe | Lubrication | severe | 0.0079 | 0.0000 | 0.0000 | 0.0000 |
| machine_type | Hydraulic Press | Lubrication | severe | 0.0033 | 0.0000 | 0.0000 | 0.0000 |
| machine_type | Belt Conveyor | Lubrication | severe | 0.0294 | 0.0000 | 0.0000 | 0.0000 |
| machine_type | Screw Compressor | Lubrication | severe | 0.0035 | 0.0000 | 0.0000 | 0.0000 |
| machine_type | EOT Crane | Lubrication | severe | 0.0122 | 0.0000 | 0.0000 | 0.0000 |
| machine_type | CNC Lathe | Sensor | affected | 0.3454 | 0.2939 | 1.0000 | 0.4543 |
| machine_type | Hydraulic Press | Sensor | affected | 0.2857 | 0.3371 | 0.9889 | 0.5028 |
| machine_type | Belt Conveyor | Sensor | affected | 0.3431 | 0.2816 | 0.9324 | 0.4326 |
| machine_type | Screw Compressor | Sensor | affected | 0.3373 | 0.3135 | 0.9405 | 0.4702 |
| machine_type | EOT Crane | Sensor | affected | 0.3413 | 0.2811 | 0.9459 | 0.4334 |
| machine_type | CNC Lathe | Sensor | severe | 0.2179 | 0.1689 | 0.5814 | 0.2618 |
| machine_type | Hydraulic Press | Sensor | severe | 0.2393 | 0.2312 | 0.7049 | 0.3482 |
| machine_type | Belt Conveyor | Sensor | severe | 0.2152 | 0.1925 | 0.6739 | 0.2995 |
| machine_type | Screw Compressor | Sensor | severe | 0.1731 | 0.1556 | 0.5385 | 0.2414 |
| machine_type | EOT Crane | Sensor | severe | 0.2135 | 0.1882 | 0.3556 | 0.2462 |
| machine_type | CNC Lathe | Fastener | affected | 0.1815 | 0.1520 | 0.6500 | 0.2464 |
| machine_type | Hydraulic Press | Fastener | affected | 0.3026 | 0.2536 | 0.8281 | 0.3883 |
| machine_type | Belt Conveyor | Fastener | affected | 0.1398 | 0.1319 | 0.5135 | 0.2099 |
| machine_type | Screw Compressor | Fastener | affected | 0.2421 | 0.2194 | 0.5484 | 0.3134 |
| machine_type | EOT Crane | Fastener | affected | 0.2246 | 0.2125 | 0.5667 | 0.3091 |
| machine_type | CNC Lathe | Fastener | severe | 0.0121 | 0.0000 | 0.0000 | 0.0000 |
| machine_type | Hydraulic Press | Fastener | severe | 0.0219 | 0.0000 | 0.0000 | 0.0000 |
| machine_type | Belt Conveyor | Fastener | severe | 0.0300 | 0.0000 | 0.0000 | 0.0000 |
| machine_type | Screw Compressor | Fastener | severe | 0.0574 | 0.2500 | 0.1667 | 0.2000 |
| machine_type | EOT Crane | Fastener | severe | 0.0199 | 0.0000 | 0.0000 | 0.0000 |
| asset_tag | AST-1041 | Bearing | affected | 0.3919 | 0.4135 | 1.0000 | 0.5851 |
| asset_tag | AST-1042 | Bearing | affected | 0.4495 | 0.4737 | 1.0000 | 0.6429 |
| asset_tag | AST-2017 | Bearing | affected | 0.6239 | 0.5113 | 1.0000 | 0.6766 |
| asset_tag | AST-2031 | Bearing | affected | 0.5056 | 0.4773 | 0.9844 | 0.6429 |
| asset_tag | AST-3008 | Bearing | affected | 0.4629 | 0.4436 | 1.0000 | 0.6146 |
| asset_tag | AST-3019 | Bearing | affected | 0.3805 | 0.3923 | 0.9808 | 0.5604 |
| asset_tag | AST-4055 | Bearing | affected | 0.5498 | 0.5639 | 1.0000 | 0.7212 |
| asset_tag | AST-4061 | Bearing | affected | 0.5313 | 0.4809 | 0.9844 | 0.6462 |
| asset_tag | AST-5003 | Bearing | affected | 0.5974 | 0.5263 | 1.0000 | 0.6897 |
| asset_tag | AST-5007 | Bearing | affected | 0.3183 | 0.3175 | 0.9091 | 0.4706 |
| asset_tag | AST-1041 | Bearing | severe | 0.3919 | 0.4135 | 1.0000 | 0.5851 |
| asset_tag | AST-1042 | Bearing | severe | 0.4495 | 0.4737 | 1.0000 | 0.6429 |
| asset_tag | AST-2017 | Bearing | severe | 0.6239 | 0.5113 | 1.0000 | 0.6766 |
| asset_tag | AST-2031 | Bearing | severe | 0.5056 | 0.4773 | 0.9844 | 0.6429 |
| asset_tag | AST-3008 | Bearing | severe | 0.4629 | 0.4436 | 1.0000 | 0.6146 |
| asset_tag | AST-3019 | Bearing | severe | 0.3805 | 0.3923 | 0.9808 | 0.5604 |
| asset_tag | AST-4055 | Bearing | severe | 0.5498 | 0.5639 | 1.0000 | 0.7212 |
| asset_tag | AST-4061 | Bearing | severe | 0.5313 | 0.4809 | 0.9844 | 0.6462 |
| asset_tag | AST-5003 | Bearing | severe | 0.5974 | 0.5263 | 1.0000 | 0.6897 |
| asset_tag | AST-5007 | Bearing | severe | 0.3183 | 0.3175 | 0.9091 | 0.4706 |
| asset_tag | AST-1041 | Seal & Gasket | affected | 0.3691 | 0.3488 | 0.9574 | 0.5114 |
| asset_tag | AST-1042 | Seal & Gasket | affected | 0.2984 | 0.2791 | 0.9730 | 0.4337 |
| asset_tag | AST-2017 | Seal & Gasket | affected | 0.4045 | 0.3511 | 0.9583 | 0.5140 |
| asset_tag | AST-2031 | Seal & Gasket | affected | 0.4011 | 0.3740 | 0.9800 | 0.5414 |
| asset_tag | AST-3008 | Seal & Gasket | affected | 0.3796 | 0.3440 | 0.9348 | 0.5029 |
| asset_tag | AST-3019 | Seal & Gasket | affected | 0.3154 | 0.2437 | 0.8056 | 0.3742 |
| asset_tag | AST-4055 | Seal & Gasket | affected | 0.4324 | 0.4091 | 1.0000 | 0.5806 |
| asset_tag | AST-4061 | Seal & Gasket | affected | 0.3363 | 0.3492 | 0.9778 | 0.5146 |
| asset_tag | AST-5003 | Seal & Gasket | affected | 0.3862 | 0.4046 | 0.9636 | 0.5699 |
| asset_tag | AST-5007 | Seal & Gasket | affected | 0.3764 | 0.2368 | 0.9000 | 0.3750 |
| asset_tag | AST-1041 | Seal & Gasket | severe | 0.2340 | 0.1951 | 0.9231 | 0.3221 |
| asset_tag | AST-1042 | Seal & Gasket | severe | 0.2362 | 0.1628 | 1.0000 | 0.2800 |
| asset_tag | AST-2017 | Seal & Gasket | severe | 0.2022 | 0.2105 | 1.0000 | 0.3478 |
| asset_tag | AST-2031 | Seal & Gasket | severe | 0.1842 | 0.1923 | 1.0000 | 0.3226 |
| asset_tag | AST-3008 | Seal & Gasket | severe | 0.1882 | 0.2061 | 0.9643 | 0.3396 |
| asset_tag | AST-3019 | Seal & Gasket | severe | 0.3029 | 0.1680 | 0.9130 | 0.2838 |
| asset_tag | AST-4055 | Seal & Gasket | severe | 0.2889 | 0.2406 | 1.0000 | 0.3879 |
| asset_tag | AST-4061 | Seal & Gasket | severe | 0.1839 | 0.1746 | 0.9167 | 0.2933 |
| asset_tag | AST-5003 | Seal & Gasket | severe | 0.1677 | 0.2030 | 1.0000 | 0.3375 |
| asset_tag | AST-5007 | Seal & Gasket | severe | 0.1788 | 0.1374 | 1.0000 | 0.2416 |
| asset_tag | AST-1041 | Drive Belt | affected | 0.1621 | 0.2030 | 1.0000 | 0.3375 |
| asset_tag | AST-1042 | Drive Belt | affected | 0.2113 | 0.2180 | 1.0000 | 0.3580 |
| asset_tag | AST-2017 | Drive Belt | affected | 0.3723 | 0.3407 | 1.0000 | 0.5083 |
| asset_tag | AST-2031 | Drive Belt | affected | 0.2664 | 0.2667 | 1.0000 | 0.4211 |
| asset_tag | AST-3008 | Drive Belt | affected | 0.2187 | 0.2326 | 1.0000 | 0.3774 |
| asset_tag | AST-3019 | Drive Belt | affected | 0.2802 | 0.1875 | 0.9600 | 0.3137 |
| asset_tag | AST-4055 | Drive Belt | affected | 0.4464 | 0.4436 | 1.0000 | 0.6146 |
| asset_tag | AST-4061 | Drive Belt | affected | 0.3706 | 0.3158 | 1.0000 | 0.4800 |
| asset_tag | AST-5003 | Drive Belt | affected | 0.3833 | 0.2701 | 1.0000 | 0.4253 |
| asset_tag | AST-5007 | Drive Belt | affected | 0.2988 | 0.2677 | 0.9714 | 0.4198 |
| asset_tag | AST-1041 | Drive Belt | severe | 0.0636 | 0.0000 | 0.0000 | 0.0000 |
| asset_tag | AST-1042 | Drive Belt | severe | 0.0278 | 0.0000 | 0.0000 | 0.0000 |
| asset_tag | AST-2017 | Drive Belt | severe | 0.0400 | 0.0000 | 0.0000 | 0.0000 |
| asset_tag | AST-2031 | Drive Belt | severe | 0.0217 | 0.0000 | 0.0000 | 0.0000 |
| asset_tag | AST-3008 | Drive Belt | severe | 0.0206 | 0.0000 | 0.0000 | 0.0000 |
| asset_tag | AST-3019 | Drive Belt | severe | 0.0076 | 0.0000 | 0.0000 | 0.0000 |
| asset_tag | AST-4055 | Drive Belt | severe | 0.0442 | 0.0000 | 0.0000 | 0.0000 |
| asset_tag | AST-4061 | Drive Belt | severe | 0.0331 | 0.0000 | 0.0000 | 0.0000 |
| asset_tag | AST-5003 | Drive Belt | severe | 0.0875 | 0.1000 | 0.3333 | 0.1538 |
| asset_tag | AST-5007 | Drive Belt | severe | 0.0768 | 0.0000 | 0.0000 | 0.0000 |
| asset_tag | AST-1041 | Filter | affected | 0.2509 | 0.1719 | 0.9565 | 0.2914 |
| asset_tag | AST-1042 | Filter | affected | 0.2112 | 0.1429 | 1.0000 | 0.2500 |
| asset_tag | AST-2017 | Filter | affected | 0.2964 | 0.2901 | 1.0000 | 0.4497 |
| asset_tag | AST-2031 | Filter | affected | 0.2898 | 0.2481 | 1.0000 | 0.3975 |
| asset_tag | AST-3008 | Filter | affected | 0.2733 | 0.2422 | 1.0000 | 0.3899 |
| asset_tag | AST-3019 | Filter | affected | 0.2189 | 0.2698 | 0.9444 | 0.4198 |
| asset_tag | AST-4055 | Filter | affected | 0.2830 | 0.2857 | 1.0000 | 0.4444 |
| asset_tag | AST-4061 | Filter | affected | 0.2192 | 0.2222 | 0.8750 | 0.3544 |
| asset_tag | AST-5003 | Filter | affected | 0.3017 | 0.3358 | 1.0000 | 0.5028 |
| asset_tag | AST-5007 | Filter | affected | 0.1534 | 0.1783 | 0.9583 | 0.3007 |
| asset_tag | AST-1041 | Filter | severe | 0.0108 | 0.0000 | 0.0000 | 0.0000 |
| asset_tag | AST-1042 | Filter | severe | 0.0054 | 0.0000 | 0.0000 | 0.0000 |
| asset_tag | AST-2017 | Filter | severe | 0.0303 | 0.0000 | 0.0000 | 0.0000 |
| asset_tag | AST-2031 | Filter | severe | 0.0194 | 0.0000 | 0.0000 | 0.0000 |
| asset_tag | AST-3008 | Filter | severe | 0.0570 | 0.0000 | 0.0000 | 0.0000 |
| asset_tag | AST-3019 | Filter | severe | 0.0162 | 0.0000 | 0.0000 | 0.0000 |
| asset_tag | AST-4055 | Filter | severe | 0.0259 | 0.0000 | 0.0000 | 0.0000 |
| asset_tag | AST-4061 | Filter | severe | 0.0591 | 0.0526 | 0.2500 | 0.0870 |
| asset_tag | AST-5003 | Filter | severe | 0.0283 | 0.0000 | 0.0000 | 0.0000 |
| asset_tag | AST-5007 | Filter | severe | 0.0306 | 0.0000 | 0.0000 | 0.0000 |
| asset_tag | AST-1041 | Electrical | affected | 0.3906 | 0.2652 | 1.0000 | 0.4192 |
| asset_tag | AST-1042 | Electrical | affected | 0.2690 | 0.2481 | 1.0000 | 0.3976 |
| asset_tag | AST-2017 | Electrical | affected | 0.3724 | 0.3485 | 1.0000 | 0.5169 |
| asset_tag | AST-2031 | Electrical | affected | 0.3440 | 0.3053 | 0.9756 | 0.4651 |
| asset_tag | AST-3008 | Electrical | affected | 0.2868 | 0.2595 | 1.0000 | 0.4121 |
| asset_tag | AST-3019 | Electrical | affected | 0.2342 | 0.2748 | 1.0000 | 0.4311 |
| asset_tag | AST-4055 | Electrical | affected | 0.2684 | 0.2857 | 1.0000 | 0.4444 |
| asset_tag | AST-4061 | Electrical | affected | 0.2844 | 0.2520 | 0.9412 | 0.3975 |
| asset_tag | AST-5003 | Electrical | affected | 0.4070 | 0.3206 | 0.9767 | 0.4828 |
| asset_tag | AST-5007 | Electrical | affected | 0.2677 | 0.2016 | 1.0000 | 0.3355 |
| asset_tag | AST-1041 | Electrical | severe | 0.1900 | 0.1739 | 0.9524 | 0.2941 |
| asset_tag | AST-1042 | Electrical | severe | 0.1717 | 0.1953 | 0.9259 | 0.3226 |
| asset_tag | AST-2017 | Electrical | severe | 0.2688 | 0.2677 | 1.0000 | 0.4224 |
| asset_tag | AST-2031 | Electrical | severe | 0.1573 | 0.1732 | 0.9565 | 0.2933 |
| asset_tag | AST-3008 | Electrical | severe | 0.1883 | 0.1130 | 0.8125 | 0.1985 |
| asset_tag | AST-3019 | Electrical | severe | 0.1907 | 0.1909 | 0.8750 | 0.3134 |
| asset_tag | AST-4055 | Electrical | severe | 0.2316 | 0.2121 | 1.0000 | 0.3500 |
| asset_tag | AST-4061 | Electrical | severe | 0.1744 | 0.1560 | 0.9444 | 0.2677 |
| asset_tag | AST-5003 | Electrical | severe | 0.2348 | 0.2240 | 0.9655 | 0.3636 |
| asset_tag | AST-5007 | Electrical | severe | 0.1189 | 0.0909 | 0.6667 | 0.1600 |
| asset_tag | AST-1041 | Coupling | affected | 0.2689 | 0.2056 | 0.8800 | 0.3333 |
| asset_tag | AST-1042 | Coupling | affected | 0.1800 | 0.1818 | 0.8000 | 0.2963 |
| asset_tag | AST-2017 | Coupling | affected | 0.3516 | 0.2598 | 0.9167 | 0.4049 |
| asset_tag | AST-2031 | Coupling | affected | 0.3350 | 0.2946 | 0.8049 | 0.4314 |
| asset_tag | AST-3008 | Coupling | affected | 0.2869 | 0.2056 | 0.8800 | 0.3333 |
| asset_tag | AST-3019 | Coupling | affected | 0.2178 | 0.1493 | 0.5000 | 0.2299 |
| asset_tag | AST-4055 | Coupling | affected | 0.2589 | 0.2698 | 0.9189 | 0.4172 |
| asset_tag | AST-4061 | Coupling | affected | 0.2669 | 0.2566 | 0.8056 | 0.3893 |
| asset_tag | AST-5003 | Coupling | affected | 0.2958 | 0.2917 | 0.8974 | 0.4403 |
| asset_tag | AST-5007 | Coupling | affected | 0.3134 | 0.2766 | 0.7647 | 0.4062 |
| asset_tag | AST-1041 | Coupling | severe | 0.0054 | 0.0000 | 0.0000 | 0.0000 |
| asset_tag | AST-1042 | Coupling | severe | - | 0.0000 | 0.0000 | 0.0000 |
| asset_tag | AST-2017 | Coupling | severe | 0.0760 | 0.1429 | 0.3333 | 0.2000 |
| asset_tag | AST-2031 | Coupling | severe | 0.0271 | 0.0000 | 0.0000 | 0.0000 |
| asset_tag | AST-3008 | Coupling | severe | 0.0574 | 0.0526 | 0.2500 | 0.0870 |
| asset_tag | AST-3019 | Coupling | severe | - | 0.0000 | 0.0000 | 0.0000 |
| asset_tag | AST-4055 | Coupling | severe | 1.0000 | 0.1667 | 1.0000 | 0.2857 |
| asset_tag | AST-4061 | Coupling | severe | 0.0587 | 0.0769 | 0.2500 | 0.1176 |
| asset_tag | AST-5003 | Coupling | severe | 0.0096 | 0.0000 | 0.0000 | 0.0000 |
| asset_tag | AST-5007 | Coupling | severe | 0.0054 | 0.0000 | 0.0000 | 0.0000 |
| asset_tag | AST-1041 | Lubrication | affected | 0.2784 | 0.2016 | 0.9630 | 0.3333 |
| asset_tag | AST-1042 | Lubrication | affected | 0.1368 | 0.1353 | 1.0000 | 0.2384 |
| asset_tag | AST-2017 | Lubrication | affected | 0.3463 | 0.2857 | 1.0000 | 0.4444 |
| asset_tag | AST-2031 | Lubrication | affected | 0.1797 | 0.1778 | 1.0000 | 0.3019 |
| asset_tag | AST-3008 | Lubrication | affected | 0.2456 | 0.1729 | 1.0000 | 0.2949 |
| asset_tag | AST-3019 | Lubrication | affected | 0.2922 | 0.1805 | 1.0000 | 0.3057 |
| asset_tag | AST-4055 | Lubrication | affected | 0.1912 | 0.2090 | 1.0000 | 0.3457 |
| asset_tag | AST-4061 | Lubrication | affected | 0.2033 | 0.1860 | 1.0000 | 0.3137 |
| asset_tag | AST-5003 | Lubrication | affected | 0.2708 | 0.2074 | 1.0000 | 0.3436 |
| asset_tag | AST-5007 | Lubrication | affected | 0.2088 | 0.1880 | 1.0000 | 0.3165 |
| asset_tag | AST-1041 | Lubrication | severe | 0.0169 | 0.0000 | 0.0000 | 0.0000 |
| asset_tag | AST-1042 | Lubrication | severe | - | 0.0000 | 0.0000 | 0.0000 |
| asset_tag | AST-2017 | Lubrication | severe | 0.0068 | 0.0000 | 0.0000 | 0.0000 |
| asset_tag | AST-2031 | Lubrication | severe | - | 0.0000 | 0.0000 | 0.0000 |
| asset_tag | AST-3008 | Lubrication | severe | - | 0.0000 | 0.0000 | 0.0000 |
| asset_tag | AST-3019 | Lubrication | severe | 0.1667 | 0.0000 | 0.0000 | 0.0000 |
| asset_tag | AST-4055 | Lubrication | severe | - | 0.0000 | 0.0000 | 0.0000 |
| asset_tag | AST-4061 | Lubrication | severe | 0.0075 | 0.0000 | 0.0000 | 0.0000 |
| asset_tag | AST-5003 | Lubrication | severe | 0.0073 | 0.0000 | 0.0000 | 0.0000 |
| asset_tag | AST-5007 | Lubrication | severe | 0.0345 | 0.0000 | 0.0000 | 0.0000 |
| asset_tag | AST-1041 | Sensor | affected | 0.3350 | 0.2727 | 1.0000 | 0.4286 |
| asset_tag | AST-1042 | Sensor | affected | 0.3700 | 0.3154 | 1.0000 | 0.4795 |
| asset_tag | AST-2017 | Sensor | affected | 0.2994 | 0.3561 | 1.0000 | 0.5251 |
| asset_tag | AST-2031 | Sensor | affected | 0.2815 | 0.3182 | 0.9767 | 0.4800 |
| asset_tag | AST-3008 | Sensor | affected | 0.3683 | 0.2913 | 0.9487 | 0.4458 |
| asset_tag | AST-3019 | Sensor | affected | 0.3480 | 0.2712 | 0.9143 | 0.4183 |
| asset_tag | AST-4055 | Sensor | affected | 0.2909 | 0.3308 | 1.0000 | 0.4972 |
| asset_tag | AST-4061 | Sensor | affected | 0.4621 | 0.2941 | 0.8750 | 0.4403 |
| asset_tag | AST-5003 | Sensor | affected | 0.3882 | 0.3588 | 0.9592 | 0.5222 |
| asset_tag | AST-5007 | Sensor | affected | 0.2356 | 0.1949 | 0.9200 | 0.3217 |
| asset_tag | AST-1041 | Sensor | severe | 0.2484 | 0.1772 | 0.6364 | 0.2772 |
| asset_tag | AST-1042 | Sensor | severe | 0.2316 | 0.1594 | 0.5238 | 0.2444 |
| asset_tag | AST-2017 | Sensor | severe | 0.2413 | 0.2393 | 0.8485 | 0.3733 |
| asset_tag | AST-2031 | Sensor | severe | 0.2517 | 0.2174 | 0.5357 | 0.3093 |
| asset_tag | AST-3008 | Sensor | severe | 0.2459 | 0.2000 | 0.8696 | 0.3252 |
| asset_tag | AST-3019 | Sensor | severe | 0.2021 | 0.1803 | 0.4783 | 0.2619 |
| asset_tag | AST-4055 | Sensor | severe | 0.1403 | 0.1053 | 0.2963 | 0.1553 |
| asset_tag | AST-4061 | Sensor | severe | 0.2192 | 0.1923 | 0.8000 | 0.3101 |
| asset_tag | AST-5003 | Sensor | severe | 0.2253 | 0.2143 | 0.5000 | 0.3000 |
| asset_tag | AST-5007 | Sensor | severe | 0.1513 | 0.0667 | 0.0667 | 0.0667 |
| asset_tag | AST-1041 | Fastener | affected | 0.2361 | 0.1522 | 0.7368 | 0.2523 |
| asset_tag | AST-1042 | Fastener | affected | 0.1699 | 0.1519 | 0.5714 | 0.2400 |
| asset_tag | AST-2017 | Fastener | affected | 0.3253 | 0.3056 | 0.8250 | 0.4459 |
| asset_tag | AST-2031 | Fastener | affected | 0.3082 | 0.1980 | 0.8333 | 0.3200 |
| asset_tag | AST-3008 | Fastener | affected | 0.1325 | 0.1176 | 0.5263 | 0.1923 |
| asset_tag | AST-3019 | Fastener | affected | 0.1642 | 0.1525 | 0.5000 | 0.2338 |
| asset_tag | AST-4055 | Fastener | affected | 0.2721 | 0.2500 | 0.6053 | 0.3538 |
| asset_tag | AST-4061 | Fastener | affected | 0.2129 | 0.1746 | 0.4583 | 0.2529 |
| asset_tag | AST-5003 | Fastener | affected | 0.2203 | 0.1869 | 0.7692 | 0.3008 |
| asset_tag | AST-5007 | Fastener | affected | 0.2775 | 0.2642 | 0.4118 | 0.3218 |
| asset_tag | AST-1041 | Fastener | severe | 0.0149 | 0.0000 | 0.0000 | 0.0000 |
| asset_tag | AST-1042 | Fastener | severe | 0.0148 | 0.0000 | 0.0000 | 0.0000 |
| asset_tag | AST-2017 | Fastener | severe | 0.0204 | 0.0000 | 0.0000 | 0.0000 |
| asset_tag | AST-2031 | Fastener | severe | 0.0258 | 0.0000 | 0.0000 | 0.0000 |
| asset_tag | AST-3008 | Fastener | severe | 0.0769 | 0.0000 | 0.0000 | 0.0000 |
| asset_tag | AST-3019 | Fastener | severe | 0.0103 | 0.0000 | 0.0000 | 0.0000 |
| asset_tag | AST-4055 | Fastener | severe | 0.1419 | 0.2500 | 0.5000 | 0.3333 |
| asset_tag | AST-4061 | Fastener | severe | 0.0216 | 0.0000 | 0.0000 | 0.0000 |
| asset_tag | AST-5003 | Fastener | severe | 0.0054 | 0.0000 | 0.0000 | 0.0000 |
| asset_tag | AST-5007 | Fastener | severe | 0.1111 | 0.0000 | 0.0000 | 0.0000 |

## 6. 순차 모델 판정

판정: **진행 보류**  
선택 시간 Feature: B  
검증 Macro AP 차이: -0.0013  
개선 Family 수: 4  
판정 사유:
- 검증 Macro AP 개선이 0.02 미만입니다.
- affected 개선 Family가 5개 미만입니다.
- 테스트 Macro AP가 정적 A보다 낮습니다.

## 7. 합성 데이터 제한과 다음 단계

현재 `breakdown_flag`는 합성 데이터의 대리 정답이며 실제 정지 시각·정비 확정·생산 손실이 아닙니다. 따라서 이 결과만으로 부품 교체를 지시하지 않습니다. 당일 Family 식별력이 충분하고 순차 모델 조건을 통과할 때만 미래 1·3·7일 예측을 별도 설계합니다.
