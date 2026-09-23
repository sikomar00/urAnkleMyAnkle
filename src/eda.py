import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,precision_recall_curve
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

df = pd.read_csv('data/raw/synthetic_industrial_machine_data.csv')
# df.info()
# df["timestamp"] = pd.to_datetime(df["timestamp"])
# df = df.sort_values("timestamp")
# print(df)
df.dropna(axis = 1,inplace = True)
df = df[['transaction_date','part_no','criticality','asset_tag','part_description','machine_type','temp_bearing_degC','temp_motor_degC','vibration_h_mms','vibration_v_mms','oil_pressure_bar','load_pct','shaft_rpm','power_consumption_kw','breakdown_flag']]
weights = {"A": 4, "B": 2, "C": 1}
# df.info()
# print(df.columns)
# print(df.head())
# print(df['machine_type'].unique())
data = df.copy()
data["transaction_date"] = pd.to_datetime(
    data["transaction_date"]
).dt.normalize()
data["criticality_weight"] = (
    data["criticality"]
    .astype("string")
    .str.strip()
    .str.upper()
    .map(weights)
)

target_data = 'breakdown_flag'
feature_data = ['temp_bearing_degC','temp_motor_degC','vibration_h_mms','vibration_v_mms','oil_pressure_bar','load_pct','shaft_rpm','power_consumption_kw']


if data["criticality_weight"].isna().any():
    raise ValueError("criticality에 A/B/C 이외의 값 또는 결측값이 있습니다.")

if not data["breakdown_flag"].isin([0, 1]).all():
    raise ValueError("breakdown_flag는 0 또는 1이어야 합니다.")

# 같은 부품을 중복으로 더하지 않도록 검사
keys = ["transaction_date", "asset_tag", "part_no"]
if data.duplicated(keys).any():
    raise ValueError("같은 날짜·장비·부품의 중복 행을 먼저 확인하세요.")

# 실제 고장 난 부품만 점수에 반영
data["failure_points"] = (
    data["breakdown_flag"] * data["criticality_weight"]
)

group_keys = ["transaction_date", "machine_type", "asset_tag"]

# 같은 장비·날짜의 센서가 공통값인지 검사
sensor_counts = data.groupby(group_keys)[feature_data].nunique(
    dropna=False
)
if sensor_counts.gt(1).any().any():
    raise ValueError(
        "같은 장비·날짜에 센서값이 다릅니다. "
        "평균/최대 등 집계 방식을 먼저 정하세요."
    )

daily = (
    data.groupby(group_keys, as_index=False)
    .agg({
        "failure_points": "sum",
        **{column: "first" for column in feature_data},
    })
)

for threshold in [12, 13, 14]:
    daily[f"target_gt_{threshold}"] = (
        daily["failure_points"] >= threshold
    ).astype(int)

print(
    daily.groupby("machine_type")[
        ["target_gt_12", "target_gt_13", "target_gt_14"]
    ].mean().rename(columns=lambda c: f"{c}_양성률")
)
# df1 = data[data['machine_type'] == 'CNC Lathe']
# for i in feature_data:
#     fig = px.scatter(
#     df1,
#     x="transaction_date",
#     y=i,
#     color_discrete_sequence=['blue'],
#     title="날짜별 feature값변동",
#     labels={
#         "transaction_date": "날짜",
#         i: 'feature'
#         })
#     fig.show()
# X = df1[feature_data]
# y = df1[target_data]
# X_train,X_test,y_train,y_test = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     random_state = 42,
#     stratify = y
# )

####LogisticRegression#############

# model = LogisticRegression(max_iter = 10000000)


# model = RandomForestClassifier(
#     n_estimators=100,
#     max_depth=20,
#     random_state=42
# )
# model.fit(X_train,y_train)


# y_pred = model.predict(X_test)
# train_pred = model.predict(X_train)

# print("학습 F1:", f1_score(y_train, train_pred, zero_division=0))
# print("테스트 F1:", f1_score(y_test, y_pred, zero_division=0))
# print(y_pred[:10])
# print(y_test[:10])

# model = HistGradientBoostingClassifier(
#     learning_rate = 0.05,
#     max_iter = 200,
#     max_depth = 10)

# # model = LGBMClassifier(
    
# # )
# model.fit(X_train,y_train)
# y_pred = model.predict(X_test)


# cm = confusion_matrix(
#     y_test,
#     y_pred
# )
# print(cm)

# accuracy = accuracy_score(
#     y_test,
#     y_pred
# )
# print("Accuracy:", accuracy)

# precision = precision_score(
#     y_test,
#     y_pred,
#     zero_division=0
# )
# recall = recall_score(
#     y_test,
#     y_pred,
#     zero_division=0
# )
# f1 = f1_score(
#     y_test,
#     y_pred,
#     zero_division=0
# )

# print("Precision:", precision)
# print("Recall:", recall)
# print("F1-score:", f1)


results = []
models = {}

for machine_type, group in daily.groupby("machine_type"):
    train = group[group["transaction_date"] < "2024-01-01"]
    valid = group[
        (group["transaction_date"] >= "2024-01-01")
        & (group["transaction_date"] < "2024-07-01")
    ]
    test = group[group["transaction_date"] >= "2024-07-01"]

    for threshold in [12, 13, 14]:
        target = f"target_gt_{threshold}"
        features = ["asset_tag", * feature_data]

        # 陽性が少ない条件も隠さず記録
        counts = {
            "train_positive": int(train[target].sum()),
            "valid_positive": int(valid[target].sum()),
            "test_positive": int(test[target].sum()),
        }

        if (
            any(part.empty for part in [train, valid, test])
            or train[target].nunique() < 2
            or valid[target].nunique() < 2
        ):
            results.append({
                "machine_type": machine_type,
                "score_threshold": threshold,
                "status": "skip: 学習・検証に両クラスが必要",
                **counts,
            })
            continue

        preprocess = ColumnTransformer([
            ("asset", OneHotEncoder(handle_unknown="ignore"),
             ["asset_tag"]),
            ("sensors", SimpleImputer(strategy="median"),
             feature_data),
        ])

        model = Pipeline([
            ("preprocess", preprocess),
            ("classifier", RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )),
        ])

        model.fit(train[features], train[target])

        positive_index = list(model.classes_).index(1)
        valid_scores = model.predict_proba(
            valid[features]
        )[:, positive_index]

        precision, recall, cutoffs = precision_recall_curve(
            valid[target], valid_scores
        )
        f1_values = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros_like(precision),
            where=(precision + recall) != 0,
        )
        probability_cutoff = float(
            cutoffs[np.argmax(f1_values[:-1])]
        )

        test_scores = model.predict_proba(
            test[features]
        )[:, positive_index]
        predictions = (test_scores >= probability_cutoff).astype(int)

        results.append({
            "machine_type": machine_type,
            "score_threshold": threshold,
            "probability_cutoff": probability_cutoff,
            "status": "ok",
            **counts,
            "Accuracy": accuracy_score(test[target], predictions),
            "Precision": precision_score(
                test[target], predictions, zero_division=0
            ),
            "Recall": recall_score(
                test[target], predictions, zero_division=0
            ),
            "F1": f1_score(
                test[target], predictions, zero_division=0
            ),
        })

        models[(machine_type, threshold)] = {
            "model": model,
            "probability_cutoff": probability_cutoff,
        }

result_df = pd.DataFrame(results)
print(result_df.to_string(index=False))
