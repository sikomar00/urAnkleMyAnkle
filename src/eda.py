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

df = pd.read_csv('data/raw/synthetic_industrial_machine_data.csv')
# df.info()
# df["timestamp"] = pd.to_datetime(df["timestamp"])
# df = df.sort_values("timestamp")
# print(df)
df.dropna(axis = 1,inplace = True)
df = df[['transaction_date','criticality','asset_tag','part_description','machine_type','temp_bearing_degC','temp_motor_degC','vibration_h_mms','vibration_v_mms','oil_pressure_bar','load_pct','shaft_rpm','power_consumption_kw','breakdown_flag']]
# df.info()
# print(df.columns)
# print(df.head())
# print(df['machine_type'].unique())
df1 = df[df['machine_type'] == 'CNC Lathe']
df1 = df1[df1['part_description'] == 'Deep Groove Ball Bearing 6205-2RS']
# print(df1.head())
df1["transaction_date"] = pd.to_datetime(df1["transaction_date"])
# print(df1.head())
target_data = 'breakdown_flag'
feature_data = ['criticality','temp_bearing_degC','temp_motor_degC','vibration_h_mms','vibration_v_mms','oil_pressure_bar','load_pct','shaft_rpm','power_consumption_kw']
criticality_weight = {
    "C": 1.0,
    "B": 2.0,
    "A": 4.0,
}

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
X = df1[feature_data]
y = df1[target_data]
X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state = 42,
    stratify = y
)

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

model = HistGradientBoostingClassifier(
    learning_rate = 0.05,
    max_iter = 200,
    max_depth = 10)

# model = LGBMClassifier(
    
# )
model.fit(X_train,y_train)
y_pred = model.predict(X_test)


cm = confusion_matrix(
    y_test,
    y_pred
)
print(cm)

accuracy = accuracy_score(
    y_test,
    y_pred
)
print("Accuracy:", accuracy)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)
recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)
f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)