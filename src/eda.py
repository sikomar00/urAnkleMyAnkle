import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


df = pd.read_csv('data/raw/synthetic_industrial_machine_data.csv')
# df.info()
# df["timestamp"] = pd.to_datetime(df["timestamp"])
# df = df.sort_values("timestamp")
# print(df)
df.dropna(axis = 1,inplace = True)
df = df[['transaction_date','asset_tag','machine_type','temp_bearing_degC','temp_motor_degC','vibration_h_mms','vibration_v_mms','oil_pressure_bar','load_pct','shaft_rpm','power_consumption_kw','breakdown_flag']]
df.info()
print(df.columns)
print(df.head())
print(df['machine_type'].unique())
df1 = df[df['machine_type'] == 'CNC Lathe']
print(df1.head())
df1["transaction_date"] = pd.to_datetime(df1["transaction_date"])
print(df1.head())
target_data = 'breakdown_flag'
feature_data = ['machine_type','temp_bearing_degC','temp_motor_degC','vibration_h_mms','vibration_v_mms','oil_pressure_bar','load_pct','shaft_rpm','power_consumption_kw','breakdown_flag']


