import pandas as pd
import numpy as np
import pyarrow.parquet as pq

# Load data
df = pd.read_csv('data/active/copy_test_data.csv')

# Parse MONTH_KEY to integer month number
df['MONTH_NUM'] = pd.to_datetime(df['MONTH_KEY']).dt.month

# Drop original columns MONTH_KEY and ORDER_ID
df = df.drop(columns=['MONTH_KEY', 'ORDER_ID'])

# Persist engineered feature set to parquet
df.to_parquet('artifacts/engineered_features.parquet', index=False)

print(f"STATE shape={df.shape} columns={list(df.columns)}")