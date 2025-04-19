import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import os

data_path = os.path.join('data', 'multivariate_unemployment_LSTNet.csv')
df = pd.read_csv(data_path, parse_dates=['date'])

#set date as the index
df.set_index('date', inplace=True)

#persistence forecast simulation: we are predicting each "next" value as the last

horizon = 7 #forecast 7 days ahead
results = {}

for col in df.columns:
    y_true = df[col].shift(-horizon).dropna()
    y_pred = df[col].iloc[:-horizon]
    y_true = y_true.loc[y_pred.index]

    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5  #manually calculate RMSE by taking sqrt, I can fix this later

    results[col] = {'MAPE': round(mape * 100, 2), 'RMSE': round(rmse, 4)}

    #results

    print("Persistence Model Baseline Performance: \n")
    for metric in results:
        print(f"{metric}: MAPE = {results[metric]['MAPE']}%, RMSE = {results[metric]['RMSE']}")
