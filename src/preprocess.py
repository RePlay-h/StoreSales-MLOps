
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import yaml

def preprocess(stores_path, oil_path, holidays_path, transactions_path, train_path):
    
    ## Read csv-files
    stores = pd.read_csv(stores_path)
    oil = pd.read_csv(oil_path)
    holidays = pd.read_csv(holidays_path)
    transactions = pd.read_csv(transactions_path)
    train = pd.read_csv(train_path)

    ## Change column name because they refflect different features
    stores = stores.rename(columns = {'type': 'store_type'})
    holidays = holidays.rename(columns={'type': 'holiday_type'})

    data = train.merge(oil, on='date', how='left')
    data = data.merge(stores, on='store_nbr', how='left')

    holidays = holidays[holidays['transferred'] == False]
    data = data.merge(holidays[['date', 'holiday_type', 'locale']], on='date', how='left')
    data['is_holiday'] = data['holiday_type'].notnull().astype(int)

    data = data.merge(transactions, on=['date', 'store_nbr'], how='left')

    ## Fill NaNs
    data.fillna({
        'holiday_type': 'Work Day',
        'transactions': 0,
        'locale': 'Not Holiday'
    }, inplace=True)

    data['dcoilwtico'] = data['dcoilwtico'].interpolate(method='linear', limit_direction='both')

    ## Create time-based features
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day_of_week'] = data['date'].dt.dayofweek
    data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)

    ## Feature Engineering: Store and Family lags & rolling-window statistics
    group_cols = ['store_nbr', 'family']
    g = data.groupby(group_cols)['sales']

    ## shift_1 - sales on the previous day
    ## roll_mean_7 - 7-days moving average of sales
    ## roll_std_7 - 7-days moving standard deviation of sales 
    data['shift_1'] = g.transform(lambda x: x.shift(1))
    data['roll_mean_7'] = g.transform(lambda x: x.shift(1).rolling(window=7).mean())
    data['roll_std_7'] = g.transform(lambda x: x.shift(1).rolling(window=7).std())

    data = data.dropna(subset=['shift_1', 'roll_mean_7', 'roll_std_7']).reset_index(drop=True)

    ## Delete useless features
    to_delete = ['id', 'date', 'is_weekend']
    data = data.drop(to_delete, axis=1)

    ## Encode string features
    label_enc = LabelEncoder()
    for col in data.select_dtypes(include='object').columns:
        data[col] = label_enc.fit_transform(data[col].astype(str))

    return data


if __name__ == "__main__":

    ## Read params to load DataFrames
    params = yaml.safe_load(open("params.yaml"))['preprocess']
    input_path = params['input']

    ## Get processed dataset
    df = preprocess(input_path['stores'], 
               input_path['oil'], 
               input_path['holidays_events'], 
               input_path['transactions'], 
               input_path['train'])
    
    ## Save dataset
    output_path = params['output']['data']
    df.to_csv(output_path, index=False)

    print(f'Processed dataset: {output_path}')
    