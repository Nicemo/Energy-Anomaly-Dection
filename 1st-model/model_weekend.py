# -*- coding: utf-8 -*-
import pandas as pd
import os
from sklearn.ensemble import IsolationForest
from scipy import stats

PARENT_FOLDER = "../"
DATA_PATH = os.path.join(PARENT_FOLDER,'data/processed/')
INPUT_PATH = os.path.join(PARENT_FOLDER,'data/raw/')
RESULT_PATH = os.path.join(PARENT_FOLDER,'data/result/')

def read_data(file_path):
    dateparse = lambda date: pd.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    df = pd.read_csv(file_path, parse_dates=['Timestamp'], index_col='Timestamp', date_parser=dateparse)
    return df

def extract_features(meter_id, df, weather):
    if meter_id == "38_9687":
        pivot_data = extract_features_meter3(df, weather)

    else:
        ts = df.loc[df.index.weekday > 4, "Values"]
        tmp = ts.groupby([ts.index.date, ts.index.hour]).mean().reset_index()
        tmp.columns = ["date", "hour", "Values"]
        pivot_data = tmp.pivot_table(index="date", columns="hour", values="Values")

        pivot_data["area_middle"] = (pivot_data.iloc[:, 9:24].T - pivot_data.iloc[:, 9:24].T.min()).sum()
        pivot_data["area_all"] = (pivot_data.iloc[:, 0:24].T - pivot_data.iloc[:, 0:24].T.min()).sum()
        pivot_data["area_percent"] = pivot_data["area_middle"] / pivot_data["area_all"]
        pivot_data["max_hour"] = pivot_data.iloc[:, 9:24].T.idxmax()
        pivot_data["range"] = pivot_data.iloc[:, 9:24].T.max() - pivot_data.iloc[:, 9:24].T.min()

        empirical = pivot_data["demand"].iloc[:, 9:24].mean()
        KL_transform = lambda x: stats.entropy(empirical, x)
        pivot_data["KL"] = pivot_data.iloc[:, 9:24].transform(KL_transform)
        pivot_data["temperature"] = weather

        return pivot_data.dropna()


def extract_features_meter3(df, weather):
    tmp = df.groupby([df.index.date, df.index.hour]).mean().reset_index()
    tmp.columns = ["date", "hour", "demand", "reactive"]
    pivot_data = tmp.pivot_table(index="date", columns="hour", values=["demand", "reactive"]).dropna()
    pivot_data["area_demand"] = (pivot_data.loc[:, "demand"].T - pivot_data.loc[:, "demand"].T.min()).sum()
    pivot_data["area_reactive"] = (pivot_data.loc[:, "reactive"].T - pivot_data.loc[:, "reactive"].T.min()).sum()
    pivot_data["area_percent_reactive"] = pivot_data['area_reactive'] / pivot_data['area_demand']
    pivot_data['reactive_max_hour'] = pivot_data.loc[:, 'reactive'].T.idxmax()

    pivot_data = pivot_data.drop(['reactive'], axis=1)
    pivot_data['temperature'] = weather
    pivot_data['area_middle'] = (pivot_data.iloc[:, 9:24].T - pivot_data.iloc[:, 9:24].T.min()).sum()
    pivot_data['area_all'] = (pivot_data.iloc[:, 0:24].T - pivot_data.iloc[:, 0:24].T.min()).sum()
    pivot_data['area_percent_middle'] = pivot_data['area_middle'] / pivot_data['area_all']
    pivot_data['range'] = pivot_data.loc[:, 'demand'].iloc[:, 9:24].T.max() - pivot_data.loc[:, 'demand'].iloc[:,9:24].T.min()

    empirical = pivot_data["demand"].iloc[:, 9:24].mean()
    KL_transform = lambda x: stats.entropy(empirical, x)
    pivot_data["KL"] = pivot_data.iloc[:, 9:24].transform(KL_transform)

    return pivot_data

def iforest_detection(df):
    iforest = IsolationForest(n_estimators=100, contamination=0.2, random_state=33)
    iforest.fit(df)
    pred = iforest.decision_function(df)
    df["pred"] = pred
    return df.sort_values(by="pred", ascending=False)

def get_anomalies(meter_id, df):

    rule1 = (df["pred"] < 0) & (df["area_middle"] > df["area_middle"].mean()) & (df.range > df.range.quantile(0.75))
    rule2 = (df["area_reactive"] > df["area_reactive"]) & rule1
    filter_rule = rule1 if meter_id != "38_9687" else rule2
    anomaly = df.loc[filter_rule]

    return anomaly

def pipeline(meter_id, df, weather):
    weather = weather.resample('D').mean()
    pivot_data = extract_features(meter_id, df, weather)
    pivot_data = iforest_detection(pivot_data)
    anomalies = get_anomalies(meter_id, pivot_data)
    return anomalies.index.astype('str').tolist()

def get_holiday_anomalies_meter2(df,holidays):

    reference_date = '2016-01-06'
    KL_dict = {}
    for date in holidays['Date']:
        try:
            if not df[date].empty:
                KL_dict[date] = stats.entropy(df[reference_date],df[date])
        except:
            pass
    return KL_dict

def get_holiday_anomalies_meter3(meter_demand_id,df,holidays,weather):

    df_holidays = pd.DataFrame()
    for i in range(len(holidays)):
        try:
            df_holidays = df_holidays.append(df.loc[holidays['Date'].iloc[i]])
        except:
            pass
    anomalies = pipeline(meter_demand_id, df_holidays, weather)
    return anomalies

def mark_anomalies(df,anomalies):

    df['is_abnormal'] = False
    for date in anomalies:
        df.loc[date,'is_abnormal'] = True
    return df.reset_index()[['obs_id', 'meter_id','Timestamp','is_abnormal']]

def main():
    weather = read_data(INPUT_PATH + 'weather.csv')
    holidays = pd.read_csv(INPUT_PATH + 'holidays.csv').iloc[:, 1:]

    # meter1, 234_203
    meter_id1 = '234_203'
    print('Detecting meter1 weekend anomalies...')
    df1 = read_data(DATA_PATH + meter_id1 + '.csv')
    weather_site1 = weather[weather.site_id == meter_id1].iloc[:, 1:].fillna(method='ffill')[['Temperature']]

    anomalies1 = pipeline(meter_id1, df1[:'2015-06-20'], weather_site1)
    res1 = mark_anomalies(df1, anomalies1)

    # meter2, 334_61
    meter_id2 = '334_61'
    print('Detecting meter2 weekend anomalies...')
    df2 = read_data(DATA_PATH + meter_id2 + '.csv')
    weather_site2 = weather[weather.site_id == meter_id2].iloc[:, 1:].fillna(method='ffill')[['Temperature']]
    anomalies2 = pipeline(meter_id2, df2, weather_site2)

    print('Detecting meter2 holiday anomalies...')
    holidays2 = holidays[holidays.site_id == meter_id2]
    KL = get_holiday_anomalies_meter2(df2['Values'].resample('H').mean().dropna(), holidays2)
    holiday_anomalies2 = sorted(KL.items(), key=lambda x: x[1])[0:6]

    anomalies2.extend([i[0] for i in holiday_anomalies2])
    res2 = mark_anomalies(df2, anomalies2)

    # meter3, 38_9687
    meter_demand_id = '38_9687'
    meter_reactive_id = '38_9688'
    print('Detecting meter3 weekend anomalies...')
    df3 = read_data(DATA_PATH + meter_demand_id + '.csv')
    demand = df3[['Values']]
    demand.columns = ['demand']
    reactive = read_data(DATA_PATH + meter_reactive_id + '.csv')[['Values']]
    reactive.columns = ['reactive']
    reactive['reactive'] = reactive['reactive'].diff(1)
    df_combine = pd.merge(demand, reactive, right_index=True, left_index=True)[['demand', 'reactive']]
    weather_site3 = weather[weather.site_id == '38'].iloc[:, 1:].fillna(method='ffill')[['Temperature']]
    # separate into two parts as there exists an obvious turning point on reactive meter curve
    df3_part1 = df_combine[:'2012-12-21']
    df3_part2 = df_combine['2012-12-21':]
    anomalies3_part1 = pipeline(meter_demand_id, df3_part1.loc[df3_part1.index.weekday > 4], weather_site3)
    anomalies3_part2 = pipeline(meter_demand_id, df3_part2.loc[df3_part2.index.weekday > 4], weather_site3)

    print('Detecting meter3 holiday anomalies...')
    holidays3 = holidays[holidays.site_id == '038']
    holiday_anomalies3 = get_holiday_anomalies_meter3(meter_demand_id, df_combine, holidays3, weather_site3)

    anomalies3 = anomalies3_part1 + anomalies3_part2 + holiday_anomalies3
    res3 = mark_anomalies(df3, anomalies3)

    # combine results of the three meters
    print('Combining weekend result...')
    res = res1.append(res2).append(res3)
    res.to_csv(RESULT_PATH + 'res_weekend_with_holiday.csv', index=False)

if __name__ == "__main__":
    main()