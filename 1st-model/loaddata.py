# -*- coding: utf-8 -*-
import logging
import pandas as pd
import os

PARENT_FOLDER = '../'
INPUT_PATH = os.path.join(PARENT_FOLDER,'data/raw/train.csv')
SUBMISSION_PATH = os.path.join(PARENT_FOLDER,'data/raw/submission_format.csv')
OUTPUT_PATH = os.path.join(PARENT_FOLDER,'data/processed/')

def meter_data_to_csv(dataframe, meter_id):
    tmp = dataframe.loc[dataframe.meter_id == meter_id]
    subm = pd.read_csv(SUBMISSION_PATH)
    if meter_id.startswith("38"):
        meter = pd.merge(subm.loc[subm.meter_id == "38_9686"], tmp[["Timestamp", "Values"]], how='left', on="Timestamp")
    else:
        meter = pd.merge(subm.loc[subm.meter_id == meter_id], tmp[["Timestamp", "Values"]], how='left', on="Timestamp")
    meter.to_csv(OUTPUT_PATH + meter_id + ".csv", index=False)

def main():
    logger = logging.getLogger(__name__)
    logger.info("提取各区电量")

    train = pd.read_csv(INPUT_PATH).iloc[:,1:]

    meter_id1 = '234_203'
    meter_data_to_csv(train, meter_id1)


    meter_id2 = '334_61'
    meter_data_to_csv(train, meter_id2)


    meter_demand = '38_9687'
    meter_data_to_csv(train, meter_demand)
    meter_reactive = '38_9688'
    meter_data_to_csv(train, meter_reactive)

if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()