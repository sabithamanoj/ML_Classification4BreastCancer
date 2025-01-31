#####################################################################################
#           Comparison of different ML models
#           To Predict             : Breast Cancer
###################################################################################

# Import packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
import os
import argparse


def main(input_file):

    # Log file
    path = "log"
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("The new directory 'results' is created!")
    from datetime import datetime
    now = datetime.now()
    date_time = now.strftime("%m%d%Y_%H%M%S")
    filename='log/training' + date_time + '.log'
    logging.basicConfig(filename=filename, level=logging.INFO, filemode='w', format='%(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())

    print(f'Input file: {input_file}')
    df = pd.read_csv(input_file)
    ###################################
    # Understanding the data
    ###################################
    logging.info(f"Shape of df: {df.shape}")
    logging.info(f"Columns in df: {df.columns}")
    # Print random 5 rows
    print(df.sample(5))
    # Check data types of columns
    logging.info('df.info()')
    logging.info(df.info())
    # Check statistical summary of df
    logging.info('Statistical summary of df')
    logging.info(df.describe().to_string())

    # Check for missing values
    logging.info(f"Number of missing values: ")
    logging.info(df.isnull().sum())
    # Check if there are any duplicates
    logging.info(f" Number of duplicates: {df.duplicated().sum()}")

    # Check if there are any duplicated ids
    duplicated_ids = df[df['id'].duplicated(keep=False)]
    # This creates a boolean Series that marks all duplicates as True, including the first occurrence.
    # keep=False means it marks all occurrences as duplicates, not just the subsequent ones
    print(f"Check for duplicated_ids : {duplicated_ids}")

    X = df.drop(['id', 'diagnosis'], axis=1).copy()

    # Correlation of input features
    logging.info('Correlation of input features')
    logging.info(X.corr().to_string())

    y = df['diagnosis'].copy()
    # check class distribution
    logging.info('***** Class distribution ***** ')
    logging.info(y.value_counts())







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', type=str,
                        required=True, help="input dataset")
    args = parser.parse_args()
    print(args.input_file)
    main(args.input_file)


