#######################################################
#       Classifier : Decision tree
######################################################
# Import packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler


def main():

    # Log file
    logging.basicConfig(filename='log/decision_tree.log', level=logging.INFO, filemode='w')
    logging.getLogger().addHandler(logging.StreamHandler())

    # Read the input dataset
    input_file = 'input_dataset/breast-cancer.csv'
    df = pd.read_csv(input_file)
    logging.info('The columns of input csv : {}'.format(df.columns))


if __name__ == '__main__':
    main()
