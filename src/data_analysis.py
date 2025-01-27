import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import argparse


def main(input_file):

    #Log file
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
    filename='log/data_analysis' + date_time + '.log'
    logging.basicConfig(filename=filename, level=logging.INFO, filemode='w', format='%(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())

    # Tp suppress future warning from Seaborn
    #warnings.simplefilter(action='ignore', category=FutureWarning)
    print(f'Input file: {input_file}')
    df = pd.read_csv(input_file)
    ###################################
    # Understanding the data
    ###################################
    logging.info(f"Shape of df: {df.shape}")
    logging.info(f"Columns in df: {df.columns}")
    # Print first 5 rows
    print(df.head())
    # Print random 5 rows
    print(df.sample(5))
    # Check data types of columns
    print(df.info())
    # Check for missing values
    print(df.isnull().sum())
    # Check statistical summary of df
    logging.info('Statistical summary of df')
    logging.info(df.describe().to_string())
    # Check if there are any duplicates
    print(df.duplicated().sum())
    # Correlation between columns
    X = df.drop(['id', 'diagnosis'], axis=1).copy()
    logging.info('Correlation of input features')
    logging.info(X.corr().to_string())

    ####################################

    # Check if there are any duplicate rows
    duplicates = df[df.duplicated()]
    logging.info('Number of duplications: {} '.format(duplicates.shape[0]))


    # To check if plots directory exists or not
    save_dir = "plots"
    # Check whether the specified directory exists or not
    isExist = os.path.exists(save_dir)
    if not isExist:
        os.makedirs(save_dir)
        print(f'{save_dir} directory created')

    # Diadnosis count plot
    plt.figure(figsize=(8, 8))
    sns.countplot(data=df, x='diagnosis')
    save_diagnosis_countplot_path = save_dir + '/' + 'diagnosis_historam.png'
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.title('Histogram of diagnosis')
    plt.savefig(save_diagnosis_countplot_path)

    # Pie chart
    target_counts = df['diagnosis'].value_counts()
    # Create the pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(target_counts, labels=target_counts.index, autopct='%1.1f%%', startangle=140)
    save_piechart_path = save_dir + '/' + 'diagnosis_pie_chart.png'
    plt.title('Diagnosis Pie Chart')
    plt.savefig(save_piechart_path)

    '''
    # Analyze the age distribution
    plt.figure(figsize=(8, 8))
    plt.hist(df['estimated_age'], bins=10)
    plt.xlabel('estimated_age')
    plt.ylabel('count')
    plt.title('Histogram of estimated_age')
    save_histogram_path = save_dir + '/' + 'age_histogram.png'
    plt.savefig(save_histogram_path)

    # Statistical analysis
    logging.info('Analyze the statistical data')
    logging.info(df['estimated_age'].describe())
    lower_bound, upper_bound = get_outlier_bounds(df['estimated_age'])
    logging.info('Upper bound of  estimated_age: {}'.format(upper_bound))

    # Box plot
    plt.figure(figsize=(8, 8))
    sns.boxplot(df['estimated_age'])
    save_boxplot_path = save_dir + '/' + 'boxplot.png'
    plt.title('Box Plot')
    plt.savefig(save_boxplot_path)

    # Bivariate analysis
    plt.figure(figsize=(8, 8))
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    sns.stripplot(data=df, x='domain', y='estimated_age', jitter=True, hue='domain', palette='Set2', marker='o', size=8)
    save_stripplot_path = save_dir + '/' + 'stripplot.png'
    plt.title('Strip plot between domain and estimated_age')
    plt.savefig(save_stripplot_path)
    '''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', type=str,
                        required=True, help="input dataset")
    args = parser.parse_args()
    print(args.input_file)
    main(args.input_file)