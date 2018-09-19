'''
Sagara Sumathipala
@date: 19-09-2018
'''

import pandas as pd
import numpy as np
import scipy

import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import logging
#logging.basicConfig(filename='../logs/k_mode.log')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

# Set pandas environment settings:
pd.set_option('display.precision', 2)
pd.options.display.float_format = "{:.4f}".format
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

df = pd.read_csv('cluster_data.csv')

# Basic Information of the DataFrame
logger.info("Shape of the DataFrame: " + str(df.shape) + '\n')
logger.info("Head of the DataFrame: \n" + str(df.head())+ '\n')
logger.info("Data Types of the DataFrame: \n" + str(df.dtypes)+ '\n')

##
# PRE-PROCESSING
##
def do_preprocessing(df):
    '''
    - Find and eliminate duplicates values
    - Find and eliminate the null and all zero values in rows
    :param df: DataFrame
    :return: Cleaned DataFrame
    '''

    logger.info("\nDisplay Null Row Counts: \n" + str(df.isna().sum())+ '\n')
    for column in df.columns[1:]:
        if (df[column].isna().sum() > 0):
            # print(df[column].name)
            df = df.dropna()
            logger.debug("\nDisplay Null Row Counts: \n" + str(df.isna().sum())+ '\n')

    # Drop Index column
    #df1 = df
    df = df[df.columns[1:]]
    logger.info("\nData Types of the Pre-Processed DataFrame: \n" + str(df.dtypes)+ '\n')

    '''
    Remove rows with all values are ZERO
    '''
    df = df[(df.T != 0).any()]
    logger.info("\nDataFrame After remove all zeoro rows: \n" + str(df.shape)+ '\n')

    return df

def log10_df(df):
    '''
    Return the log10 value of all the values in the DataFrame
    '''
    df = df + 1
    df = np.log10(df)
    logger.info("\nHead of the DataFrame After Log10: \n" + str(df.head())+ '\n')
    return df

def scale_df(df):
    scaler = MinMaxScaler()
    scaler.fit(df)
    logger.info("\nMinMaxScaler Data Max: \n" + str(scaler.data_max_)+ '\n')
    df = pd.DataFrame(scaler.transform(df))
    logger.info("\nSample of the DataFrame After MinMaxScaler: \n" + str(df.sample(5))+ '\n')
    logger.info("\nShape of the DataFrame After MinMaxScaler: \n" + str(df.shape) + '\n')
    return df

def corr_df(df, corr_val):
    '''
    Obj: Drops features that are strongly correlated to other features.
          This lowers model complexity, and aids in generalizing the model.
    Inputs:
          df: features df (x)
          corr_val: Columns are dropped relative to the corr_val input (e.g. 0.8)
    Output: df that only includes uncorrelated features
    '''

    # Creates Correlation Matrix and Instantiates
    corr_matrix = df.corr().abs()
    logger.info("\nCorrelation Matrix between Parameters:\n" + str(corr_matrix)+ '\n')
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    # Iterates through Correlation Matrix Table to find correlated columns
    for i in iters:
        for j in range(i):
            item = corr_matrix.iloc[j:(j + 1), (i + 1):(i + 2)]
            col = item.columns
            row = item.index
            val = item.values
            if abs(val) >= corr_val:
                # Prints the correlated feature set and the corr val
                print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                drop_cols.append(i)

    drops = sorted(set(drop_cols))[::-1]

    # Drops the correlated columns
    for i in drops:
        col = df.iloc[:, (i + 1):(i + 2)].columns.values
        df = df.drop(col, axis=1)

    logger.info("\nData Types of the DataFrame after remove out Correlations: \n" + str(df.dtypes)+ '\n')
    logger.info("\nShape of the DataFrame of filter out Correlations: \n" + str(df.shape)+ '\n')
    return df


def do_pca(df, n_components=2):
    pca = PCA(n_components=n_components)
    pca.fit(df)
    logger.info('\nPercentage of variance explained by each of the selected components.\n' +
                str(pca.explained_variance_ratio_) + '\n')
    logger.info('\nThe singular values corresponding to each of the selected components.\n' +
                str(pca.singular_values_) + '\n')
    X_reduced = pca.transform(df)
    logger.info("\nShape of the PCA DataFrame: \n" + str(X_reduced.shape) + '\n')

    X_pca = pd.DataFrame(X_reduced, columns=['PC1', 'PC2'])  # PC=principal component
    logger.info("\nSample of the PCA DataFrame: \n" + str(X_pca.sample(5)) + '\n')

    return X_reduced, X_pca


def create_clusters(df, n_components=3, n_clusters=5, isPlot=True, plotSamplesize=1000, saveClusterData=False):
    X_reduced, X_pca = do_pca(df, n_components)

    kmeans = KMeans(n_clusters).fit(X_reduced)
    global labels
    labels = kmeans.predict(X_reduced)

    global centroids
    centroids = kmeans.cluster_centers_
    logger.info("Cluster Centroids:\n"+str(centroids)+ '\n')

    if isPlot:
        X_pca['ClusterKmeans'] = labels
        df2 = X_pca.sample(plotSamplesize)
        fig = plt.figure()
        for i in range(0, n_clusters):
            x = df2.loc[df2['ClusterKmeans'] == i]
            #print(x)
            pc1 = x['PC1'].values
            pc2 = x['PC2'].values
            plt.scatter(pc1, pc2, label='cluster '+str(i), s=7, cmap='viridis')
        lgd = plt.legend(loc='upper center', bbox_to_anchor=(1.2, 1.0), frameon=True,
          fancybox=True, shadow=True)
        plt.grid(True)
        plt.title('Customer Segments using K-Means Clusters')
        plt.xlabel('Principle Component 1')
        plt.ylabel('Principle Component 2')
        plt.savefig('Cluster_Visualization_'+str(n_clusters)+'_s_'+'.png', dpi=300, format='png',
                    bbox_extra_artists=(lgd,), bbox_inches='tight')  # save the figure to file
        plt.close()

    if saveClusterData:
        df['Cluster'] = labels
        df.to_csv('Clustered_data.csv')


df = do_preprocessing(df)
df = corr_df(df, 0.8)
df = log10_df(df)
df = scale_df(df)
create_clusters(df, n_components=2, n_clusters=3, isPlot=True, plotSamplesize=4000, saveClusterData=False)