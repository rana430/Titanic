import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder  # OneHotEncoder
import seaborn as sns
from pandas.plotting import scatter_matrix
import os

# Display all columns in PD
desired_width = 420
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 20)


# Read diamonds.csv as a dataframe .


def load_diamonds_data(diamonds_path):
    csv_path = os.path.join(diamonds_path, "diamonds.csv")
    return pd.read_csv(csv_path)


if __name__ == '__main__':
    diamond = load_diamonds_data('')

    # Check the head of diamond, and check out its info() and describe() methods.
    print(diamond.head())

    print(diamond.info())

    # Drop Colums 0
    diamond.drop(diamond.columns[0], axis=1, inplace=True)

    # find Duplicated Data and Remove it
    print(diamond.duplicated())
    diamond.drop_duplicates(inplace=True)

    print(diamond.info())

    # displays the total count of missing values per column.
    print(diamond.isna().sum())

    # choose Features from dataset
    diamond_features = diamond.loc[:, ['carat', 'depth', 'table', 'x', 'y', 'z']]

    # our Target "Price"
    target = diamond.loc[:, 'price']

    # create PLT list and create Correlation Graph
    plt_list = []
    for i in diamond_features:
        plt_list.append(i)

    for i in plt_list:
        plt.scatter(diamond_features[i], target)
        plt.title('Correlation Graph')
        plt.xlabel(i)
        plt.ylabel('Price')
        plt.show()

    # convert the correlation matrix to a heatmap plot
    plt.figure(figsize=(12,8))
    sns.heatmap(diamond.corr(), annot=True, cmap='viridis')
    plt.show()

    #  check for correlation between attributes is to use the pandas scatter_matrix()
    scatter_matrix(diamond[plt_list], figsize=(12, 8))
    plt.show()

    # find if we have group in columns
    X = diamond.loc[:, ['carat', 'cut', 'color', 'clarity', 'depth', 'table']]
    X_array = np.array(X)

    print(np.unique(X_array[:, 1]))
    print(np.unique(X_array[:, 2]))
    print(np.unique(X_array[:, 3]))

    # encode Group Columns using OrdinalEncoder
    encoder = OrdinalEncoder(
        categories=[['Fair', 'Good', 'Very Good', 'Ideal', 'Premium'], ['J', 'I', 'H', 'G', 'F', 'E', 'D'],
                    ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']])
    X_array[:, 1:4] = encoder.fit_transform(X_array[:, 1:4])

    # convert x_array to DataFrame
    X = pd.DataFrame(X_array)
    print(X)

    # Split Train and Test  Data
    X_array = np.array(X)
    Y_array = np.array(target)

    X_train, X_test, Y_train, Y_test = train_test_split(X_array, Y_array, test_size=0.2, random_state=42)
