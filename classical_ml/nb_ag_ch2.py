#%%
print("Welcome to chapter 2 of Ageron's Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow!")
#%%
import sys
assert sys.version_info >= (3, 7), "Python version must be >= 3.7"
# %%
from packaging import version
import sklearn
assert version.parse(sklearn.__version__) >= version.parse("1.0.1")
#%%
import pandas as pd
#%%
import kagglehub
def load_housing_data():
    # Download the latest version of a housing dataset from Kaggle
    path = kagglehub.dataset_download("camnugent/california-housing-prices")
    print("Path to dataset files:", sys.path)
    # Load the CSV file into a Pandas DataFrame
    return pd.read_csv(path + "/housing.csv")

housing = load_housing_data()

# below is code for data on local machine
# data_path = '../data/housing.csv'
# housing = pd.read_csv(data_path)
housing.describe()  # describe() method shows a quick statistic summary of the data
housing.head()
#%%
housing.info() # The info() method is useful to get a quick description of the data, in particular the total number of rows, each attribute’s type, and the number of non-null values
housing["ocean_proximity"].value_counts() # find out what categories exist and how many districts belong to each category by using the value_counts() method
# %%
import matplotlib.pyplot as plt
housing.hist(bins = 50, figsize=(24,16)) # call the hist() method on the whole dataset and it will plot a histogram for each numerical attribute
plt.show()
#%%
# Creating a test set is theoretically simple; pick some instances randomly,
# typically 20% of the dataset (or less if your dataset is very large), and set them aside:
 
import numpy as np
def shuffle_and_split_data(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data)) #randomizes the indices of data
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices] 

train_set, test_set = shuffle_and_split_data(housing, 0.2) # this function will create different test and train set every single time the program is run
len(train_set)
#%%
len(test_set)
